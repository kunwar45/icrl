# ABOUTME: Runs one browser episode: config-templated prompts → LLM → parsed action →
# ABOUTME: env.step, with loop/stagnation recovery. Returns a raw EpisodeResult dict.
"""
The episode loop is benchmark-agnostic: everything env-specific comes through
the BenchmarkAdapter, everything run-specific (prompts, model, limits) through
the config. Robustness features carried over from the first collection
campaign, each of which was added because its absence lost real episodes:

  - duplicate send_msg_to_user blocked (episodes stalled waiting for replies)
  - repeated-action detection → hint, then forced re-generation
  - URL stagnation detection → navigation hint
  - env.step wrapped so a single bad action logs instead of killing the episode
"""
from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

_REPEAT_HINT_AFTER = 3      # identical actions in a row before hinting
_STAGNATION_HINT_AFTER = 5  # steps on one URL before hinting


def _render(template: str, fields: dict[str, str]) -> str:
    """format_map that leaves unknown placeholders visible instead of crashing."""
    class _Default(dict):
        def __missing__(self, key):  # noqa: ANN001
            return "{" + key + "}"
    return template.format_map(_Default(fields))


def _llm_action(client, model_cfg: dict, prompt_cfg: dict, fields: dict[str, str],
                adapter, temperature: float, hint: Optional[str]) -> tuple[str, Optional[str]]:
    """One LLM call → (raw_output, parsed_action_or_None)."""
    fields = dict(fields, hint_block=f"\n# Hint from system\n{hint}" if hint else "")
    messages = [
        {"role": "system", "content": _render(prompt_cfg["system"], fields)},
        {"role": "user", "content": _render(prompt_cfg["user"], fields)},
    ]
    response = client.chat.completions.create(
        model=model_cfg["name"],
        messages=messages,
        max_tokens=int(model_cfg.get("max_tokens", 512)),
        temperature=temperature,
    )
    raw = (response.choices[0].message.content or "").strip()
    return raw, adapter.parse_action(raw)


def run_episode(adapter, client, cfg: dict, task_id: int | str,
                temperature: float, extra_fields: Optional[dict] = None) -> dict:
    """
    Run one episode of `task_id`; never raises for in-episode failures.

    extra_fields are merged into the prompt fields every step — this is how the
    trajectory-generation pipeline injects its refined {plan} while reusing this
    loop. The runner also provides {actions_so_far} (the episode's own action
    history) to every template.

    Returns: {task_id, reward, terminated, truncated, n_steps, steps,
              policies, safety_report, violated_policies, error}
    """
    model_cfg = cfg["model"]
    prompt_cfg = cfg["prompt"]
    max_steps = int(cfg["episode"]["max_steps"])

    result: dict[str, Any] = {
        "task_id": task_id, "reward": 0.0, "terminated": False, "truncated": False,
        "n_steps": 0, "steps": [], "policies": [], "safety_report": [],
        "violated_policies": [], "error": None,
    }

    env = None
    try:
        env = adapter.make_env(task_id)
        obs = adapter.reset(env)
        result["policies"] = getattr(adapter, "raw_policies", lambda o: [])(obs)

        total_reward, terminated, truncated = 0.0, False, False
        info: dict = {}
        steps: list[dict] = []
        sent_user_message = False
        last_action: Optional[str] = None
        repeat_count = 0
        last_url: Optional[str] = None
        stagnation_count = 0

        for step_idx in range(max_steps):
            fields = adapter.prompt_fields(obs)
            fields["actions_so_far"] = "\n".join(s["action"] for s in steps) or "(none yet)"
            if extra_fields:
                fields.update(extra_fields)

            url = fields.get("url", "")
            stagnation_count = stagnation_count + 1 if url == last_url else 0
            last_url = url

            hint = None
            if stagnation_count >= _STAGNATION_HINT_AFTER:
                hint = (f"WARNING: {stagnation_count} steps on {url} with no progress. "
                        "Try goto() to reach the record directly, or an Actions/kebab "
                        "menu you have not clicked yet.")
            elif repeat_count >= _REPEAT_HINT_AFTER:
                hint = (f"WARNING: '{last_action}' repeated {repeat_count} times with no "
                        "effect. Click a different element or use goto().")

            raw, action = _llm_action(client, model_cfg, prompt_cfg, fields,
                                      adapter, temperature, hint)
            if action is None:
                logger.warning("task %s step %d: no parsable action in %r",
                               task_id, step_idx, raw[:200])
                action = "noop()"

            if action.startswith("send_msg_to_user"):
                if sent_user_message:
                    # A second notification stalls the episode; policies require
                    # exactly one. Substitute a noop and move on.
                    action = "noop()"
                else:
                    sent_user_message = True

            repeat_count = repeat_count + 1 if action == last_action else 0
            last_action = action

            try:
                obs, reward, terminated, truncated, info = adapter.step(env, action)
            except Exception as e:
                logger.warning("task %s step %d: env.step(%r) failed: %s",
                               task_id, step_idx, action, e)
                steps.append({"step_idx": step_idx, "action": action,
                              "observation": fields.get("axtree", ""), "step_error": str(e)})
                continue

            total_reward += reward
            steps.append({"step_idx": step_idx, "action": action,
                          "observation": fields.get("axtree", "")})
            if terminated or truncated:
                break

        report = adapter.safety_report(info)
        result.update({
            "reward": total_reward,
            "terminated": terminated,
            "truncated": truncated,
            "n_steps": len(steps),
            "steps": steps,
            "safety_report": report,
            "violated_policies": [r for r in report if r.get("violated")],
        })

    except Exception as e:
        result["error"] = str(e)
        logger.warning("task %s: episode error: %s", task_id, e)
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass

    return result
