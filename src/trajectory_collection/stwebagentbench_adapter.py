# ABOUTME: ST-WebAgentBench adapter — BrowserGym env over SuiteCRM with per-policy
# ABOUTME: ground-truth safety verdicts. The only file that imports browsergym/stwebagentbench.
"""
Wraps browsergym/STWebAgentBenchEnv.<task_id> for the collection engine.

Compute Canada notes baked in:
  - Playwright needs --no-sandbox (user namespaces disabled on compute nodes)
    and --disable-dev-shm-usage (/dev/shm undersized in cgroups).
  - The env reaches SuiteCRM via WA_SUITECRM (login-node Apptainer instance);
    this module never boots the web app.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from src.trajectory_data.browser_actions import extract_action, normalize_safety_report, strip_kwargs
from src.trajectory_collection.benchmark_adapter import BenchmarkAdapter

logger = logging.getLogger(__name__)

# Chromium flags required on SLURM compute nodes.
_PLAYWRIGHT_ARGS = ["--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu"]


class STWebAgentBenchAdapter(BenchmarkAdapter):
    name = "stwebagentbench"

    def __init__(self, benchmark_cfg: dict):
        super().__init__(benchmark_cfg)
        # Built once: BrowserGym renders custom actions with inspect.getsource(),
        # which is why build_action_set defines answer() at module level.
        from src.environments.stwebagentbench_environment import build_action_set
        self._action_set = build_action_set(multiaction=False)
        self._action_space_description = self._action_set.describe(
            with_long_description=False, with_examples=True
        )

    # ── Environment lifecycle ────────────────────────────────────────────────

    def make_env(self, task_id: int | str, max_steps: int | None = None) -> Any:
        import gymnasium as gym
        import browsergym.stwebagentbench  # noqa: F401 — registers the envs

        # The benchmark's GenericWebArenaTask terminates every episode at its
        # own max_steps (default 20) regardless of the caller's loop bound, so
        # the config's episode.max_steps must be pushed into the task itself.
        # gym.make replaces the registered task_kwargs dict wholesale — task_id
        # must be re-supplied alongside the override.
        task_kwargs: dict[str, Any] = {"task_id": int(task_id)}
        if max_steps is not None:
            task_kwargs["max_steps"] = int(max_steps)

        return gym.make(
            f"browsergym/STWebAgentBenchEnv.{task_id}",
            headless=True,
            action_mapping=self._action_set.to_python_code,
            pw_extra_args=_PLAYWRIGHT_ARGS,
            task_kwargs=task_kwargs,
        )

    def reset(self, env: Any) -> dict:
        obs, _info = env.reset()
        return obs

    def step(self, env: Any, action: str) -> tuple[dict, float, bool, bool, dict]:
        obs, reward, terminated, truncated, info = env.step(action)
        return obs, float(reward), bool(terminated), bool(truncated), info or {}

    # ── Prompting ────────────────────────────────────────────────────────────

    def prompt_fields(self, obs: dict) -> dict[str, str]:
        from browsergym.utils.obs import flatten_axtree_to_str

        policies = obs.get("policies", [])
        try:
            from stwebagentbench.policy_context import format_policy_context
            policies_block = format_policy_context(policies)
        except Exception:
            policies_block = "\n".join(str(p) for p in policies) or "None"

        chat_history = "\n".join(
            f"{m['role']}: {m['message']}" for m in obs.get("chat_messages", [])
        )
        return {
            "goal": obs.get("goal", ""),
            "url": obs.get("url", "N/A"),
            "axtree": flatten_axtree_to_str(obs.get("axtree_object", {})),
            "chat_history": chat_history,
            "policies_block": policies_block,
            "action_space": self._action_space_description,
        }

    def parse_action(self, llm_output: str) -> Optional[str]:
        action = extract_action(llm_output)
        # Keyword args (button='left', ...) become NamedArgument objects that
        # crash BrowserGym's ActionTrace JSON serializer — strip to positional.
        return strip_kwargs(action) if action else None

    # ── Ground truth ─────────────────────────────────────────────────────────

    def safety_report(self, info: dict) -> list[dict]:
        return [normalize_safety_report(r) for r in info.get("safety_report", [])]

    def task_ids(self) -> list[int | str]:
        ids = self.cfg.get("task_ids")
        if not ids:
            raise ValueError(
                "benchmark.task_ids is required — the config must state exactly "
                "which tasks a collection run covers"
            )
        return list(ids)

    @staticmethod
    def raw_policies(obs: dict) -> list:
        """Full policy dicts (org/user/task levels) for the trace payload."""
        return obs.get("policies", []) or []
