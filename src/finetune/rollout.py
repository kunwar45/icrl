"""
Policy rollout plumbing shared by fine-tuning (src/finetune/lagrangian.py) and
evaluation (scripts/eval_finetune.py).

One place decides how an observation becomes a prompt, how a generation becomes
an action, and how an episode becomes a Trajectory — so the reward the trainer
optimises and the CuP the evaluator reports are computed the same way.
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol

import torch

from src.data.actions import (
    AGENT_SYSTEM_PROMPT,
    extract_action,
    normalize_safety_report,
)
from src.data.st_webagent import obs_repr, _strip_screenshot
from src.data.trajectory import Step, Trajectory
from src.finetune.reward import RewardModel

logger = logging.getLogger(__name__)


class TaskEnvironment(Protocol):
    """Contract satisfied by both STWebAgentBench and MockBenchmark."""

    def load_tasks(self) -> Dict[str, Dict[str, Any]]: ...

    def env_for_task(self, task_id: str, headless: bool = True) -> Any: ...


# ── Episode result ────────────────────────────────────────────────────────────

@dataclass
class StepRecord:
    """Token ids for one (prompt, generated action) pair — the PG training unit."""
    prompt_ids: torch.Tensor    # (P,) long, CPU
    action_ids: torch.Tensor    # (A,) long, CPU


@dataclass
class RolloutResult:
    trajectory: Trajectory
    task_reward: float                      # R(tau) from the RewardModel
    env_reward: float                       # raw benchmark reward
    violations: List[Dict[str, Any]] = field(default_factory=list)
    records: List[StepRecord] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def n_violations(self) -> int:
        return len(self.violations)

    @property
    def completed(self) -> bool:
        return RewardModel.completion(self.trajectory)

    @property
    def cup(self) -> bool:
        return RewardModel.cup(self.trajectory, self.n_violations)


# ── Policy actor ──────────────────────────────────────────────────────────────

class PolicyActor:
    """Turns an observation string into an action string with a causal LM."""

    def __init__(
        self,
        model,
        tokenizer,
        max_prompt_tokens: int = 1024,
        max_new_tokens: int = 48,
        temperature: float = 0.7,
        do_sample: bool = True,
        system_prompt: str = AGENT_SYSTEM_PROMPT,
        use_chat_template: Optional[bool] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.max_prompt_tokens = max_prompt_tokens
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.do_sample = do_sample
        self.system_prompt = system_prompt
        # Instruct models need their chat template; base/test models have none.
        if use_chat_template is None:
            use_chat_template = bool(getattr(tokenizer, "chat_template", None))
        self.use_chat_template = use_chat_template
        self._overhead_cache: Optional[int] = None

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def _fit_observation(self, obs_text: str) -> str:
        """
        Shrink the observation so the *whole* prompt fits the token budget.

        Truncating the assembled prompt instead would either cut the system
        prompt (left) or the generation cue (right). A flattened axtree runs
        past the budget routinely, so keep the head — GOAL / URL / POLICIES —
        and the tail of the page, and drop the middle.
        """
        budget = max(64, self.max_prompt_tokens - self._overhead_tokens())
        ids = self.tokenizer(obs_text, add_special_tokens=False)["input_ids"]
        if len(ids) <= budget:
            return obs_text
        head = ids[: budget // 3]
        tail = ids[-(budget - budget // 3):]
        return (self.tokenizer.decode(head)
                + "\n…[observation truncated]…\n"
                + self.tokenizer.decode(tail))

    def _overhead_tokens(self) -> int:
        """Token cost of the scaffolding around the observation."""
        if self._overhead_cache is None:
            scaffold = self.build_prompt("", fit=False)
            self._overhead_cache = len(
                self.tokenizer(scaffold, add_special_tokens=False)["input_ids"]
            ) + 8  # small margin for special tokens
        return self._overhead_cache

    def build_prompt(self, obs_text: str, fit: bool = True) -> str:
        if fit:
            obs_text = self._fit_observation(obs_text)
        if self.use_chat_template:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user",
                 "content": f"{obs_text}\n\nReply with exactly one action call."},
            ]
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        return f"{self.system_prompt}\n\n{obs_text}\n\nAction:"

    @torch.no_grad()
    def act(self, obs_text: str) -> tuple[str, StepRecord]:
        prompt = self.build_prompt(obs_text)
        enc = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_prompt_tokens,
            add_special_tokens=not self.use_chat_template,
        )
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        was_training = self.model.training
        self.model.eval()
        try:
            out = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.do_sample,
                temperature=self.temperature,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
        finally:
            if was_training:
                self.model.train()

        gen_ids = out[0][input_ids.shape[1]:]
        raw = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        action = extract_action(raw) or "noop()"

        record = StepRecord(
            prompt_ids=input_ids[0].detach().cpu(),
            action_ids=gen_ids.detach().cpu(),
        )
        return action, record


# ── Episode driver ────────────────────────────────────────────────────────────

USER_AUTO_REPLY = "Acknowledged. Please proceed with the task."


def _violations_from_info(info: dict) -> List[Dict[str, Any]]:
    reports = [normalize_safety_report(r) for r in (info or {}).get("safety_report", [])]
    return [r for r in reports if r["violated"]]


def _add_chat_message(env, role: str, message: str) -> None:
    """
    Mirror the benchmark contract: every agent action is announced on the chat
    before env.step, and the user's reply is posted back.

    The consent evaluators score `chat.messages`, so an agent that asks for
    confirmation without this gets no credit for asking. The mock environment
    has no chat, hence the guard.
    """
    chat = getattr(env, "chat", None)
    if chat is None:
        return
    try:
        chat.add_message(role, message)
    except Exception as e:
        logger.debug("chat.add_message(%s) failed: %s", role, e)


def rollout_episode(
    actor: PolicyActor,
    env_provider: TaskEnvironment,
    task_id: str,
    reward_model: RewardModel,
    max_steps: int = 30,
    headless: bool = True,
    auto_reply_limit: int = 3,
) -> RolloutResult:
    """
    Run one episode and package it as a RolloutResult.

    Follows the same environment contract as scripts/collect_safe_trajectories.py,
    so trajectories produced here are comparable with the collected demos:

      * the action is posted to the chat before env.step
      * a scripted user reply follows send_msg_to_user, up to auto_reply_limit.
        Without it, asking for confirmation is a dead end — the agent waits for
        an answer that never comes and burns the rest of its step budget, so the
        *safe* behaviour would score CuP=0 by construction and the fine-tune
        would learn never to ask.
      * a failing env.step falls back to noop() instead of aborting the episode
    """
    tasks = env_provider.load_tasks()
    task = tasks.get(str(task_id), {"task_id": str(task_id), "task_type": "unknown"})

    env = env_provider.env_for_task(str(task_id), headless=headless)
    steps: List[Step] = []
    records: List[StepRecord] = []
    env_reward = 0.0
    terminated = False
    violations: List[Dict[str, Any]] = []
    error: Optional[str] = None
    n_auto_replies = 0
    info: dict = {}

    try:
        raw_obs, info = env.reset()
        obs_text = obs_repr(_strip_screenshot(raw_obs))
        violations = _violations_from_info(info)

        for step_idx in range(max_steps):
            action, record = actor.act(obs_text)
            steps.append(Step(step_idx=step_idx, action=action, observation=obs_text))
            records.append(record)

            _add_chat_message(env, "assistant", action)

            truncated = False
            try:
                raw_next, reward, terminated, truncated, info = env.step(action)
            except Exception as e:
                # A malformed action must cost a step, not the whole episode.
                logger.debug("step(%r) failed on task %s: %s", action, task_id, e)
                try:
                    raw_next, reward, terminated, truncated, info = env.step("noop()")
                except Exception:
                    raise

            env_reward = max(env_reward, float(reward))
            violations = _violations_from_info(info)

            if "send_msg_to_user" in action and n_auto_replies < auto_reply_limit:
                _add_chat_message(env, "user", USER_AUTO_REPLY)
                raw_next = dict(raw_next)
                raw_next["chat_messages"] = list(raw_next.get("chat_messages", [])) + [
                    {"role": "user", "message": USER_AUTO_REPLY}
                ]
                n_auto_replies += 1

            obs_text = obs_repr(_strip_screenshot(raw_next))

            if terminated or truncated:
                break
    except Exception as e:  # a browser crash must not kill the training run
        error = f"{type(e).__name__}: {e}"
        logger.warning("Rollout for task %s failed: %s", task_id, error)
    finally:
        try:
            env.close()
        except Exception:
            pass

    trajectory = Trajectory(
        trajectory_id=uuid.uuid4().hex[:8],
        task_type=str(task.get("task_type", "unknown")),
        task_instance_id=str(task_id),
        steps=steps,
        is_safe=len(violations) == 0,
        source="policy",
        reward=env_reward,
        terminated=bool(terminated),
    )

    return RolloutResult(
        trajectory=trajectory,
        task_reward=reward_model.score(trajectory),
        env_reward=env_reward,
        violations=violations,
        records=records,
        error=error,
    )


# ── Environment factory ───────────────────────────────────────────────────────

def build_env_provider(cfg) -> TaskEnvironment:
    """
    Construct the rollout environment from config.

    finetune.env.backend:
        "mock"          — src.data.mock_env.MockBenchmark (no browser, no CRM)
        "stwebagent"    — the real benchmark; needs BrowserGym + a live SuiteCRM
    """
    from omegaconf import OmegaConf

    backend = OmegaConf.select(cfg, "finetune.env.backend") or "stwebagent"
    max_steps = int(OmegaConf.select(cfg, "finetune.env.max_steps") or 30)

    if backend == "mock":
        from src.data.mock_env import MockBenchmark
        task_ids = OmegaConf.select(cfg, "finetune.env.task_ids")
        return MockBenchmark(
            max_steps=max_steps,
            task_ids=[str(t) for t in task_ids] if task_ids else None,
        )

    if backend == "stwebagent":
        from src.data.st_webagent import STWebAgentBench
        return STWebAgentBench(benchmark_root=str(cfg.paths.benchmark_root or ""))

    raise ValueError(f"Unknown finetune.env.backend: {backend!r} "
                     f"(expected 'mock' or 'stwebagent')")


def select_task_ids(
    env_provider: TaskEnvironment,
    cfg,
    key: str = "finetune.env.task_ids",
) -> List[str]:
    """Explicit task_ids from config, else every task the provider knows."""
    from omegaconf import OmegaConf

    configured = OmegaConf.select(cfg, key)
    if configured:
        return [str(t) for t in configured]
    return sorted(env_provider.load_tasks())
