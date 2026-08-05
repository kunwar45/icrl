"""BrowserGym / ST-WebAgentBench environment wrapper.

Wraps a single ST-WebAgentBench task as a BaseEnv.  The reward is shaped:
  -step_cost per step, +completion_bonus if task succeeds (browsergym reward = 1.0).

Screenshot bytes are stripped from observations to keep memory footprint small.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

from icrl.core.interfaces import BaseEnv

logger = logging.getLogger(__name__)

try:
    import gymnasium as gym
except ImportError as e:
    raise ImportError("gymnasium not found. pip install gymnasium") from e

try:
    import browsergym  # noqa: F401 — registers envs on import
except ImportError as e:
    raise ImportError(
        "BrowserGym not found. Install with: pip install -e '.[stwebagent]'"
    ) from e


@dataclass
class STWebAgentEnvConfig:
    task_id: str = "235"
    headless: bool = True
    completion_bonus: float = 10.0
    step_cost: float = 0.1
    max_steps: int = 50


_OBS_KEYS_TO_KEEP = {
    "goal", "url", "axtree_object", "policies", "chat_messages", "pruned_html"
}


def _strip_screenshot(obs: dict) -> dict:
    return {k: v for k, v in obs.items() if k in _OBS_KEYS_TO_KEEP}


def _format_policies(policies) -> str:
    if not policies:
        return "None"
    if isinstance(policies, list):
        return "\n".join(f"- {p}" for p in policies)
    return str(policies)


def _flatten_axtree(axtree, max_chars: int = 2000) -> str:
    if axtree is None:
        return ""
    text = str(axtree)
    if len(text) > max_chars:
        text = text[:max_chars] + "…[truncated]"
    return text


class STWebAgentEnv(BaseEnv):
    def __init__(self, config: STWebAgentEnvConfig):
        self.config = config
        self._env_id = f"browsergym/STWebAgentBenchEnv.{config.task_id}"
        self._gym_env: Optional[Any] = None
        self._step_count = 0

    def reset(self, seed: Optional[int] = None) -> tuple[dict, dict]:
        if self._gym_env is not None:
            self._gym_env.close()
        self._gym_env = gym.make(
            self._env_id,
            headless=self.config.headless,
        )
        obs, info = self._gym_env.reset(seed=seed)
        self._step_count = 0
        obs = _strip_screenshot(obs)
        return obs, info

    def step(self, action: str) -> tuple[dict, float, bool, bool, dict]:
        assert self._gym_env is not None, "Call reset() before step()"
        raw_obs, bg_reward, terminated, truncated, info = self._gym_env.step(action)
        self._step_count += 1

        obs = _strip_screenshot(raw_obs)

        reward = -self.config.step_cost
        if terminated and float(bg_reward) >= 1.0:
            reward += self.config.completion_bonus

        if self._step_count >= self.config.max_steps and not (terminated or truncated):
            truncated = True

        return obs, reward, terminated, truncated, info

    def obs_repr(self, obs: dict) -> str:
        goal = obs.get("goal", "")
        url = obs.get("url", "")
        policies = _format_policies(obs.get("policies", []))
        axtree = _flatten_axtree(obs.get("axtree_object"))
        return (
            f"GOAL: {goal}\n"
            f"URL: {url}\n"
            f"POLICIES:\n{policies}\n"
            f"PAGE:\n{axtree}"
        )

    def action_repr(self, action: str) -> str:
        return action

    def close(self) -> None:
        if self._gym_env is not None:
            self._gym_env.close()
            self._gym_env = None
