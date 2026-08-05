# ABOUTME: BenchmarkAdapter base class + registry — the seam that keeps the collection
# ABOUTME: engine benchmark-agnostic. New benchmark = new <benchmark>_adapter.py, nothing else.
"""
Everything benchmark-specific lives behind this interface: how an environment is
built, how an observation becomes prompt fields, and how the benchmark's
ground-truth safety verdict is read back. The engine (episode_runner,
collection_runner) sees only this class.
"""
from __future__ import annotations

import importlib
from abc import ABC, abstractmethod
from typing import Any, Optional


class BenchmarkAdapter(ABC):
    """One instance per collection run, built from the config's `benchmark:` block."""

    #: registry key; must equal the adapter module's `<name>_adapter.py` prefix
    name: str = ""

    def __init__(self, benchmark_cfg: dict):
        self.cfg = benchmark_cfg

    # ── Environment lifecycle ────────────────────────────────────────────────

    @abstractmethod
    def make_env(self, task_id: int | str) -> Any:
        """Build (but do not reset) the environment for one task."""

    @abstractmethod
    def reset(self, env: Any) -> dict:
        """Reset and return the first observation."""

    @abstractmethod
    def step(self, env: Any, action: str) -> tuple[dict, float, bool, bool, dict]:
        """Execute one action: (obs, reward, terminated, truncated, info)."""

    # ── Prompting ────────────────────────────────────────────────────────────

    @abstractmethod
    def prompt_fields(self, obs: dict) -> dict[str, str]:
        """
        Flatten an observation into the string fields the config's prompt
        templates may reference (e.g. goal, url, axtree, chat_history,
        policies_block, action_space).
        """

    @abstractmethod
    def parse_action(self, llm_output: str) -> Optional[str]:
        """Extract one executable action string from raw LLM output, or None."""

    # ── Ground truth ─────────────────────────────────────────────────────────

    @abstractmethod
    def safety_report(self, info: dict) -> list[dict]:
        """
        Normalized per-policy verdicts from the final step's info, each at least
        {policy_id, policy_category, description, violated, reason}.
        """

    @abstractmethod
    def task_ids(self) -> list[int | str]:
        """The task set this run collects over, from the config."""

    def task_metadata(self, task_id: int | str) -> dict[str, str]:
        """
        Goal + policy text for a task WITHOUT running an episode — what the
        trajectory-generation pipeline plans from. The default boots the env,
        reads the first observation's prompt fields, and closes; adapters may
        override with a cheaper task-config lookup.
        """
        env = self.make_env(task_id)
        try:
            fields = self.prompt_fields(self.reset(env))
        finally:
            try:
                env.close()
            except Exception:
                pass
        return {"goal": fields.get("goal", ""),
                "policies_block": fields.get("policies_block", ""),
                "action_space": fields.get("action_space", "")}


def get_adapter(benchmark_cfg: dict) -> BenchmarkAdapter:
    """
    Instantiate the adapter named by `benchmark_cfg["name"]`.

    Resolution is strictly by naming convention:
    `src.data_collection.<name>_adapter` must define a BenchmarkAdapter subclass
    whose `name` matches. Fail loudly otherwise — a typo in the config must not
    silently collect the wrong thing.
    """
    name = benchmark_cfg.get("name")
    if not name:
        raise ValueError("benchmark config has no `name` field")
    module = importlib.import_module(f"src.data_collection.{name}_adapter")
    for attr in vars(module).values():
        if (isinstance(attr, type) and issubclass(attr, BenchmarkAdapter)
                and attr is not BenchmarkAdapter and attr.name == name):
            return attr(benchmark_cfg)
    raise ValueError(
        f"src.data_collection.{name}_adapter defines no BenchmarkAdapter with name={name!r}"
    )
