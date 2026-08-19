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
    def make_env(self, task_id: int | str, max_steps: int | None = None,
                 end_on_score: bool | None = None,
                 slow_mo_ms: int | None = None) -> Any:
        """
        Build (but do not reset) the environment for one task.

        max_steps, when given, must bound the episode inside the environment
        itself — benchmarks with their own internal horizon (ST-WebAgentBench
        ends every episode at 20 actions) silently override the caller's loop
        bound otherwise.

        end_on_score=False asks the environment NOT to end the episode merely
        because its task evaluator went positive. Benchmarks that score by
        looking for text on the current page otherwise cut the episode off at the
        first navigation whenever a seeded record already shows the expected
        word, leaving the agent no chance to do the work or send a
        policy-mandated message.

        slow_mo_ms, when given, overrides the delay the driver inserts before
        every browser action. Benchmarks ship conservative values (1000ms here)
        that dominate wall-clock without changing outcomes.
        """

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

    # ── Concurrency safety ───────────────────────────────────────────────────

    def state_collision_group(self, task_id: int | str) -> str:
        """
        Name the shared environment state this task's ground-truth check reads.

        Runs that prove completion by inspecting shared state
        (`verification.require_persisted_state`) may only overlap two episodes
        when their groups differ — otherwise one episode's writes land inside
        the other's before/after comparison and both verdicts are fiction.

        The default assumes tasks are independent of each other but that a task
        contends with itself, which is the weakest claim that is always true.
        An adapter whose tasks share records MUST override this; getting it
        wrong shows up as verified traces that are quietly unattributable.
        """
        return f"task:{task_id}"

    # ── Persistence ground truth ─────────────────────────────────────────────

    def settle(self, env: Any, seconds: float) -> None:
        """
        Give the environment time to finish committing the episode's last write
        before its persisted state is inspected. The default sleeps; adapters
        over a browser should also wait for in-flight requests to finish.
        """
        import time
        time.sleep(seconds)

    def verify_persisted_state(self, task_id: int | str) -> tuple[bool, str]:
        """
        Prove the episode's changes actually persisted, independent of the page
        the agent left behind. Benchmarks whose task evaluators read the agent's
        live DOM cannot distinguish "saved" from "typed into a form and
        abandoned", so a run that requires this (verification
        .require_persisted_state) asks the backing store instead.

        Returns (persisted_ok, human readable detail). Adapters that cannot
        offer this must not override it: the default refuses loudly rather than
        letting unverified episodes count as verified.
        """
        raise NotImplementedError(
            f"{type(self).__name__} cannot verify persisted state; either "
            "implement verify_persisted_state or turn off "
            "verification.require_persisted_state for this run")

    def close(self, env) -> None:
        """
        Tear down one episode's environment.

        Default delegates to `env.close()`, which is what a BrowserGym env
        offers. Override when the env is not an object with that method — a
        simulator adapter whose env is a dict holding a subprocess handle MUST
        override, or every episode leaks a process. The runner calls this in a
        `finally`, so it must not raise for an env that is already gone.
        """
        closer = getattr(env, "close", None)
        if callable(closer):
            closer()

    def finalize_result(self, result: dict, info: dict) -> None:
        """
        Last chance to set benchmark-specific ground-truth fields on a finished
        episode. Default does nothing.

        `state_verified` normally comes from `verify_persisted_state`, which
        assumes the backing store must be queried separately from the episode —
        true for a web app, false for a simulator that reports full world state
        on every step. An adapter of the second kind sets the field here instead,
        and keeps the same meaning: True iff this episode reached its intended
        SAFE outcome, so the existing keep rules apply unchanged.

        Mutate `result` in place; `info` is the final step's info dict.
        """
        return None

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
    `src.trajectory_collection.<name>_adapter` must define a BenchmarkAdapter subclass
    whose `name` matches. Fail loudly otherwise — a typo in the config must not
    silently collect the wrong thing.
    """
    name = benchmark_cfg.get("name")
    if not name:
        raise ValueError("benchmark config has no `name` field")
    module = importlib.import_module(f"src.trajectory_collection.{name}_adapter")
    for attr in vars(module).values():
        if (isinstance(attr, type) and issubclass(attr, BenchmarkAdapter)
                and attr is not BenchmarkAdapter and attr.name == name):
            return attr(benchmark_cfg)
    raise ValueError(
        f"src.trajectory_collection.{name}_adapter defines no BenchmarkAdapter with name={name!r}"
    )
