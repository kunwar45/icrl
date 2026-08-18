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
        # Registering the envs is a module import with side effects. Done here,
        # once, rather than inside make_env: with concurrent episodes several
        # threads would otherwise race into the same first import, and the
        # per-episode import lookup is pure overhead either way.
        import browsergym.stwebagentbench  # noqa: F401 — registers the envs

        # Episodes may run in a thread pool, and BrowserGym's Playwright
        # instance is process-global — which the sync API forbids sharing
        # across threads. Installing unconditionally keeps single-threaded runs
        # behaving identically (one thread, one driver) while making concurrent
        # ones legal.
        from src.environments.playwright_thread_isolation import \
            install_thread_isolated_playwright
        install_thread_isolated_playwright()

        # Skip the per-step extractions nothing here reads (see
        # src/environments/browsergym_lean_observation.py). Config-controlled so
        # a run can fall back to BrowserGym's full observation if a future
        # consumer needs screenshots or DOM properties.
        self._lean_observation = bool(benchmark_cfg.get("lean_observation", True))

    # ── Environment lifecycle ────────────────────────────────────────────────

    def make_env(self, task_id: int | str, max_steps: int | None = None,
                 end_on_score: bool | None = None,
                 slow_mo_ms: int | None = None) -> Any:
        import gymnasium as gym

        # The benchmark's GenericWebArenaTask terminates every episode at its
        # own max_steps (default 20) regardless of the caller's loop bound, so
        # the config's episode.max_steps must be pushed into the task itself.
        # gym.make replaces the registered task_kwargs dict wholesale — task_id
        # must be re-supplied alongside the override.
        task_kwargs: dict[str, Any] = {"task_id": int(task_id)}
        if max_steps is not None:
            task_kwargs["max_steps"] = int(max_steps)
        if end_on_score is not None:
            task_kwargs["end_on_score"] = bool(end_on_score)
        if slow_mo_ms is not None:
            task_kwargs["slow_mo_ms"] = int(slow_mo_ms)

        # Re-assert the thread-local Playwright patch. The benchmark's own env
        # module (stwebagentbench/browser_env/custom_env.py) imports the
        # accessor by value and may only be imported when gym.make runs, so a
        # single install at construction time can miss it. The sweep is a
        # dictionary walk; the failure it prevents is every episode dying in
        # reset() with 'NoneType' object has no attribute 'selectors'.
        from src.environments.playwright_thread_isolation import \
            install_thread_isolated_playwright
        install_thread_isolated_playwright()

        env = gym.make(
            f"browsergym/STWebAgentBenchEnv.{task_id}",
            headless=True,
            action_mapping=self._action_set.to_python_code,
            pw_extra_args=_PLAYWRIGHT_ARGS,
            task_kwargs=task_kwargs,
        )
        if self._lean_observation:
            # Before reset(), which builds the first observation and would
            # otherwise pay the full extraction cost once per episode.
            from src.environments.browsergym_lean_observation import \
                install_lean_observation
            install_lean_observation(env)
        return env

    def reset(self, env: Any) -> dict:
        obs, _info = env.reset()
        return obs

    # ── Task metadata (cached across cycles) ─────────────────────────────────

    def task_metadata(self, task_id: int | str) -> dict[str, str]:
        """
        Goal + policy text for a task, cached on disk between runs.

        The default implementation boots a browser, logs into SuiteCRM and reads
        the first observation — roughly 40 seconds of a 150-second generation
        cycle, spent re-reading strings that are fixed task configuration and
        cannot change between cycles. Worse, that login is the flakiest step in
        the pipeline under concurrency: on 2026-08-16 task 237 lost a whole
        cycle to `Locator.click: Timeout 30000ms exceeded` waiting for the login
        form while another episode logged in beside it.

        The cache is keyed by task id and lives outside the trace directories so
        it is never mistaken for run output. Delete the directory to force a
        re-read (e.g. after changing the benchmark's task definitions).
        """
        import json
        import os
        from pathlib import Path

        cache_dir = Path(self.cfg.get("metadata_cache_dir") or
                         os.path.join(os.environ.get("SCRATCH", "/tmp"),
                                      "icrl_cache", "stwebagentbench_metadata"))
        cache_file = cache_dir / f"task_{int(task_id)}.json"

        if cache_file.is_file():
            try:
                cached = json.loads(cache_file.read_text())
                # A truncated or empty read would poison every later cycle, so
                # only a complete record counts as a hit.
                if cached.get("goal") and "policies_block" in cached:
                    logger.debug("task %s: metadata from cache", task_id)
                    return cached
                logger.warning("task %s: cached metadata looks incomplete — "
                               "re-reading from the environment", task_id)
            except Exception as e:
                logger.warning("task %s: unreadable metadata cache (%s) — "
                               "re-reading from the environment", task_id, e)

        metadata = super().task_metadata(task_id)

        if metadata.get("goal"):
            try:
                cache_dir.mkdir(parents=True, exist_ok=True)
                # Write-then-rename: two concurrent chains may reach this at
                # once, and a half-written file read by the next cycle would be
                # worse than no cache at all.
                temporary = cache_file.with_suffix(f".{os.getpid()}.tmp")
                temporary.write_text(json.dumps(metadata, indent=2))
                temporary.replace(cache_file)
            except Exception as e:
                logger.warning("task %s: could not cache metadata (%s) — "
                               "continuing without it", task_id, e)
        return metadata

    # ── Concurrency safety ───────────────────────────────────────────────────

    def state_collision_group(self, task_id: int | str) -> str:
        """
        Name the shared state a task's ground-truth check reads.

        Two tasks with the SAME group must never run at once against one
        SuiteCRM: their checks are global `SELECT COUNT(*)` queries, so one
        episode's writes land inside the other's before/after comparison and
        both verdicts become fiction. Tasks in different groups are safe to
        overlap, which is what makes concurrency usable on the expert side.

        Grouping is scoped to THIS run's task list. A task that is not being
        generated cannot corrupt anything, so letting it merge two groups only
        costs parallelism: globally, task 252's check joins accounts and
        contacts, which chained all 14 account/contact CRUD tasks into one
        serial run even when 252 was nowhere in the list.
        """
        from src.trajectory_collection.stwebagentbench_state_verifier import \
            task_collision_group
        scope = self.cfg.get("task_ids")
        return task_collision_group(int(task_id),
                                    scope=scope if scope else None)

    # ── Persistence ground truth ─────────────────────────────────────────────

    def settle(self, env: Any, seconds: float) -> None:
        """Wait for the page's in-flight requests, then a margin, so the last
        save is committed before the database is asked about it."""
        import time
        deadline_ms = max(1000, int(seconds * 1000))
        try:
            page = env.unwrapped.page
            page.wait_for_load_state("networkidle", timeout=deadline_ms)
        except Exception:
            # A closed/navigating page just means we fall back to waiting.
            pass
        time.sleep(min(seconds, 5.0))

    def verify_persisted_state(self, task_id: int | str) -> tuple[bool, str]:
        """Ask SuiteCRM's database whether this task's changes really landed."""
        from src.trajectory_collection.stwebagentbench_state_verifier import \
            verify_task_state
        return verify_task_state(int(task_id))

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
