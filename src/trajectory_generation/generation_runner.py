# ABOUTME: Orchestrates trajectory generation per task: plan → refine → plan-guided
# ABOUTME: execution → ground-truth verify → revise-on-failure. Saves verified traces only.
"""
The loop per task:

    metadata = adapter.task_metadata(task)          # goal + policies, no episode
    while traces_on_disk < generation_loop.traces_per_task:
        plan = refine(propose(metadata))            # a FRESH plan per trace
        for revision in 0..max_plan_revisions:
            result = run_episode(..., extra_fields={plan})  # executor, REAL env
            if keep(result): save(result + plan); break     # ground-truth verified
            plan = revise(plan, failure_report(result))     # planner sees what broke
        if nothing was kept this plan cycle: stop, let the next pass retry

Every task is generated to the same per-task target so no single task can
dominate the dataset (see `traces_per_task` in the config). The target counts
traces already on disk, so repeated passes converge on it instead of piling
more traces onto the tasks that happen to keep most easily; a task already at
target is skipped without booting a browser.

Saved traces use the collection trace schema plus provenance fields
(pipeline, plan, plan_revisions), so trace_loader and every downstream stage
consume them unchanged.
"""
from __future__ import annotations

import csv
import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from src.trajectory_collection.benchmark_adapter import BenchmarkAdapter, get_adapter
from src.trajectory_collection.collection_runner import (KEEP_RULES, _next_trace_n,
                                                        _temperature)
from src.trajectory_collection.episode_concurrency import (group_into_chains,
                                                           run_chains)
from src.trajectory_collection.episode_runner import run_episode
from src.trajectory_data.dataset_shape import check_target, describe_band
from src.trajectory_generation.plan_generator import (build_failure_report,
                                                      diversify_plan,
                                                      plan_similarity,
                                                      propose_plan, refine_plan,
                                                      revise_plan)

logger = logging.getLogger(__name__)

#: Attempts at reading a task's goal + policies before giving up on it for this
#: cycle. Reading metadata boots a browser and logs in, and concurrent logins to
#: one SuiteCRM time out intermittently — a transport failure, not a task
#: failure, and losing it costs the task its entire run. The cache is warmed
#: serially in run_generation, so these retries are the second line of defence.
_METADATA_ATTEMPTS = 3

#: Base backoff between metadata attempts, multiplied by the attempt number. A
#: login timeout lasts 30s, so retries spaced tighter than that just re-enter
#: the same failure window — which is exactly how tasks 244 and 247 lost both
#: their attempts on 2026-08-17.
_METADATA_RETRY_SECONDS = 20


def _build_client(model_cfg: dict):
    from src.utils.llm_client import (make_hf_client, make_openrouter_client,
                                      make_vllm_client)
    backend = model_cfg["backend"]
    if backend == "vllm":
        return make_vllm_client(model_cfg["vllm_url"])
    if backend == "openrouter":
        return make_openrouter_client()
    if backend == "hf-local":
        return make_hf_client(model_name=model_cfg["name"])
    raise ValueError(f"unknown model backend {backend!r}")


def resolve_traces_per_task(cfg: dict) -> int:
    """
    The per-task trace target, validated against the design band.

    A config that asks for 1 or for 110 is a dataset-design error, not a knob
    setting, so it fails here rather than three stages downstream where it
    surfaces as an unexplained AUROC.
    """
    loop_cfg = cfg.get("generation_loop") or {}
    if "traces_per_task" not in loop_cfg:
        raise KeyError(
            "generation_loop.traces_per_task is required — every set is generated "
            f"to the same per-task target ({describe_band()})")
    return check_target(int(loop_cfg["traces_per_task"]),
                        "generation_loop.traces_per_task")


def existing_trace_count(output_dir: Path, task_id) -> int:
    """Verified traces already on disk for a task, across every earlier pass."""
    return len(list(output_dir.glob(f"task_{task_id}_trace_*.json")))


def _save_trace(result: dict, cfg: dict, plan: str, revisions: int,
                trace_n: int, output_dir: Path) -> Path:
    path = output_dir / f"task_{result['task_id']}_trace_{trace_n}.json"
    payload = {
        "task_id": result["task_id"],
        "generation": cfg["generation"],
        "pipeline": "trajectory_generation",
        "benchmark": cfg["benchmark"]["name"],
        "set": cfg["keep"]["set"],
        "model": cfg["models"]["executor"]["name"],
        "planner_model": cfg["models"]["planner"]["name"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_steps": result["n_steps"],
        "reward": result["reward"],
        "cup": True,
        # `terminated` is the environment's flag and is False whenever the runner
        # itself ended the episode at goal confirmation, so record how the episode
        # actually finished alongside it.
        "terminated": result["terminated"],
        "finished_deliberately": result.get("finished_deliberately"),
        "ended_on_goal_confirmed": result.get("ended_on_goal_confirmed", False),
        # How this trace was proven: the benchmark evaluator alone cannot tell a
        # saved record from an abandoned form, so record whether the database
        # confirmed persistence and what it checked.
        "state_verified": result.get("state_verified"),
        "state_detail": result.get("state_detail", ""),
        # Verdicts from policies judged by scraping the final page, which the
        # database check supersedes. Recorded rather than dropped so a reader can
        # see exactly what was discounted and why this trace was still kept.
        "page_scraped_violations": result.get("page_scraped_violations", []),
        # Provenance: the plan that produced this episode, and how many times
        # it had to meet reality before passing verification.
        "plan": plan,
        "plan_revisions": revisions,
        "policies": result["policies"],
        "safety_report": result["safety_report"],
        "steps": result["steps"],
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path


def run_generation(cfg: dict) -> dict:
    """Execute one generation run; returns {kept, episodes, output_dir}."""
    adapter: BenchmarkAdapter = get_adapter(cfg["benchmark"])
    planner_cfg = cfg["models"]["planner"]
    executor_cfg = cfg["models"]["executor"]
    planner = _build_client(planner_cfg)
    executor = _build_client(executor_cfg)
    keep = KEEP_RULES[cfg["keep"]["rule"]]
    prompts = cfg["prompts"]
    max_revisions = int(cfg["generation_loop"]["max_plan_revisions"])
    traces_per_task = resolve_traces_per_task(cfg)

    diversity_cfg = cfg.get("diversity") or {}
    diversity_on = bool(diversity_cfg.get("enabled", False))
    max_similarity = float(diversity_cfg.get("max_similarity", 0.85))
    history_size = int(diversity_cfg.get("history", 8))
    plan_history: list[str] = []

    # run_episode consumes this shape; prompts.execute may reference {plan}.
    episode_cfg = {
        "model": executor_cfg,
        "prompt": prompts["execute"],
        "episode": cfg["episode"],
        # Lets run_episode ask the adapter for database proof of persistence.
        "verification": cfg.get("verification", {}),
    }
    # Revisions only ever differed by their plan: a temperature-0 executor
    # reproduces almost the same actions each time, so tasks that fail on
    # execution rather than planning burn all four attempts identically (244,
    # 246, 247, 248 and 252 each failed 8/8 that way). Reuse the collection
    # runner's schedule — greedy on revision 0, so the tasks that already keep
    # reliably are untouched, then warmer to actually explore.
    executor_temperature = float(executor_cfg.get("temperature", 0.0))
    revision_temperatures = executor_cfg.get("temperature_schedule")

    output_dir = Path(cfg["output"]["dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {"started": datetime.now(timezone.utc).isoformat(), "config": cfg}
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    task_ids = list(adapter.task_ids())
    # Tasks run concurrently, so the vLLM server keeps batching while each
    # episode waits on its browser. Safety comes from the chaining below, not
    # from a lower number here.
    concurrency = max(1, int(cfg["generation_loop"].get("concurrency", 1)))

    # plan_history is read by every refine/diversify call and appended to by
    # every keep; the trace counter is a read-modify-write over the output
    # directory. Both need a lock once tasks overlap.
    history_lock = threading.Lock()
    save_lock = threading.Lock()
    stats_lock = threading.Lock()

    summary_by_task: dict = {}
    totals = {"kept": 0, "episodes": 0}

    def read_metadata(task_id):
        """
        Goal + policy text for a task, retried.

        This boots a browser and logs in purely to read two static strings, and
        under concurrency that login is flaky: on 2026-08-16 task 237 lost an
        entire cycle to `Locator.click: Timeout 30000ms exceeded` waiting for
        the login form while another episode was logging in beside it. The task
        was fine — the read was not. One retry converts a lost cycle into a few
        wasted seconds.
        """
        last_error = None
        for attempt in range(_METADATA_ATTEMPTS):
            try:
                return adapter.task_metadata(task_id)
            except Exception as e:
                last_error = e
                logger.warning("task %s: metadata attempt %d/%d failed: %s",
                               task_id, attempt + 1, _METADATA_ATTEMPTS, e)
                # Back off before retrying. Immediate retries were useless: both
                # of task 244's attempts on 2026-08-17 landed inside the same
                # 30-second login timeout window and failed identically, losing
                # the task for the whole run.
                if attempt + 1 < _METADATA_ATTEMPTS:
                    time.sleep(_METADATA_RETRY_SECONDS * (attempt + 1))
        raise last_error

    def run_task(task_id) -> None:
        on_disk = existing_trace_count(output_dir, task_id)
        if on_disk >= traces_per_task:
            # Reached across earlier passes. Skipping costs nothing and is what
            # stops a task that keeps easily from running away with the set.
            logger.info("task %s: already at target (%d/%d traces) — skipping",
                        task_id, on_disk, traces_per_task)
            with stats_lock:
                summary_by_task[task_id] = {
                    "task_id": task_id, "kept": 0, "traces_on_disk": on_disk,
                    "target": traces_per_task, "episodes_run": 0,
                    "plan_revisions": 0, "plan_similarity": 0.0,
                    "outcome": "at target"}
            return

        logger.info("task %s: reading metadata (%d/%d traces on disk)",
                    task_id, on_disk, traces_per_task)
        try:
            metadata = read_metadata(task_id)
        except Exception as e:
            logger.warning("task %s: metadata failed: %s — skipping", task_id, e)
            with stats_lock:
                summary_by_task[task_id] = {
                    "task_id": task_id, "kept": 0, "traces_on_disk": on_disk,
                    "target": traces_per_task, "episodes_run": 0,
                    "plan_revisions": 0, "plan_similarity": 0.0,
                    "outcome": f"metadata error: {e}"}
            return

        kept_this_task = 0
        episodes_this_task = 0
        similarity = 0.0
        outcome = "exhausted revisions"

        # One plan cycle per trace. Re-planning rather than re-running the same
        # plan is the point: N traces of one task are only worth having if they
        # differ, and the diversity gate below measures each new plan against
        # the ones already verified.
        while on_disk + kept_this_task < traces_per_task:
            # Snapshot the history: refining against a list another thread is
            # appending to would be a race, and a plan's diversity only needs to
            # be measured against what was verified when it was written.
            with history_lock:
                history_snapshot = list(plan_history)

            plan = propose_plan(planner, planner_cfg, prompts, metadata)
            plan = refine_plan(planner, planner_cfg, prompts, metadata, plan,
                               history_snapshot)

            # Diversity gate: if the refined plan is near-verbatim of a recent
            # one, force one explicit diversification pass and record what
            # happened.
            similarity = plan_similarity(plan, history_snapshot)
            if diversity_on and similarity > max_similarity:
                logger.info("task %s: plan too similar to a previous one (%.2f > %.2f) "
                            "— diversifying", task_id, similarity, max_similarity)
                plan = diversify_plan(planner, planner_cfg, prompts, metadata,
                                      plan, history_snapshot)
                similarity = plan_similarity(plan, history_snapshot)
            logger.info("task %s: plan ready (%d chars, max similarity %.2f), "
                        "trace %d/%d", task_id, len(plan), similarity,
                        on_disk + kept_this_task + 1, traces_per_task)

            kept_this_plan = False
            stop_task = False

            for revision in range(max_revisions + 1):
                result = run_episode(adapter, executor, episode_cfg, task_id,
                                     (_temperature(revision, revision_temperatures)
                                      if revision_temperatures else executor_temperature),
                                     extra_fields={"plan": plan})
                episodes_this_task += 1
                with stats_lock:
                    totals["episodes"] += 1

                if result["error"]:
                    logger.warning("task %s revision %d errored: %s",
                                   task_id, revision, result["error"])
                if keep(result):
                    with save_lock:
                        path = _save_trace(result, cfg, plan, revision,
                                           _next_trace_n(output_dir, task_id), output_dir)
                    kept_this_task += 1
                    logger.info("task %s: verified and kept %s (revision %d) "
                                "— %d/%d traces", task_id, path.name, revision,
                                on_disk + kept_this_task, traces_per_task)
                    with stats_lock:
                        totals["kept"] += 1
                    outcome = f"kept {kept_this_task} this pass"
                    kept_this_plan = True
                    # Only plans that passed verification enter the diversity
                    # context — failed plans are noise, not templates to avoid.
                    with history_lock:
                        plan_history.append(plan)
                        del plan_history[:-history_size]
                    break

                report = build_failure_report(result)
                logger.info("task %s revision %d failed verification:\n%s",
                            task_id, revision, report)

                # A destructive or one-shot task consumes its own precondition:
                # once the record is deleted or created, no later revision can be
                # proven to have done it. Retrying just burns GPU hours on
                # episodes that can never be kept — stop and let the next
                # (reseeded) pass retry. Such a task reaches its target one trace
                # per reseed, which is why the target counts traces on disk.
                if result.get("state_satisfied_before"):
                    outcome = "goal consumed by an earlier episode — needs a reseed"
                    logger.info("task %s: stopping revisions, %s", task_id, outcome)
                    stop_task = True
                    break

                if revision < max_revisions:
                    plan = revise_plan(planner, planner_cfg, prompts, metadata,
                                       plan, report)

            if stop_task:
                break
            if not kept_this_plan:
                # A whole plan cycle failed every revision. Another fresh plan in
                # the same pass would most likely fail the same way against the
                # same environment state; stop here and let the next pass retry
                # rather than burning the job's remaining GPU hours on it.
                outcome = (f"exhausted revisions after keeping {kept_this_task}"
                           if kept_this_task else "exhausted revisions")
                break

        with stats_lock:
            summary_by_task[task_id] = {
                "task_id": task_id, "kept": kept_this_task,
                "traces_on_disk": on_disk + kept_this_task,
                "target": traces_per_task,
                "episodes_run": episodes_this_task,
                "plan_revisions": max(0, episodes_this_task - kept_this_task),
                "plan_similarity": round(similarity, 3),
                "outcome": outcome}

    # Warm the metadata cache SERIALLY before any episode starts.
    #
    # Reading a task's goal and policies boots a browser and logs into SuiteCRM.
    # With the cache cold, every worker thread does that at the same instant and
    # the login form cannot serve them: on 2026-08-17 job 4845953 lost tasks 244
    # and 247 outright — both metadata attempts timed out
    # (`Locator.click: Timeout 30000ms exceeded`) inside the same login storm, so
    # neither task ran a single episode and the run looked like an execution
    # failure rather than a login one.
    #
    # One login at a time costs ~40s per uncached task ONCE (the cache is on
    # disk and survives cycles), and removes the storm entirely — including the
    # subtler version where a cache miss races an episode already in flight.
    uncached = []
    for task_id in task_ids:
        if existing_trace_count(output_dir, task_id) >= traces_per_task:
            continue  # nothing to generate, so nothing to read
        try:
            read_metadata(task_id)
        except Exception as e:
            logger.warning("task %s: metadata unavailable (%s) — it will be "
                           "retried inside the run", task_id, e)
            uncached.append(task_id)
    if uncached:
        logger.warning("metadata still missing for %s after the serial warm-up",
                       uncached)

    # Every kept trace here is proven by a differential database check, so two
    # tasks reading the same tables must never be in flight together — their
    # writes would land inside each other's before/after comparison. Tasks that
    # read disjoint tables are chained separately and run in parallel.
    chains = group_into_chains(task_ids, group_of=adapter.state_collision_group)
    logger.info("generating %d tasks to %d traces each, in %d collision-free "
                "chains, concurrency %d",
                len(task_ids), traces_per_task, len(chains), concurrency)
    run_chains(chains, run_task, concurrency)

    summary_rows = [summary_by_task[task_id] for task_id in task_ids
                    if task_id in summary_by_task]
    n_kept, n_episodes = totals["kept"], totals["episodes"]

    # Traces accumulate as trace_0, trace_1, ... across passes, so a single
    # summary.csv would silently overwrite every earlier pass's outcomes (it did,
    # twice, hiding which pass produced a keep). Keep a per-pass copy alongside.
    summary_fields = ["task_id", "kept", "traces_on_disk", "target", "episodes_run",
                      "plan_revisions", "plan_similarity", "outcome"]
    pass_id = os.environ.get("SLURM_JOB_ID")
    if pass_id:
        with open(output_dir / f"summary_pass_{pass_id}.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=summary_fields)
            writer.writeheader()
            writer.writerows(summary_rows)

    with open(output_dir / "summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    # How far the set is from its target shape, not just how much it grew. A pass
    # that kept traces but left half the tasks short is not a finished set.
    at_target = sum(1 for r in summary_rows if r["traces_on_disk"] >= traces_per_task)
    short = sorted(r["task_id"] for r in summary_rows
                   if r["traces_on_disk"] < traces_per_task)

    manifest["finished"] = datetime.now(timezone.utc).isoformat()
    manifest["kept"] = n_kept
    manifest["episodes"] = n_episodes
    manifest["traces_per_task"] = traces_per_task
    manifest["tasks_at_target"] = at_target
    manifest["tasks_short_of_target"] = short
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    logger.info("generation done: kept %d verified traces from %d episodes; "
                "%d/%d tasks at %d traces → %s",
                n_kept, n_episodes, at_target, len(summary_rows), traces_per_task,
                output_dir)
    if short:
        logger.info("tasks still short of target: %s", short)
    return {"kept": n_kept, "episodes": n_episodes, "output_dir": str(output_dir),
            "traces_per_task": traces_per_task, "tasks_at_target": at_target,
            "tasks_short_of_target": short}
