# ABOUTME: Orchestrates trajectory generation per task: plan → refine → plan-guided
# ABOUTME: execution → ground-truth verify → revise-on-failure. Saves verified traces only.
"""
The loop per task:

    metadata = adapter.task_metadata(task)          # goal + policies, no episode
    plan     = refine(propose(metadata))            # planner model, 2 calls
    for revision in 0..max_plan_revisions:
        result = run_episode(..., extra_fields={plan})   # executor model, REAL env
        if keep(result): save(result + plan); break      # ground-truth verified
        plan = revise(plan, failure_report(result))      # planner sees what broke

Saved traces use the collection trace schema plus provenance fields
(pipeline, plan, plan_revisions), so trace_loader and every downstream stage
consume them unchanged.
"""
from __future__ import annotations

import csv
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from src.trajectory_collection.benchmark_adapter import BenchmarkAdapter, get_adapter
from src.trajectory_collection.collection_runner import KEEP_RULES, _next_trace_n
from src.trajectory_collection.episode_runner import run_episode
from src.trajectory_generation.plan_generator import (build_failure_report,
                                                      diversify_plan,
                                                      plan_similarity,
                                                      propose_plan, refine_plan,
                                                      revise_plan)

logger = logging.getLogger(__name__)


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
        "terminated": result["terminated"],
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
    }
    executor_temperature = float(executor_cfg.get("temperature", 0.0))

    output_dir = Path(cfg["output"]["dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {"started": datetime.now(timezone.utc).isoformat(), "config": cfg}
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    summary_rows: list[dict] = []
    n_kept = n_episodes = 0

    for task_id in adapter.task_ids():
        logger.info("task %s: reading metadata", task_id)
        try:
            metadata = adapter.task_metadata(task_id)
        except Exception as e:
            logger.warning("task %s: metadata failed: %s — skipping", task_id, e)
            summary_rows.append({"task_id": task_id, "kept": 0, "episodes_run": 0,
                                 "plan_revisions": 0, "outcome": f"metadata error: {e}"})
            continue

        plan = propose_plan(planner, planner_cfg, prompts, metadata)
        plan = refine_plan(planner, planner_cfg, prompts, metadata, plan,
                           plan_history)

        # Diversity gate: if the refined plan is near-verbatim of a recent one,
        # force one explicit diversification pass and record what happened.
        similarity = plan_similarity(plan, plan_history)
        if diversity_on and similarity > max_similarity:
            logger.info("task %s: plan too similar to a previous one (%.2f > %.2f) "
                        "— diversifying", task_id, similarity, max_similarity)
            plan = diversify_plan(planner, planner_cfg, prompts, metadata,
                                  plan, plan_history)
            similarity = plan_similarity(plan, plan_history)
        logger.info("task %s: plan ready (%d chars, max similarity %.2f)",
                    task_id, len(plan), similarity)

        kept_this_task = 0
        episodes_this_task = 0
        outcome = "exhausted revisions"

        for revision in range(max_revisions + 1):
            result = run_episode(adapter, executor, episode_cfg, task_id,
                                 executor_temperature, extra_fields={"plan": plan})
            n_episodes += 1
            episodes_this_task += 1

            if result["error"]:
                logger.warning("task %s revision %d errored: %s",
                               task_id, revision, result["error"])
            if keep(result):
                path = _save_trace(result, cfg, plan, revision,
                                   _next_trace_n(output_dir, task_id), output_dir)
                logger.info("task %s: verified and kept %s (revision %d)",
                            task_id, path.name, revision)
                kept_this_task = 1
                n_kept += 1
                outcome = f"kept at revision {revision}"
                # Only plans that passed verification enter the diversity
                # context — failed plans are noise, not templates to avoid.
                plan_history.append(plan)
                del plan_history[:-history_size]
                break

            report = build_failure_report(result)
            logger.info("task %s revision %d failed verification:\n%s",
                        task_id, revision, report)
            if revision < max_revisions:
                plan = revise_plan(planner, planner_cfg, prompts, metadata,
                                   plan, report)

        summary_rows.append({"task_id": task_id, "kept": kept_this_task,
                             "episodes_run": episodes_this_task,
                             "plan_revisions": episodes_this_task - 1,
                             "plan_similarity": round(similarity, 3),
                             "outcome": outcome})

    with open(output_dir / "summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["task_id", "kept", "episodes_run",
                                               "plan_revisions", "plan_similarity",
                                               "outcome"])
        writer.writeheader()
        writer.writerows(summary_rows)

    manifest["finished"] = datetime.now(timezone.utc).isoformat()
    manifest["kept"] = n_kept
    manifest["episodes"] = n_episodes
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    logger.info("generation done: kept %d verified traces from %d episodes → %s",
                n_kept, n_episodes, output_dir)
    return {"kept": n_kept, "episodes": n_episodes, "output_dir": str(output_dir)}
