# ABOUTME: The collection loop: tasks × episodes → keep-rule filter → task_<id>_trace_<n>.json
# ABOUTME: plus manifest.json + summary.csv. Called only by scripts/collect_trajectories.py.
"""
Two episode modes, chosen by the config:

  retry_until_kept   Expert collection. Retry each task until one episode passes
                     the keep rule (or retries run out). Attempt 0 is greedy
                     (temperature 0); later attempts ramp so a retry diverges
                     from the failed greedy path instead of replaying it.
  fixed_rollouts     Unsafe collection. Roll each task a fixed number of times
                     and keep every episode that passes the keep rule.

Keep rules (`keep.rule`):

  cup_one            reward == 1.0 AND zero violations AND terminated cleanly
  any_violation      at least one ground-truth policy violation
  everything         keep all completed episodes (debugging only)

Every run writes a manifest.json recording the resolved config, so a trace
directory is always self-describing.
"""
from __future__ import annotations

import csv
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from src.data_collection.benchmark_adapter import BenchmarkAdapter, get_adapter
from src.data_collection.episode_runner import run_episode

logger = logging.getLogger(__name__)


# ── Keep rules ────────────────────────────────────────────────────────────────

def _keep_cup_one(result: dict) -> bool:
    return (result["reward"] >= 1.0 and not result["violated_policies"]
            and result["terminated"])


def _keep_any_violation(result: dict) -> bool:
    return len(result["violated_policies"]) > 0


def _keep_everything(result: dict) -> bool:
    return True


KEEP_RULES = {
    "cup_one": _keep_cup_one,
    "any_violation": _keep_any_violation,
    "everything": _keep_everything,
}


# ── Temperature schedule ──────────────────────────────────────────────────────

def _temperature(attempt: int, schedule: dict) -> float:
    if attempt == 0:
        return float(schedule.get("first", 0.0))
    ramp = float(schedule.get("base", 0.2)) + float(schedule.get("step", 0.15)) * attempt
    return min(ramp, float(schedule.get("max", 0.8)))


# ── Client construction ───────────────────────────────────────────────────────

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


# ── Saving ────────────────────────────────────────────────────────────────────

def _save_trace(result: dict, cfg: dict, trace_n: int, output_dir: Path) -> Path:
    """Write one kept episode in the task_<id>_trace_<n>.json schema
    (src/data/trace_loader.py is the reader — keep them in sync)."""
    path = output_dir / f"task_{result['task_id']}_trace_{trace_n}.json"
    payload = {
        "task_id": result["task_id"],
        "collection": cfg["collection"],
        "benchmark": cfg["benchmark"]["name"],
        "set": cfg["keep"]["set"],
        "model": cfg["model"]["name"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_steps": result["n_steps"],
        "reward": result["reward"],
        "cup": _keep_cup_one(result),
        # Downstream must distinguish "decided it was done" from "ran out of
        # steps": ICRL wants near-optimal experts, and truncation is the tell.
        "terminated": result["terminated"],
        "policies": result["policies"],
        "safety_report": result["safety_report"],
        "steps": result["steps"],
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path


def _next_trace_n(output_dir: Path, task_id) -> int:
    """Resume-safe counter: continue numbering from existing traces."""
    n = 0
    for p in output_dir.glob(f"task_{task_id}_trace_*.json"):
        try:
            n = max(n, int(p.stem.rsplit("_", 1)[1]) + 1)
        except ValueError:
            pass
    return n


# ── The loop ──────────────────────────────────────────────────────────────────

def run_collection(cfg: dict) -> dict:
    """Execute one collection run; returns {kept, episodes, output_dir}."""
    adapter: BenchmarkAdapter = get_adapter(cfg["benchmark"])
    client = _build_client(cfg["model"])
    keep = KEEP_RULES[cfg["keep"]["rule"]]

    episode_cfg = cfg["episode"]
    mode = episode_cfg["mode"]
    if mode not in ("retry_until_kept", "fixed_rollouts"):
        raise ValueError(f"unknown episode.mode {mode!r}")

    output_dir = Path(cfg["output"]["dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "started": datetime.now(timezone.utc).isoformat(),
        "config": cfg,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    summary_rows: list[dict] = []
    n_kept = n_episodes = 0

    for task_id in adapter.task_ids():
        n_attempts = (int(episode_cfg.get("max_retries", 1))
                      if mode == "retry_until_kept"
                      else int(episode_cfg.get("n_rollouts", 1)))
        kept_this_task = 0
        episodes_this_task = 0
        violations_seen: set[str] = set()

        for attempt in range(n_attempts):
            temperature = (_temperature(attempt, episode_cfg.get("temperature", {}))
                           if mode == "retry_until_kept"
                           else float(episode_cfg.get("temperature", {}).get("max", 0.8)))
            logger.info("task %s attempt %d/%d (temperature=%.2f)",
                        task_id, attempt + 1, n_attempts, temperature)

            result = run_episode(adapter, client, cfg, task_id, temperature)
            n_episodes += 1
            episodes_this_task += 1

            if result["error"]:
                logger.warning("task %s attempt %d errored: %s",
                               task_id, attempt + 1, result["error"])
                continue

            for r in result["violated_policies"]:
                if r.get("policy_category"):
                    violations_seen.add(r["policy_category"])

            if keep(result):
                path = _save_trace(result, cfg, _next_trace_n(output_dir, task_id),
                                   output_dir)
                logger.info("task %s: kept %s", task_id, path.name)
                kept_this_task += 1
                n_kept += 1
                if mode == "retry_until_kept":
                    break

        if mode == "retry_until_kept" and kept_this_task == 0:
            logger.warning("task %s: nothing kept after %d attempts", task_id, n_attempts)

        summary_rows.append({
            "task_id": task_id,
            "episodes_run": episodes_this_task,
            "kept": kept_this_task,
            "violation_categories_seen": ";".join(sorted(violations_seen)),
        })

    with open(output_dir / "summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys())
                                if summary_rows else ["task_id"])
        writer.writeheader()
        writer.writerows(summary_rows)

    manifest["finished"] = datetime.now(timezone.utc).isoformat()
    manifest["kept"] = n_kept
    manifest["episodes"] = n_episodes
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    logger.info("collection done: kept %d / %d episodes → %s",
                n_kept, n_episodes, output_dir)
    return {"kept": n_kept, "episodes": n_episodes, "output_dir": str(output_dir)}
