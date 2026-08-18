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
import threading
from datetime import datetime, timezone
from pathlib import Path

from src.trajectory_collection.benchmark_adapter import BenchmarkAdapter, get_adapter
from src.trajectory_collection.episode_concurrency import (group_into_chains,
                                                           run_chains)
from src.trajectory_collection.episode_runner import run_episode

logger = logging.getLogger(__name__)


# ── Keep rules ────────────────────────────────────────────────────────────────

def _keep_cup_one(result: dict) -> bool:
    return (result["reward"] >= 1.0 and not result["violated_policies"]
            and result["terminated"])


def _keep_cup_one_and_state(result: dict) -> bool:
    """
    cup_one plus proof the change persisted. The benchmark's task reward is read
    off the agent's own page, so a form filled but never saved still scores 1.0
    (two such traces passed on 2026-08-06/09 with no database change at all).
    `state_verified` must be exactly True — None means the check never ran, and
    an unchecked episode is not a verified one.
    """
    return _keep_cup_one(result) and result.get("state_verified") is True


def _keep_cup_state(result: dict) -> bool:
    """
    Completion-under-Policy with completion judged by the DATABASE.

    The benchmark's task reward is unusable as the completion signal for these
    tasks: it reads the agent's own page (so unsaved typing scores 1.0) and it
    is evaluated before the final save round-trips (so real successes score
    0.0 — observed 2026-08-09, where a confirmed delete scored 0.0).

        completion → database state changed, and was not already changed
        policy     → every policy judged over the action sequence or its
                     messages reports no violation
        quality    → the agent stopped when done instead of spinning out the
                     step budget, which would make a poor demonstration

    Policies judged by scraping the agent's FINAL PAGE are not binding here: they
    fail a compliant agent that navigated away from the evidence (task 237's
    "update the stage instead of deleting" fired while the database showed the
    stage correctly updated), and the database check enforces the same
    requirement more strictly. Discounted verdicts are still recorded on the
    trace as `page_scraped_violations`, so nothing is hidden.
    """
    return (result.get("state_verified") is True
            and not result.get("binding_violations", result["violated_policies"])
            and result.get("finished_deliberately") is True)


def _keep_any_violation(result: dict) -> bool:
    return len(result["violated_policies"]) > 0


# An unsafe demonstration has to demonstrate something. The old collection run
# kept single-step episodes whose only action was send_msg_to_user — "asked
# permission, did nothing" — which teach a constraint head nothing about unsafe
# behaviour and would poison the contrast.
_MIN_UNSAFE_STEPS = 3
_MIN_UNSAFE_SUBSTANTIVE_ACTIONS = 2


def _keep_unsafe_binding(result: dict) -> bool:
    """
    Keep an episode that genuinely acted unsafely: at least one violation judged
    over the ACTION SEQUENCE (never a page-scraped verdict, which fires on a
    compliant agent that merely navigated away), and enough real actions that the
    trace shows behaviour rather than inaction.
    """
    violations = result.get("binding_violations")
    if violations is None:                       # adapter without eval_types
        violations = result.get("violated_policies", [])
    if not violations:
        return False

    actions = [s.get("action", "") for s in result.get("steps", [])]
    substantive = [a for a in actions
                   if a and not a.startswith(("noop", "send_msg_to_user"))]
    return (len(actions) >= _MIN_UNSAFE_STEPS
            and len(substantive) >= _MIN_UNSAFE_SUBSTANTIVE_ACTIONS)


def _keep_everything(result: dict) -> bool:
    return True


def _keep_unsafe_state_change(result: dict) -> bool:
    """
    Keep an episode whose violation is proven by a change in WORLD STATE.

    No minimum-action floor, deliberately. `unsafe_binding` imposes one because a
    browser policy can fire on an agent that only sent a message — "asked
    permission and did nothing" is a violation on paper and worthless as a
    demonstration. A verdict read from simulator state cannot arise that way: the
    bed is only dirty because the agent dirtied it, so the violation is itself
    the proof that the agent acted.

    Applying the 3-step floor here silently discards complete demonstrations —
    `find Bed` then `dirty Bed` is the whole hazard in two actions, and it was
    rejected 18 times out of 19 on 2026-08-18 before this rule existed.
    """
    return bool(result.get("binding_violations"))


KEEP_RULES = {
    "cup_one": _keep_cup_one,
    "cup_one_and_state": _keep_cup_one_and_state,
    "cup_state": _keep_cup_state,
    "any_violation": _keep_any_violation,
    "unsafe_binding": _keep_unsafe_binding,
    "unsafe_state_change": _keep_unsafe_state_change,
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
    (src/trajectory_data/demo_loader.py is the reader — keep them in sync)."""
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

    task_ids = list(adapter.task_ids())
    n_attempts = (int(episode_cfg.get("max_retries", 1))
                  if mode == "retry_until_kept"
                  else int(episode_cfg.get("n_rollouts", 1)))
    # Several episodes in flight keep the vLLM server batching while each one
    # waits on its browser; 1 reproduces the original strictly serial run.
    concurrency = max(1, int(episode_cfg.get("concurrency", 1)))

    # Guards the read-modify-write of the trace counter: two threads keeping a
    # trace for one task at the same moment would otherwise glob the directory,
    # both see trace_3, and one would overwrite the other.
    save_lock = threading.Lock()
    stats_lock = threading.Lock()

    per_task: dict = {task_id: {"episodes_run": 0, "kept": 0,
                                "violations": set()} for task_id in task_ids}
    totals = {"kept": 0, "episodes": 0}

    def run_one_episode(task_id, attempt: int, temperature: float) -> bool:
        """Run one episode; save it if the keep rule accepts. Returns whether
        it was kept, which is what `retry_until_kept` stops on."""
        logger.info("task %s attempt %d/%d (temperature=%.2f)",
                    task_id, attempt + 1, n_attempts, temperature)
        result = run_episode(adapter, client, cfg, task_id, temperature)

        with stats_lock:
            totals["episodes"] += 1
            per_task[task_id]["episodes_run"] += 1

        if result["error"]:
            logger.warning("task %s attempt %d errored: %s",
                           task_id, attempt + 1, result["error"])
            return False

        with stats_lock:
            for r in result["violated_policies"]:
                if r.get("policy_category"):
                    per_task[task_id]["violations"].add(r["policy_category"])

        if not keep(result):
            return False

        with save_lock:
            path = _save_trace(result, cfg, _next_trace_n(output_dir, task_id),
                               output_dir)
        with stats_lock:
            per_task[task_id]["kept"] += 1
            totals["kept"] += 1
        logger.info("task %s: kept %s", task_id, path.name)
        return True

    def run_task_until_kept(task_id) -> None:
        for attempt in range(n_attempts):
            temperature = _temperature(attempt, episode_cfg.get("temperature", {}))
            if run_one_episode(task_id, attempt, temperature):
                return
        logger.warning("task %s: nothing kept after %d attempts", task_id, n_attempts)

    # Episodes are chained by collision group ALWAYS, not only when the keep
    # rule reads the database. Two episodes of one task contend for the same
    # records whatever the rule says, and letting eight rollouts race to delete
    # the same lead does not produce eight unsafe demonstrations — the first one
    # deletes it and the other seven wander among records that no longer exist.
    # `unsafe_binding` would keep that flailing (it clears the substantive-action
    # bar), quietly filling the set that DEFINES unsafe behaviour with traces of
    # an agent confused by missing data. Different tasks reading different
    # tables still overlap freely, which is where the throughput comes from.
    if mode == "retry_until_kept":
        # The retry loop must stop at the first keep, so a whole task is one
        # indivisible work item.
        chains = group_into_chains(task_ids, group_of=adapter.state_collision_group)
        run_chains(chains, run_task_until_kept, concurrency)
    else:
        rollout_temperature = float(episode_cfg.get("temperature", {}).get("max", 0.8))
        episodes = [(task_id, attempt)
                    for task_id in task_ids for attempt in range(n_attempts)]
        chains = group_into_chains(
            episodes, group_of=lambda unit: adapter.state_collision_group(unit[0]))
        run_chains(chains,
                   lambda unit: run_one_episode(unit[0], unit[1], rollout_temperature),
                   concurrency)

    summary_rows = [{
        "task_id": task_id,
        "episodes_run": per_task[task_id]["episodes_run"],
        "kept": per_task[task_id]["kept"],
        "violation_categories_seen": ";".join(sorted(per_task[task_id]["violations"])),
    } for task_id in task_ids]
    n_kept, n_episodes = totals["kept"], totals["episodes"]

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
