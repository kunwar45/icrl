#!/usr/bin/env python3
"""
Phase 3 verification for saved safe trajectories.

Spot-checks saved task_*_trace_*.json files and produces:
  - Per-file sanity check (CuP=1, reward=1.0, zero violations, policies present)
  - Yield rate per task (from run_log.jsonl)
  - Summary of most commonly violated evaluator dimensions in failed runs

Usage:
    python scripts/verify_trajectories.py
    python scripts/verify_trajectories.py --traj-dir trajectories/safe --spot-check 5
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_trajectories(traj_dir: Path) -> list[dict]:
    paths = sorted(traj_dir.glob("task_*_trace_*.json"))
    trajs = []
    for p in paths:
        try:
            with open(p) as f:
                trajs.append((p, json.load(f)))
        except Exception as e:
            print(f"  [WARN] Could not load {p}: {e}")
    return trajs


def spot_check(trajs: list[tuple[Path, dict]], n: int) -> tuple[int, int]:
    """Check n trajectories. Returns (passed, failed)."""
    sample = trajs[:n]
    passed = failed = 0
    for path, traj in sample:
        issues = []

        if traj.get("reward") != 1.0:
            issues.append(f"reward={traj.get('reward')} (expected 1.0)")

        if not traj.get("cup"):
            issues.append("cup=False")

        violated = [r for r in traj.get("safety_report", []) if r.get("violated")]
        if violated:
            issues.append(f"{len(violated)} policy violation(s): {[r.get('policy_category') for r in violated]}")

        if not traj.get("policies"):
            issues.append("policies list is empty — policy text not saved")

        if not traj.get("steps"):
            issues.append("steps list is empty")

        if not traj.get("timestamp"):
            issues.append("no timestamp")

        if issues:
            print(f"  ✗ {path.name}: {'; '.join(issues)}")
            failed += 1
        else:
            n_policies = len(traj.get("policies", []))
            n_steps = traj.get("n_steps", len(traj.get("steps", [])))
            print(f"  ✓ {path.name}  reward=1.0  steps={n_steps}  policies={n_policies}")
            passed += 1

    return passed, failed


def yield_report(traj_dir: Path) -> None:
    """Print per-task attempt count from run_log.jsonl."""
    log_path = traj_dir / "run_log.jsonl"
    if not log_path.exists():
        print("  (no run_log.jsonl found — yield rate unavailable)")
        return

    attempts: dict[int, int] = {}
    cup_attempt: dict[int, int] = {}

    with open(log_path) as f:
        for line in f:
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            tid = entry["task_id"]
            attempts[tid] = attempts.get(tid, 0) + 1
            if entry.get("cup") and tid not in cup_attempt:
                cup_attempt[tid] = entry["attempt"]

    print(f"\n  {'task_id':<10} {'attempts':<10} {'cup_on_attempt'}")
    print(f"  {'-'*8:<10} {'-'*8:<10} {'-'*14}")
    for tid in sorted(attempts):
        cup_str = str(cup_attempt[tid]) if tid in cup_attempt else "FLAGGED"
        print(f"  {tid:<10} {attempts[tid]:<10} {cup_str}")


def violation_report(traj_dir: Path) -> None:
    """Summarize which evaluator dimensions most commonly fired on failed runs."""
    log_path = traj_dir / "run_log.jsonl"
    if not log_path.exists():
        return

    # Re-load saved trajectories to get violation detail from failed runs
    # (run_log only has counts, not categories — load individual failed files)
    # Instead, scan the run_log for n_violations > 0 and report aggregated counts
    # from the saved CuP=1 traces' safety_report (failed runs don't have JSON files).
    # Best we can do without failed-run files: report from the log's n_violations.
    with open(log_path) as f:
        entries = [json.loads(l) for l in f if l.strip()]

    failed = [e for e in entries if not e.get("cup") and e.get("n_violations", 0) > 0]
    if not failed:
        print("  No failed runs with violations recorded.")
        return

    total_runs = len(entries)
    total_with_violations = len(failed)
    print(f"  Runs with violations: {total_with_violations}/{total_runs}")
    print(f"  (Detailed per-evaluator breakdown requires saved failed-run files)")


def main():
    parser = argparse.ArgumentParser(description="Verify saved safe trajectories")
    parser.add_argument("--traj-dir", type=Path, default=Path("trajectories/safe"),
                        help="Directory containing task_*_trace_*.json files")
    parser.add_argument("--spot-check", type=int, default=5,
                        help="Number of trajectories to spot-check (default: 5)")
    args = parser.parse_args()

    traj_dir = args.traj_dir
    if not traj_dir.exists():
        print(f"Directory not found: {traj_dir}")
        print("Run collect_safe_trajectories.py first to generate trajectories.")
        sys.exit(1)

    trajs = load_trajectories(traj_dir)
    if not trajs:
        print(f"No task_*_trace_*.json files found in {traj_dir}")
        sys.exit(1)

    print(f"\n{'═'*60}")
    print(f"Trajectory Verification Report")
    print(f"Directory : {traj_dir.resolve()}")
    print(f"Files     : {len(trajs)}")
    print(f"{'═'*60}")

    print(f"\n── Spot-check (first {min(args.spot_check, len(trajs))}) ──")
    n = min(args.spot_check, len(trajs))
    passed, failed = spot_check(trajs, n)
    print(f"\n  {passed}/{n} passed spot-check")

    print(f"\n── Yield rate per task ──")
    yield_report(traj_dir)

    print(f"\n── Failed-run violation summary ──")
    violation_report(traj_dir)

    print(f"\n── All saved trajectory stats ──")
    all_steps = [len(t.get("steps", [])) for _, t in trajs]
    if all_steps:
        print(f"  Avg steps   : {sum(all_steps)/len(all_steps):.1f}")
        print(f"  Min steps   : {min(all_steps)}")
        print(f"  Max steps   : {max(all_steps)}")
    print(f"  Total saved : {len(trajs)}")


if __name__ == "__main__":
    main()
