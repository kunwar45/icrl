#!/usr/bin/env python3
# ABOUTME: Splits collected demos into train and held-out eval sets, grouped by task instance to avoid leakage
# ABOUTME: Run: python scripts/make_demo_splits.py [--safe <dir_or_jsonl>] [--held-out-frac 0.3 --seed 7]
"""
Split collected demos into train + held-out evaluation sets.

The constraint gate check (scripts/evaluate_constraint.py) must run on trajectories
C_theta never saw during training, otherwise AUROC is meaningless. This script
produces that split.

Splitting is done by *task instance*, not by trajectory: all trajectories for a
given task_instance_id land on the same side of the split. Two trajectories of
the same task share almost identical observations, so a per-trajectory split
leaks the eval set into training.

Usage:
    python scripts/make_demo_splits.py                      # defaults below
    python scripts/make_demo_splits.py --held-out-frac 0.3 --seed 7

Inputs — either format, per side:
    a .jsonl of Trajectory dicts        (data/demos/safe.jsonl)
    a directory of task_*_trace_*.json  (what the SLURM collection job writes to
                                         $SCRATCH/trajectories/safe)

    python scripts/make_demo_splits.py --safe $SCRATCH/trajectories/safe

Outputs:
    data/train/safe.jsonl        data/train/unsafe.jsonl
    data/eval/safe_held_out.jsonl  data/eval/unsafe_held_out.jsonl
    data/splits.json             manifest: counts + which task ids went where
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.trajectory_data.demo_loader import load_demos  # noqa: E402


def load_rows(path: Path) -> list[dict]:
    """Load either input format and return plain Trajectory dicts."""
    return [t.to_dict() for t in load_demos(path)]


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def split_by_task(rows: list[dict], held_out_frac: float, rng: random.Random):
    """Group by task_instance_id, then assign whole groups to train / held-out."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[str(r.get("task_instance_id", "?"))].append(r)

    task_ids = sorted(groups)
    rng.shuffle(task_ids)

    n_held = max(1, round(len(task_ids) * held_out_frac)) if task_ids else 0
    # Never hand the entire set to eval — training needs at least one task.
    n_held = min(n_held, max(0, len(task_ids) - 1))

    held_ids, train_ids = task_ids[:n_held], task_ids[n_held:]
    held = [r for t in held_ids for r in groups[t]]
    train = [r for t in train_ids for r in groups[t]]
    return train, held, sorted(train_ids), sorted(held_ids)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--safe", type=Path, default=Path("data/demos/safe.jsonl"))
    ap.add_argument("--unsafe", type=Path, default=Path("data/demos/unsafe.jsonl"))
    ap.add_argument("--train-dir", type=Path, default=Path("data/train"))
    ap.add_argument("--eval-dir", type=Path, default=Path("data/eval"))
    ap.add_argument("--manifest", type=Path, default=Path("data/splits.json"))
    ap.add_argument("--held-out-frac", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=None,
                    help="Keep only the first N trajectories of each label (smoke runs)")
    args = ap.parse_args()

    for p in (args.safe, args.unsafe):
        if not p.exists():
            print(f"ERROR: {p} not found. Collect demos first.", file=sys.stderr)
            return 1

    rng = random.Random(args.seed)
    manifest: dict = {"seed": args.seed, "held_out_frac": args.held_out_frac, "labels": {}}

    for label, src, train_out, eval_out in (
        ("safe", args.safe, args.train_dir / "safe.jsonl",
         args.eval_dir / "safe_held_out.jsonl"),
        ("unsafe", args.unsafe, args.train_dir / "unsafe.jsonl",
         args.eval_dir / "unsafe_held_out.jsonl"),
    ):
        rows = load_rows(src)
        if args.limit:
            rows = rows[: args.limit]
        train, held, train_ids, held_ids = split_by_task(rows, args.held_out_frac, rng)

        write_jsonl(train, train_out)
        write_jsonl(held, eval_out)

        n_terminated = sum(1 for r in rows if r.get("terminated"))
        manifest["labels"][label] = {
            "source": str(src),
            "n_total": len(rows),
            "n_terminated": n_terminated,
            "n_train": len(train),
            "n_held_out": len(held),
            "train_task_ids": train_ids,
            "held_out_task_ids": held_ids,
        }
        print(f"{label:6s}  total={len(rows):4d}  terminated={n_terminated:4d}  "
              f"train={len(train):4d} ({len(train_ids)} tasks)  "
              f"held_out={len(held):4d} ({len(held_ids)} tasks)")

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2))
    print(f"\nManifest → {args.manifest}")

    # Guard against a degenerate eval set: AUROC needs both classes present.
    n_safe_held = manifest["labels"]["safe"]["n_held_out"]
    n_unsafe_held = manifest["labels"]["unsafe"]["n_held_out"]
    if n_safe_held == 0 or n_unsafe_held == 0:
        empty = "safe" if n_safe_held == 0 else "unsafe"
        n_tasks = len(manifest["labels"][empty]["train_task_ids"])
        print(
            f"ERROR: the held-out set has no {empty} trajectories, so AUROC is "
            f"undefined and the gate cannot run.\n"
            f"       Only {n_tasks} distinct {empty} task instance(s) available — a "
            f"held-out split needs at least 2.\n"
            f"       Collect more {empty} demos, or point --{empty} at a larger set.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
