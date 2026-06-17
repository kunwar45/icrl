#!/usr/bin/env python3
"""
Collect safe + unsafe demos from the 170 SuiteCRM tasks.

Randomly samples tasks, runs Qwen, labels with benchmark ground truth,
and saves until --target safe demos are collected. Unsafe demos are saved
alongside so both are available for constraint training.

Resumable: re-running picks up from the existing file counts.

Usage:
    python scripts/collect_demos.py                  # 150 safe demos
    python scripts/collect_demos.py --target 10      # quick test
    python scripts/collect_demos.py --no-headless    # watch the browser
"""
from __future__ import annotations

import argparse
import importlib.resources
import json
import os
import random
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))

import stwebagentbench

from src.data.agent import run_episode
from src.utils.llm_client import QWEN_72B

SAFE_PATH  = Path("data/demos/safe.jsonl")
UNSAFE_PATH = Path("data/demos/unsafe.jsonl")


def get_suitecrm_task_ids() -> list[int]:
    raw = importlib.resources.files(stwebagentbench).joinpath("test.raw.json").read_text()
    tasks = json.loads(raw)
    return [i for i, t in enumerate(tasks) if (t.get("sites") or [""])[0] == "suitecrm"]


def count_lines(path: Path) -> int:
    return sum(1 for _ in open(path)) if path.exists() else 0


def append(traj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(traj.to_dict()) + "\n")
        f.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=150,
                        help="Stop after this many safe demos (default: 150)")
    parser.add_argument("--max-steps", type=int, default=30,
                        help="Max steps per episode (default: 30)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-headless", action="store_true")
    parser.add_argument("--model", default=QWEN_72B)
    args = parser.parse_args()

    for var in ("OPENROUTER_API_KEY", "WA_SUITECRM"):
        if not os.environ.get(var):
            print(f"ERROR: {var} not set in .env")
            sys.exit(1)

    task_ids = get_suitecrm_task_ids()
    rng = random.Random(args.seed)

    safe_count  = count_lines(SAFE_PATH)
    unsafe_count = count_lines(UNSAFE_PATH)

    print(f"Tasks available : {len(task_ids)} SuiteCRM")
    print(f"Target          : {args.target} safe demos")
    print(f"Resuming from   : {safe_count} safe, {unsafe_count} unsafe")
    print(f"Output          : {SAFE_PATH}  /  {UNSAFE_PATH}")
    print()

    attempt = 0
    while safe_count < args.target:
        task_id = rng.choice(task_ids)
        attempt += 1

        print(f"[{attempt:4d}]  safe={safe_count}/{args.target}  unsafe={unsafe_count}  task={task_id}")

        try:
            traj = run_episode(
                task_id=task_id,
                model=args.model,
                headless=not args.no_headless,
                max_steps=args.max_steps,
            )
        except Exception as e:
            print(f"        episode failed: {e}")
            continue

        if traj.is_safe:
            append(traj, SAFE_PATH)
            safe_count += 1
        else:
            append(traj, UNSAFE_PATH)
            unsafe_count += 1

    print(f"\nDone.")
    print(f"  Safe demos  : {safe_count}  →  {SAFE_PATH}")
    print(f"  Unsafe demos: {unsafe_count}  →  {UNSAFE_PATH}")


if __name__ == "__main__":
    main()
