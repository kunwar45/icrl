#!/usr/bin/env python3
"""
Run one ST-WebAgentBench episode with Qwen 2.5-72B and save the demo.

Usage:
    python scripts/run_demo.py                  # task 47, headless
    python scripts/run_demo.py --task-id 55
    python scripts/run_demo.py --no-headless    # watch the browser
    python scripts/run_demo.py --max-steps 30
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.agent import run_episode
from src.data.trajectory import load_trajectories
from src.utils.llm_client import QWEN_72B

OUTPUT_PATH = Path("data/demos/single_run.jsonl")


def _count_lines(path: Path) -> int:
    return sum(1 for _ in open(path)) if path.exists() else 0


def save(traj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(traj.to_dict()) + "\n")
    print(f"  Saved → {path}  ({_count_lines(path)} total)")


def display(path: Path) -> None:
    trajs = load_trajectories(str(path))
    t = trajs[-1]
    print(f"\n{'='*60}")
    print(f"  trajectory_id   : {t.trajectory_id}")
    print(f"  task_instance   : {t.task_instance_id}")
    print(f"  is_safe         : {t.is_safe}")
    print(f"  constraint_score: {t.constraint_score}")
    print(f"  reward          : {t.reward}")
    print(f"  n_steps         : {len(t.steps)}")
    print(f"  First actions   : {[s.action for s in t.steps[:3]]}")
    print(f"  JSONL (first 300 chars):")
    print(f"    {json.dumps(t.to_dict())[:300]}...")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", type=int, default=47)
    parser.add_argument("--model", default=QWEN_72B)
    parser.add_argument("--no-headless", action="store_true")
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--output", default=str(OUTPUT_PATH))
    args = parser.parse_args()

    for var in ("OPENROUTER_API_KEY", "WA_SUITECRM"):
        if not os.environ.get(var):
            print(f"ERROR: {var} not set in .env")
            sys.exit(1)

    traj = run_episode(
        task_id=args.task_id,
        model=args.model,
        headless=not args.no_headless,
        max_steps=args.max_steps,
    )
    out = Path(args.output)
    save(traj, out)
    display(out)


if __name__ == "__main__":
    main()
