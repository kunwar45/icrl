#!/usr/bin/env python3
"""
Smoke-test one ST-WebAgentBench episode with Qwen and verify the CuP score.

Follows the benchmark's st_bench_example.py pattern exactly.
Focuses on SuiteCRM tasks 235-294 (self-hosted via Docker, no AWS needed).

Usage:
    # Random task from the 235-294 SuiteCRM range:
    python scripts/verify_cup.py

    # Specific task:
    python scripts/verify_cup.py --task-id 243

    # Via vLLM on cluster:
    python scripts/verify_cup.py --model Qwen/Qwen2.5-72B-Instruct \\
        --vllm-base-url http://localhost:8000/v1

Exit code: 0 if CuP=1, 1 otherwise.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()


def main():
    parser = argparse.ArgumentParser(description="Verify one ST-WebAgentBench CuP episode")
    parser.add_argument("--task-id", type=str, default=None,
                        help="Task ID (default: random from 235-294 SuiteCRM range)")
    parser.add_argument("--model", type=str, default="qwen/qwen-2.5-72b-instruct",
                        help="Model name (default: Qwen-72B via OpenRouter)")
    parser.add_argument("--vllm-base-url", type=str, default=None,
                        help="vLLM base URL (e.g. http://localhost:8000/v1)")
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--no-headless", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("data/demos/verify_run.jsonl"))
    args = parser.parse_args()

    # Default to the SuiteCRM tiered range (235-294), self-hosted via Docker
    task_id = args.task_id or str(random.randint(235, 294))

    print("═" * 60)
    print("  ST-WebAgentBench CuP Verification")
    print("═" * 60)
    print(f"  Task ID  : {task_id}")
    print(f"  Model    : {args.model}")
    print(f"  Backend  : {'vLLM @ ' + args.vllm_base_url if args.vllm_base_url else 'OpenRouter'}")
    print(f"  Max steps: {args.max_steps}")
    print("─" * 60)
    print()

    from src.data.agent import run_episode
    from src.data.trajectory import save_trajectories

    traj = run_episode(
        task_id=task_id,
        model=args.model,
        headless=not args.no_headless,
        max_steps=args.max_steps,
        verbose=True,
        vllm_base_url=args.vllm_base_url,
    )

    # Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "a") as f:
        f.write(json.dumps(traj.to_dict()) + "\n")
    print(f"\n  Trajectory saved → {args.output}")

    # CuP result
    cup = traj.is_safe and traj.reward == 1.0
    print()
    print("═" * 60)
    if cup:
        print("  RESULT: PASS — CuP=1  (task completed + no policy violations)")
        print("  This is a valid safe demo for ICRL training.")
    else:
        reasons = []
        if traj.reward != 1.0:
            reasons.append(f"task not completed (reward={traj.reward:.1f})")
        if not traj.is_safe:
            reasons.append("policy violation recorded")
        print(f"  RESULT: FAIL — CuP=0  ({'; '.join(reasons)})")
    print("═" * 60)

    print(f"""
To read the saved trajectory:
  from src.data.trajectory import load_trajectories
  t = load_trajectories("{args.output}")[-1]
  print(f"task={{t.task_instance_id}} safe={{t.is_safe}} reward={{t.reward}} steps={{len(t.steps)}}")
  for s in t.steps: print(f"  [{{s.step_idx:2d}}] {{s.action}}")
""")

    sys.exit(0 if cup else 1)


if __name__ == "__main__":
    main()
