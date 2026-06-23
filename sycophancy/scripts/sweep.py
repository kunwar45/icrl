#!/usr/bin/env python3
"""
Epsilon sweep for Stage 3 (Lagrangian PPO).

Launches run_finetune.py at three constraint budget values:
    ε ∈ {0.05, 0.10, 0.20}

Lower ε = tighter constraint = less sycophancy allowed, but potentially
higher task reward cost as the policy is more restricted.

Usage:
    python sycophancy/scripts/sweep.py \\
        --constraint_head sycophancy/checkpoints/syco_constraint_v1/constraint_head.pt \\
        --compute local \\
        [--dry_run]
"""
from __future__ import annotations

import argparse
import subprocess
import sys

EPSILONS = [0.05, 0.10, 0.20]

BASE_CMD = [
    sys.executable, "sycophancy/scripts/run_finetune.py",
    "+finetune=lagrangian_ppo",
    "+constraint=supervised_default",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--constraint_head", required=True,
                        help="Path to trained constraint_head.pt")
    parser.add_argument("--compute", default="local",
                        choices=["local", "gcp", "carleton"])
    parser.add_argument("--dry_run", action="store_true",
                        help="Print commands without running")
    args = parser.parse_args()

    for eps in EPSILONS:
        run_name = f"syco_finetune_eps{eps:.2f}".replace(".", "")
        cmd = BASE_CMD + [
            f"+compute={args.compute}",
            f"run_name={run_name}",
            f"finetune.constraint.epsilon={eps}",
            f"finetune.constraint.constraint_model_path={args.constraint_head}",
        ]
        print(f"\n{'='*60}")
        print(f"ε = {eps}  →  run_name={run_name}")
        print(" ".join(cmd))
        if not args.dry_run:
            result = subprocess.run(cmd, check=False)
            if result.returncode != 0:
                print(f"Run failed for ε={eps}. Check logs.")


if __name__ == "__main__":
    main()
