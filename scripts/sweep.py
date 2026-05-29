#!/usr/bin/env python3
"""
Launch multi-seed / multi-epsilon sweeps.

Usage:
    python scripts/sweep.py --mode local
    python scripts/sweep.py --mode slurm
"""
import subprocess
import itertools
import argparse

EPSILONS = [0.05, 0.1, 0.2]
SEEDS = [42, 123, 456]
COMPUTE_TARGET = "carleton"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["local", "slurm"], default="slurm")
    parser.add_argument("--compute", default=COMPUTE_TARGET)
    args = parser.parse_args()

    for epsilon, seed in itertools.product(EPSILONS, SEEDS):
        run_name = f"finetune_eps{epsilon}_seed{seed}"
        cmd_args = [
            f"+compute={args.compute}",
            f"run_name={run_name}",
            f"finetune.constraint.epsilon={epsilon}",
            f"seed={seed}",
        ]

        if args.mode == "local":
            cmd = ["python", "scripts/run_finetune.py"] + cmd_args
            print(f"Running: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)

        elif args.mode == "slurm":
            cmd = ["sbatch", "slurm/finetune_job.sh", "--", *cmd_args]
            print(f"Submitting: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
