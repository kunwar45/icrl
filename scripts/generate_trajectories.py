#!/usr/bin/env python3
# ABOUTME: THE entrypoint for synthetic trajectory generation. The config fully defines the run.
# ABOUTME: Run: python scripts/generate_trajectories.py --config configs/trajectory_generation/stwebagentbench_expert.yaml [--smoke]
"""
Thin CLI over src/trajectory_generation. Variants (new benchmark, different
planner/executor, prompt ablations) are configs in configs/trajectory_generation/
— never new scripts.

    python scripts/generate_trajectories.py \
        --config configs/trajectory_generation/stwebagentbench_expert.yaml
    ... --smoke                          # 1 task, OpenRouter backends, laptop-safe
    ... --override generation_loop.max_plan_revisions=1
    ... --dry-run                        # print resolved config, run nothing

On the cluster, use the SLURM wrapper instead (starts the vLLM servers first):
    CONFIG=configs/trajectory_generation/stwebagentbench_expert.yaml \
        sbatch slurm/generate_trajectories_job.sh
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from omegaconf import OmegaConf


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True,
                    help="path to a configs/trajectory_generation/*.yaml run config")
    ap.add_argument("--smoke", action="store_true",
                    help="merge the config's smoke: block — tiny slice, full wiring")
    ap.add_argument("--override", action="append", default=[],
                    help="OmegaConf dotlist override, repeatable")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the resolved config and exit without generating")
    args = ap.parse_args()

    load_dotenv()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")

    cfg = OmegaConf.load(args.config)
    if args.smoke:
        if "smoke" not in cfg:
            raise SystemExit(f"{args.config} has no smoke: block")
        cfg = OmegaConf.merge(cfg, cfg.smoke)
    if args.override:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.override))
    cfg = OmegaConf.to_container(cfg, resolve=True)
    cfg.pop("smoke", None)
    cfg["smoke_run"] = args.smoke

    if args.dry_run:
        print(json.dumps(cfg, indent=2))
        return 0

    from src.trajectory_generation import run_generation
    outcome = run_generation(cfg)

    print(f"\nkept {outcome['kept']} verified traces from {outcome['episodes']} "
          f"episodes → {outcome['output_dir']}")
    # Non-zero when nothing survived verification, so SLURM marks the job failed
    # instead of leaving an empty directory that looks like a finished run.
    return 0 if outcome["kept"] > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
