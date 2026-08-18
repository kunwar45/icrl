#!/usr/bin/env python3
# ABOUTME: THE entrypoint for the ICRL contrast dataset — generates the expert AND unsafe sets from one config
# ABOUTME: Run: python scripts/generate_contrast_dataset.py --config configs/trajectory_generation/stwebagentbench_contrast.yaml [--smoke]
"""
One config, both sets, one job.

C_theta is learned by contrast, so the expert and unsafe sets are two halves of
one dataset and are generated together from one config over one task list with
one model. Generating them separately is what let them drift apart into
different tasks and different model sizes — differences C_theta then learns
instead of safety.

    python scripts/generate_contrast_dataset.py \
        --config configs/trajectory_generation/stwebagentbench_contrast.yaml
    ... --smoke                          # 1 task, OpenRouter backends, laptop-safe
    ... --sets expert                    # one half only (resuming a partial run)
    ... --override generation_loop.traces_per_task=5
    ... --dry-run                        # print both resolved configs, run nothing

Each set is resolved into the flat config shape src.trajectory_generation
consumes, so the runner itself is unchanged and `--sets expert` still generates
one half on its own when a run needs resuming.

On the cluster, use the SLURM wrapper instead (it starts the vLLM server first):
    sbatch scripts/slurm/generate_contrast_dataset_job.sh
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger("generate_contrast_dataset")

#: Set names in generation order. Expert first: it is the half that can fail to
#: verify, and its per-task keep rate is what decides whether the task list is
#: usable at all. Learning that in hour one beats learning it in hour three.
SET_ORDER = ("expert", "unsafe")

#: Blocks a set may override. Everything else is shared by construction — the
#: two sets must differ in the experimental condition and nothing else, and a
#: config that tries to give them different models or task lists is a bug, not
#: a configuration.
OVERRIDABLE = {"keep", "output", "prompts", "verification"}


def resolve_set(cfg: DictConfig, set_name: str) -> dict:
    """
    Merge one set's overrides over the shared config.

    Returns the flat config shape src.trajectory_generation.run_generation takes:
    the shared benchmark/models/episode/generation_loop blocks, plus that set's
    keep rule, output directory and prompts.
    """
    sets = cfg.get("sets")
    if sets is None or set_name not in sets:
        raise SystemExit(f"config has no sets.{set_name}")

    set_cfg = sets[set_name]
    unexpected = set(set_cfg.keys()) - OVERRIDABLE
    if unexpected:
        raise SystemExit(
            f"sets.{set_name} overrides {sorted(unexpected)}, which both sets must "
            f"share. Only {sorted(OVERRIDABLE)} may differ — anything else becomes "
            "a feature C_theta can learn instead of safety.")

    shared = OmegaConf.masked_copy(
        cfg, [k for k in cfg.keys() if k not in ("sets", "smoke", "dataset")])
    merged = OmegaConf.merge(shared, set_cfg)

    # run_generation records this verbatim in every trace and in the manifest.
    merged["generation"] = f"{cfg['dataset']}_{set_name}"
    return OmegaConf.to_container(merged, resolve=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True,
                    help="path to a configs/trajectory_generation/*.yaml with a sets: block")
    ap.add_argument("--sets", default=",".join(SET_ORDER),
                    help=f"comma-separated subset of {','.join(SET_ORDER)} "
                         "(default: both, expert first)")
    ap.add_argument("--smoke", action="store_true",
                    help="merge the config's smoke: block — tiny slice, full wiring")
    ap.add_argument("--override", action="append", default=[],
                    help="OmegaConf dotlist override applied to the SHARED config, repeatable")
    ap.add_argument("--dry-run", action="store_true",
                    help="print both resolved configs and exit without generating")
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

    selected = [s.strip() for s in args.sets.split(",") if s.strip()]
    unknown = [s for s in selected if s not in SET_ORDER]
    if unknown:
        raise SystemExit(f"unknown set(s) {unknown}; valid: {list(SET_ORDER)}")
    # Always expert first regardless of the order given on the command line.
    selected = [s for s in SET_ORDER if s in selected]

    resolved = {name: resolve_set(cfg, name) for name in selected}
    for r in resolved.values():
        r["smoke_run"] = args.smoke

    if args.dry_run:
        print(json.dumps(resolved, indent=2))
        return 0

    # Both sets run against the same task list, so they must never be in flight
    # together: an unsafe episode really does delete the record an expert
    # episode is about to be verified against. Sequential, expert first.
    from src.trajectory_generation import run_generation

    outcomes = {}
    for name in selected:
        logger.info("── generating the %s set ──", name)
        outcomes[name] = run_generation(resolved[name])

    print()
    target = None
    for name, outcome in outcomes.items():
        target = outcome["traces_per_task"]
        short = outcome["tasks_short_of_target"]
        print(f"{name:8s} kept {outcome['kept']:4d} traces from "
              f"{outcome['episodes']:4d} episodes  |  "
              f"{outcome['tasks_at_target']} task(s) at {target}/task"
              f"{'' if not short else f'  |  short: {short}'}")
        print(f"{'':8s} → {outcome['output_dir']}")

    # The dataset is only usable if BOTH halves cover the same tasks: a task with
    # expert traces and no unsafe ones (or the reverse) teaches C_theta to
    # separate the classes by task identity. Say so plainly rather than leaving
    # it to be discovered at training time.
    if len(outcomes) == len(SET_ORDER):
        covered = {name: _tasks_with_traces(Path(o["output_dir"]))
                   for name, o in outcomes.items()}
        only_expert = sorted(covered["expert"] - covered["unsafe"])
        only_unsafe = sorted(covered["unsafe"] - covered["expert"])
        if only_expert or only_unsafe:
            print("\nWARNING: the two sets do not cover the same tasks — a head "
                  "trained on this can separate them by task identity.")
            if only_expert:
                print(f"  expert only: {only_expert}")
            if only_unsafe:
                print(f"  unsafe only: {only_unsafe}")
        else:
            print(f"\nboth sets cover the same {len(covered['expert'])} task(s)")

    # Non-zero when a half came up empty, so SLURM marks the job failed instead
    # of leaving a directory that looks finished.
    return 0 if all(o["kept"] > 0 for o in outcomes.values()) else 1


def _tasks_with_traces(output_dir: Path) -> set[str]:
    """Task ids that have at least one trace on disk, across every pass."""
    return {p.stem.split("_")[1] for p in output_dir.glob("task_*_trace_*.json")}


if __name__ == "__main__":
    raise SystemExit(main())
