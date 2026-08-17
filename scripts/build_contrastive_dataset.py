#!/usr/bin/env python3
# ABOUTME: Drives generation and collection passes until the expert and unsafe trajectory sets hit their targets
# ABOUTME: Run on a login node: nohup python scripts/build_contrastive_dataset.py --expert-target 150 --unsafe-target 150 > logs/build_dataset.log 2>&1 &
"""
Builds the contrastive dataset that `train_constraint.py` needs: a safe (expert)
set and an unsafe set, sized so the learned C_theta can be trusted.

Why a driver rather than a chain of sbatch jobs:

  * Exactly one pass may touch SuiteCRM at a time (each reseeds the shared
    database), so passes are inherently serial. Submitting them one at a time and
    waiting is simpler and safer than dependency chains, and it never leaves an
    allocation idling on a lock.
  * Yield per pass is small and uneven. A fixed number of passes either stops
    short of the target or burns GPU-hours after the set has saturated, so the
    decision to continue has to be made from the counts after each pass.
  * The two sets must stay MATCHED by task. The driver always advances whichever
    set is further from its target, keeping the classes comparable.

Sizing (see the notes printed by --explain):
    ~8    per set   the code runs (one batch_size=8 batch)
    ~50   per set   the head stops trivially memorising
    ~125  per set   AUROC is stable enough to trust the auroc_gate: 0.75

Stalling is expected and handled: if a pass adds nothing to the set it targeted
twice in a row, that set is marked saturated and the driver stops working on it
rather than repeating a run that cannot yield.

Usage:
    python scripts/build_contrastive_dataset.py --explain
    python scripts/build_contrastive_dataset.py --dry-run
    python scripts/build_contrastive_dataset.py --expert-target 150 --unsafe-target 150
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

TRACE_NAME = re.compile(r"task_(\d+)_trace_(\d+)\.json$")

# Thresholds explained by --explain; also used to grade the final report.
MIN_RUNNABLE = 8
MIN_NON_MEMORISING = 50
MIN_TRUSTWORTHY_AUROC = 125


class Side:
    """One class of the contrastive dataset and how to produce more of it."""

    def __init__(self, name: str, config: Path, job_script: Path, target: int):
        self.name = name
        self.config = config
        self.job_script = job_script
        self.target = target
        self.saturated = False
        self.empty_passes = 0
        self.passes_run = 0
        self.output_dir = _resolve_output_dir(config)

    def counts_by_task(self) -> dict[int, int]:
        found: dict[int, int] = collections.Counter()
        if not self.output_dir.is_dir():
            return dict(found)
        for path in self.output_dir.glob("task_*_trace_*.json"):
            m = TRACE_NAME.search(path.name)
            if m:
                found[int(m.group(1))] += 1
        return dict(found)

    def total(self) -> int:
        return sum(self.counts_by_task().values())

    def deficit(self) -> int:
        return max(0, self.target - self.total())


def _resolve_output_dir(config: Path) -> Path:
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(config)
    # output.dir interpolates ${oc.env:SCRATCH}; resolve it against this env.
    return Path(str(OmegaConf.to_container(cfg, resolve=True)["output"]["dir"]))


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--expert-target", type=int, default=150)
    ap.add_argument("--unsafe-target", type=int, default=150)
    ap.add_argument("--expert-config", type=Path,
                    default=REPO_ROOT / "configs/trajectory_generation/stwebagentbench_expert.yaml")
    ap.add_argument("--unsafe-config", type=Path,
                    default=REPO_ROOT / "configs/trajectory_collection/stwebagentbench_unsafe.yaml")
    ap.add_argument("--account", default=os.environ.get("ICRL_ACCOUNT", "aip-s2ganapa"))
    ap.add_argument("--gres", default="gpu:l40s:4",
                    help="for expert: the 72B planner needs tensor_parallel=4. "
                         "L40S schedules far sooner than the degraded h100 pool")
    ap.add_argument("--unsafe-gres", default="gpu:l40s:1",
                    help="the unsafe run is a 7B at tensor_parallel=1, so asking "
                         "for 4 GPUs wastes three and queues far slower")
    ap.add_argument("--time-limit", default="03:00:00")
    ap.add_argument("--max-passes", type=int, default=40,
                    help="hard stop on total passes, whatever the counts say")
    ap.add_argument("--stall-limit", type=int, default=2,
                    help="empty passes for one side before it is called saturated")
    ap.add_argument("--poll-seconds", type=int, default=120)
    ap.add_argument("--unsafe-parallel", type=int, default=4,
                    help="unsafe passes to run at once. Unsafe keeps an episode "
                         "for violating a policy, judged from the action sequence "
                         "— it needs neither a pristine database nor the app lock, "
                         "so its passes are embarrassingly parallel. Expert passes "
                         "are always serial: they reseed and must attribute state.")
    ap.add_argument("--cycles", type=int, default=8,
                    help="reseed-and-generate rounds inside each expert job. The "
                         "72B costs ~130s to load and re-queueing costs far more, "
                         "so one job doing many cycles beats many one-pass jobs.")
    ap.add_argument("--dry-run", action="store_true",
                    help="report current counts and the first action, submit nothing")
    ap.add_argument("--explain", action="store_true",
                    help="print the sizing rationale and exit")
    return ap.parse_args()


def explain() -> None:
    print(__doc__)
    print("""
Why the targets are what they are
---------------------------------
train_constraint.py freezes Qwen2.5-1.5B, mean-pools one embedding per
trajectory, and trains a ~400k-parameter MLP head (head_hidden: 256) for
n_iterations x n_constraint_steps = 5000 gradient steps at batch_size 8.

  * At ~15 traces per set each example is seen thousands of times: the head
    memorises, train AUROC hits 1.0 and held-out AUROC is noise.
  * auroc_gate: 0.75 is only meaningful with roughly 25+ per class in the eval
    split, i.e. ~125 per class at an 80/20 split.

Two things matter more than the totals:

  1. MATCHED TASKS. The head separates the classes by whatever is easiest. If the
     sets cover different tasks it learns task identity, scores a great AUROC and
     teaches C_theta nothing. Both configs must list the same task_ids.
  2. MEAN-POOLING. Averaging over the trajectory erases order, so ordering
     policies ("notes before status") are nearly invisible to the head, while
     presence/absence ones ("did it delete", "did it ask first") survive. Prefer
     contrasts of that shape.
""")


def submit_pass(side: Side, args: argparse.Namespace) -> list[str]:
    """Submit one expert pass, or --unsafe-parallel unsafe passes at once."""
    env = dict(os.environ, CONFIG=str(side.config))
    if side.name == "expert":
        # Reseeding makes each cycle's state attributable; cycles amortise the
        # model load. Both are wrong for unsafe: it wants varied, dirty state.
        env["RESEED_BEFORE_RUN"] = "1"
        env["CYCLES"] = str(args.cycles)
        n_jobs = 1
    else:
        n_jobs = max(1, args.unsafe_parallel)

    job_ids = []
    for _ in range(n_jobs):
        gres = args.gres if side.name == "expert" else args.unsafe_gres
        cmd = ["sbatch", "--parsable", f"--account={args.account}",
               f"--gres={gres}", f"--time={args.time_limit}",
               str(side.job_script)]
        out = subprocess.run(cmd, cwd=REPO_ROOT, env=env, capture_output=True,
                             text=True, check=True).stdout.strip()
        job_ids.append(out.split(";")[0])
    return job_ids


def job_finished(job_id: str) -> bool:
    """True once the job is neither queued nor running.

    An empty squeue result is NOT proof on its own — a failed connection looks
    the same — so it is confirmed against sacct before being believed.
    """
    queued = subprocess.run(["squeue", "-h", "-j", job_id, "-o", "%T"],
                            capture_output=True, text=True).stdout.strip()
    if queued:
        return False
    state = subprocess.run(["sacct", "-j", job_id, "-X", "-n", "-o", "State"],
                           capture_output=True, text=True).stdout.strip()
    if not state:
        return False
    return not any(s in state for s in ("PENDING", "RUNNING", "REQUEUED"))


def wait_for(job_ids: list[str], poll_seconds: int) -> None:
    remaining = list(job_ids)
    while remaining:
        time.sleep(poll_seconds)
        remaining = [j for j in remaining if not job_finished(j)]


def report(sides: list[Side], *, final: bool = False) -> dict:
    print("\n" + ("=" * 72))
    print("FINAL DATASET" if final else "PROGRESS")
    summary = {}
    for side in sides:
        by_task = side.counts_by_task()
        total = sum(by_task.values())
        grade = ("below the 8 needed to form a batch" if total < MIN_RUNNABLE else
                 "runnable, but the head will memorise" if total < MIN_NON_MEMORISING else
                 "trainable; AUROC still noisy" if total < MIN_TRUSTWORTHY_AUROC else
                 "enough for a trustworthy AUROC")
        print(f"\n  {side.name}: {total}/{side.target} — {grade}")
        print(f"    dir: {side.output_dir}")
        for task_id, n in sorted(by_task.items()):
            print(f"      task {task_id}: {n}")
        if side.saturated:
            print("    SATURATED — passes stopped adding traces; the environment, "
                  "not the target, is the limit")
        summary[side.name] = {"total": total, "target": side.target,
                              "by_task": by_task, "saturated": side.saturated,
                              "passes_run": side.passes_run}

    if final:
        expert_tasks = set(summary.get("expert", {}).get("by_task", {}))
        unsafe_tasks = set(summary.get("unsafe", {}).get("by_task", {}))
        shared = expert_tasks & unsafe_tasks
        print(f"\n  tasks in BOTH sets (usable as matched pairs): "
              f"{sorted(shared) if shared else 'NONE'}")
        only_expert = sorted(expert_tasks - unsafe_tasks)
        only_unsafe = sorted(unsafe_tasks - expert_tasks)
        if only_expert or only_unsafe:
            print(f"  unmatched — expert only {only_expert}, unsafe only {only_unsafe}")
            print("  WARNING: unmatched tasks let the head separate the classes by "
                  "task identity instead of by safety. Prefer training on the "
                  "matched subset.")
        if len(shared) < 3:
            print("  WARNING: very few shared tasks. C_theta risks learning "
                  "task-specific cues rather than a general constraint.")
    print("=" * 72, flush=True)
    return summary


def main() -> int:
    args = parse_args()
    if args.explain:
        explain()
        return 0

    sides = [
        Side("expert", args.expert_config,
             REPO_ROOT / "scripts/slurm/generate_trajectories_job.sh", args.expert_target),
        Side("unsafe", args.unsafe_config,
             REPO_ROOT / "scripts/slurm/collect_trajectories_job.sh", args.unsafe_target),
    ]

    from omegaconf import OmegaConf
    task_lists = {s.name: set(OmegaConf.load(s.config).benchmark.task_ids) for s in sides}
    expert_tasks, unsafe_tasks = task_lists["expert"], task_lists["unsafe"]
    print(f"\nexpert tasks: {sorted(expert_tasks)}\nunsafe tasks: {sorted(unsafe_tasks)}")
    if not expert_tasks & unsafe_tasks:
        print("WARNING: the two configs share NO tasks. The head will separate the "
              "classes by task identity, score a fine AUROC and teach C_theta "
              "nothing. Fix this before running.", flush=True)
    elif expert_tasks - unsafe_tasks:
        print(f"WARNING: expert-only tasks {sorted(expert_tasks - unsafe_tasks)} have "
              "no unsafe counterpart, so those traces are unpairable.", flush=True)
    elif unsafe_tasks - expert_tasks:
        # The intended shape: expert covers what the executor can actually
        # complete, unsafe covers more because failure is easy everywhere.
        print(f"note: unsafe also covers {sorted(unsafe_tasks - expert_tasks)}, where "
              "no expert trace is obtainable. Those add unpaired negatives — train "
              f"on the matched subset {sorted(expert_tasks & unsafe_tasks)} if the "
              "head starts keying on task identity.", flush=True)

    report(sides)
    total_passes = 0
    while total_passes < args.max_passes:
        pending = [s for s in sides if not s.saturated and s.deficit() > 0]
        if not pending:
            break
        # Advance whichever set is furthest from its target, keeping the classes
        # balanced rather than finishing one and starting the other.
        side = max(pending, key=lambda s: s.deficit())

        before = side.total()
        print(f"\n[pass {total_passes + 1}] {side.name}: {before}/{side.target} "
              f"(deficit {side.deficit()}) — submitting {side.job_script.name}",
              flush=True)
        if args.dry_run:
            print("DRY RUN — nothing submitted.")
            return 0

        try:
            job_ids = submit_pass(side, args)
        except subprocess.CalledProcessError as e:
            print(f"sbatch failed: {e.stderr.strip()}", flush=True)
            return 1
        print(f"  jobs {','.join(job_ids)} submitted; waiting", flush=True)
        wait_for(job_ids, args.poll_seconds)

        side.passes_run += len(job_ids)
        total_passes += len(job_ids)
        gained = side.total() - before
        print(f"  finished: +{gained} {side.name} traces", flush=True)
        if gained == 0:
            side.empty_passes += 1
            if side.empty_passes >= args.stall_limit:
                side.saturated = True
                print(f"  {side.name} saturated after {side.empty_passes} empty "
                      "passes — no longer targeting it", flush=True)
        else:
            side.empty_passes = 0
        report(sides)

    summary = report(sides, final=True)
    (Path(sides[0].output_dir).parent / "dataset_build_summary.json").write_text(
        json.dumps(summary, indent=2))
    print("\nNext: verify and publish with\n"
          "  python scripts/push_trajectories_to_huggingface.py "
          "--traj-dir <dir> --set <expert_synthetic|unsafe> --namespace <ns>")
    return 0


if __name__ == "__main__":
    sys.exit(main())
