#!/usr/bin/env python3
"""
Collect WebArena demos using AgentLab/BrowserGym.

Two modes:

  --from-hf       Download pre-recorded BrowserGym traces from HuggingFace,
                  apply safety labeling, write safe/unsafe JSONL.

  --run-agentlab  Run a live AgentLab study (GPT-4o by default) on the 179
                  WebArena tasks at N rollouts each, filter terminated episodes,
                  apply safety labeling, write safe/unsafe JSONL.

Usage:
    # Bootstrap from HuggingFace pre-recorded traces:
    python scripts/collect_webarena_demos.py \\
        --from-hf \\
        --hf-dataset ServiceNow/browsergym-webarena-traces \\
        --output data/demos/webarena_safe.jsonl

    # Fresh GPT-4o rollouts (requires OPENAI_API_KEY):
    python scripts/collect_webarena_demos.py \\
        --run-agentlab \\
        --model gpt-4o \\
        --n-rollouts 5 \\
        --n-jobs 8 \\
        --output data/demos/webarena_safe.jsonl

    # Smoke test — single rollout per task, 2 workers:
    python scripts/collect_webarena_demos.py \\
        --run-agentlab --n-rollouts 1 --n-jobs 2 \\
        --output data/demos/smoke.jsonl
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_OUTPUT = Path("data/demos/webarena_safe.jsonl")
DEFAULT_UNSAFE = Path("data/demos/webarena_unsafe.jsonl")
DEFAULT_STUDY_DIR = Path("data/agentlab_study")
DEFAULT_HF_DATASET = "ServiceNow/browsergym-webarena-traces"


def cmd_from_hf(args):
    """Download HF traces, apply safety labeling, write JSONL."""
    from src.data.agentlab_loader import download_hf_webarena_traces, load_agentlab_study
    from src.data.safety_verifier import SafetyVerifier
    from src.data.trajectory import save_trajectories

    hf_local = Path(args.hf_local_dir)
    logger.info("Downloading traces from HuggingFace: %s", args.hf_dataset)
    trace_dir = download_hf_webarena_traces(hf_local, dataset_id=args.hf_dataset)

    logger.info("Loading traces from %s", trace_dir)
    all_trajs = load_agentlab_study(trace_dir)

    # Filter near-optimal
    terminated = [t for t in all_trajs if t.terminated and (t.reward or 0) >= args.min_reward]
    logger.info("%d/%d trajectories are terminated with reward >= %.1f",
                len(terminated), len(all_trajs), args.min_reward)

    _label_and_save(terminated, args)


def cmd_run_agentlab(args):
    """Run a live AgentLab study and write safety-labeled JSONL."""
    from src.data.demo_collector import AgentLabDemoCollector

    collector = AgentLabDemoCollector(
        output_dir=str(args.output.parent),
        model=args.model,
        min_confidence=args.min_confidence,
    )
    safe, unsafe = collector.collect(
        n_rollouts_per_task=args.n_rollouts,
        n_jobs=args.n_jobs,
        min_reward=args.min_reward,
        study_dir=str(args.study_dir),
        safe_output_path=str(args.output),
        unsafe_output_path=str(args.unsafe_output),
    )
    _print_summary(safe, unsafe, args.output, args.unsafe_output)


def _label_and_save(trajs, args):
    """Apply SafetyVerifier to a list of trajectories and write JSONL files."""
    from src.data.safety_verifier import SafetyVerifier
    from src.data.trajectory import save_trajectories

    verifier = SafetyVerifier()
    safe, unsafe = [], []

    for traj in trajs:
        task_desc = {"task_id": traj.task_instance_id, "constraint_description": "",
                     "policies": []}
        result = verifier.verify(task_desc, traj.steps)

        if result.confidence < args.min_confidence:
            continue

        traj.is_safe = result.is_safe
        traj.constraint_score = result.confidence

        (safe if result.is_safe else unsafe).append(traj)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if safe:
        save_trajectories(safe, str(args.output))
    if unsafe:
        save_trajectories(unsafe, str(args.unsafe_output))

    _print_summary(safe, unsafe, args.output, args.unsafe_output)


def _print_summary(safe, unsafe, safe_path, unsafe_path):
    total = len(safe) + len(unsafe)
    print(f"\n{'─'*52}")
    print(f"Total labeled : {total}")
    print(f"Safe          : {len(safe)}  → {safe_path}")
    print(f"Unsafe        : {len(unsafe)}  → {unsafe_path}")
    if total:
        avg_steps = sum(len(t.steps) for t in safe + unsafe) / total
        print(f"Avg steps     : {avg_steps:.1f}")
    print(f"""
╔══════════════════════════════════════════════════════╗
║  Next steps                                          ║
╠══════════════════════════════════════════════════════╣
║  python scripts/train_constraint.py \\               ║
║      +constraint=icrl_default run_name=<name>        ║
╚══════════════════════════════════════════════════════╝""")


def main():
    parser = argparse.ArgumentParser(
        description="Collect WebArena demos via AgentLab/BrowserGym",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--from-hf", action="store_true",
                      help="Download pre-recorded BrowserGym traces from HuggingFace")
    mode.add_argument("--run-agentlab", action="store_true",
                      help="Run a live AgentLab study on WebArena tasks")

    # Shared output paths
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help=f"JSONL path for safe demos (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--unsafe-output", type=Path, default=DEFAULT_UNSAFE,
                        help=f"JSONL path for unsafe demos (default: {DEFAULT_UNSAFE})")

    # HF mode options
    parser.add_argument("--hf-dataset", type=str, default=DEFAULT_HF_DATASET,
                        help=f"HuggingFace dataset id (default: {DEFAULT_HF_DATASET})")
    parser.add_argument("--hf-local-dir", type=str, default="data/hf_traces/webarena",
                        help="Local directory for HF snapshot (default: data/hf_traces/webarena)")

    # AgentLab mode options
    parser.add_argument("--model", type=str, default="gpt-4o",
                        help="OpenAI model for AgentLab agent (default: gpt-4o)")
    parser.add_argument("--n-rollouts", type=int, default=5,
                        help="Rollouts per WebArena task (default: 5)")
    parser.add_argument("--n-jobs", type=int, default=8,
                        help="Parallel worker processes (default: 8)")
    parser.add_argument("--study-dir", type=Path, default=DEFAULT_STUDY_DIR,
                        help=f"AgentLab study output dir (default: {DEFAULT_STUDY_DIR})")

    # Safety filter options
    parser.add_argument("--min-reward", type=float, default=1.0,
                        help="Minimum reward to keep a terminated episode (default: 1.0)")
    parser.add_argument("--min-confidence", type=float, default=0.7,
                        help="Minimum SafetyVerifier confidence to keep verdict (default: 0.7)")

    args = parser.parse_args()

    if args.from_hf:
        cmd_from_hf(args)
    else:
        cmd_run_agentlab(args)


if __name__ == "__main__":
    main()
