#!/usr/bin/env python3
"""
Collect ST-WebAgentBench demonstrations using the Qwen model family.

Two modes:

  --mode safe     Run Qwen-72B with policy-aware system prompt + best-of-N.
                  Keep only terminated=True AND CuP=1 episodes (ground-truth).
                  Target: 40-70 safe demos across 222 tasks.

  --mode unsafe   Run Qwen-7B WITHOUT safety prompt.
                  Keep episodes with ≥1 policy violation.
                  Target: ~200 unsafe demos. Model incompetence is the signal.

Inference backends:
  Cloud  (OpenRouter):  omit --vllm-base-url  (uses OPENROUTER_API_KEY)
  Cluster (vLLM):       --vllm-base-url http://localhost:8000/v1

Usage:
    # Local smoke test via OpenRouter (2 rollouts, 5 tasks):
    python scripts/collect_stwebagent_demos.py \\
        --mode safe \\
        --model qwen/qwen-2.5-72b-instruct \\
        --n-rollouts 2 --task-limit 5 \\
        --output data/demos/smoke_safe.jsonl \\
        --benchmark-root /path/to/stwebagentbench

    # Full safe collection on cluster (vLLM):
    python scripts/collect_stwebagent_demos.py \\
        --mode safe \\
        --model Qwen/Qwen2.5-72B-Instruct \\
        --vllm-base-url http://localhost:8000/v1 \\
        --n-rollouts 10 \\
        --output data/demos/stwebagent_safe.jsonl

    # Unsafe collection on cluster:
    python scripts/collect_stwebagent_demos.py \\
        --mode unsafe \\
        --model Qwen/Qwen2.5-7B-Instruct \\
        --vllm-base-url http://localhost:8000/v1 \\
        --n-rollouts 5 \\
        --output data/demos/stwebagent_unsafe.jsonl

See slurm/collect_safe_demos.sh and slurm/collect_unsafe_demos.sh for the
full SLURM workflow that starts vLLM and runs this script.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_SAFE_OUTPUT = Path("data/demos/stwebagent_safe.jsonl")
DEFAULT_UNSAFE_OUTPUT = Path("data/demos/stwebagent_unsafe.jsonl")
DEFAULT_SAFE_MODEL = "Qwen/Qwen2.5-72B-Instruct"
DEFAULT_UNSAFE_MODEL = "Qwen/Qwen2.5-7B-Instruct"


def main():
    parser = argparse.ArgumentParser(
        description="Collect ST-WebAgentBench demos using Qwen family models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--mode", choices=["safe", "unsafe"], required=True,
                        help="safe: policy-aware Qwen-72B, CuP=1 filter; "
                             "unsafe: naive Qwen-7B, violation filter")
    parser.add_argument("--model", type=str, default=None,
                        help=f"Model name. Defaults: safe={DEFAULT_SAFE_MODEL}, "
                             f"unsafe={DEFAULT_UNSAFE_MODEL}")
    parser.add_argument("--vllm-base-url", type=str, default=None,
                        help="vLLM OpenAI-compatible endpoint URL. "
                             "Omit to use OpenRouter (OPENROUTER_API_KEY required).")
    parser.add_argument("--n-rollouts", type=int, default=None,
                        help="Rollouts per task (defaults: safe=10, unsafe=5)")
    parser.add_argument("--max-steps", type=int, default=30,
                        help="Max steps per episode (default: 30)")
    parser.add_argument("--benchmark-root", type=str, default=None,
                        help="Path to ST-WebAgentBench installation root. "
                             "Required when not discoverable via STWEBAGENT_ROOT env var.")
    parser.add_argument("--task-ids", nargs="+", type=str, default=None,
                        help="Explicit task IDs to run (default: all tasks)")
    parser.add_argument("--task-limit", type=int, default=None,
                        help="Max number of tasks to process (for smoke tests)")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output JSONL path")
    args = parser.parse_args()

    # Resolve defaults based on mode
    if args.model is None:
        args.model = DEFAULT_SAFE_MODEL if args.mode == "safe" else DEFAULT_UNSAFE_MODEL
    if args.n_rollouts is None:
        args.n_rollouts = 10 if args.mode == "safe" else 5
    if args.output is None:
        args.output = DEFAULT_SAFE_OUTPUT if args.mode == "safe" else DEFAULT_UNSAFE_OUTPUT

    # Resolve benchmark root
    import os
    benchmark_root = args.benchmark_root or os.environ.get("STWEBAGENT_ROOT", "")

    # Build benchmark
    from src.data.st_webagent import STWebAgentBench
    bench = STWebAgentBench(benchmark_root)

    # Build collector
    from src.data.demo_collector import STWebAgentDemoCollector
    collector = STWebAgentDemoCollector(
        benchmark=bench,
        model=args.model,
        n_rollouts=args.n_rollouts,
        max_steps=args.max_steps,
        vllm_base_url=args.vllm_base_url,
    )

    # Resolve task IDs
    task_ids = args.task_ids
    if task_ids is None and args.task_limit is not None:
        all_ids = list(bench.load_tasks().keys())
        task_ids = all_ids[: args.task_limit]
        logger.info("Task limit: using first %d of %d tasks", len(task_ids), len(all_ids))

    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Run
    if args.mode == "safe":
        logger.info("Starting SAFE collection: model=%s, n_rollouts=%d, output=%s",
                    args.model, args.n_rollouts, args.output)
        trajs = collector.collect_safe(str(args.output), task_ids=task_ids)
    else:
        logger.info("Starting UNSAFE collection: model=%s, n_rollouts=%d, output=%s",
                    args.model, args.n_rollouts, args.output)
        trajs = collector.collect_unsafe(str(args.output), task_ids=task_ids)

    # Summary
    print(f"\n{'─'*52}")
    print(f"Mode          : {args.mode}")
    print(f"Model         : {args.model}")
    print(f"Demos written : {len(trajs)}")
    print(f"Output        : {args.output}")
    if trajs:
        avg_steps = sum(len(t.steps) for t in trajs) / len(trajs)
        safe_count = sum(1 for t in trajs if t.is_safe)
        term_count = sum(1 for t in trajs if t.terminated)
        print(f"Avg steps     : {avg_steps:.1f}")
        print(f"Safe / total  : {safe_count}/{len(trajs)}")
        print(f"Terminated    : {term_count}/{len(trajs)}")
    print(f"""
╔══════════════════════════════════════════════════════╗
║  Next steps                                          ║
╠══════════════════════════════════════════════════════╣
║  python scripts/train_constraint.py \\               ║
║      +constraint=icrl_default run_name=<name>        ║
╚══════════════════════════════════════════════════════╝""")


if __name__ == "__main__":
    main()
