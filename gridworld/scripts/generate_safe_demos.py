"""Generate safe demonstrations for ST-WebAgentBench by running an LLM policy.

The LLM follows the safe system prompt (requests confirmation before irreversible
actions), producing genuinely safe trajectories for use as ICRL positive examples.

Usage:
    export TOGETHER_API_KEY=your-key-here
    python scripts/generate_safe_demos.py --task-ids 235 --output data/stwebagent_safe_demos.jsonl
    python scripts/generate_safe_demos.py --task-ids 235,236,237 --n-per-task 3 --output data/demos.jsonl
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from icrl.envs.stwebagent import STWebAgentEnv, STWebAgentEnvConfig
from icrl.policies.llm_policy import LLMPolicy, LLMPolicyConfig
from icrl.trainer.rollout_buffer import collect_rollouts

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _traj_to_record(traj, task_id: str) -> dict:
    return {
        "episode_id": traj.episode_id,
        "task_id": task_id,
        "is_safe": True,
        "total_reward": traj.total_reward,
        "n_steps": len(traj.transitions),
        "transitions": [
            {
                "obs": t.obs,
                "action": t.action,
                "reward": t.reward,
                "next_obs": t.next_obs,
                "done": t.done,
                "info": {
                    k: v
                    for k, v in t.info.items()
                    if k not in ("screenshot",)
                },
            }
            for t in traj.transitions
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate safe ST-WebAgentBench demos")
    parser.add_argument("--task-ids", required=True, help="Comma-separated task IDs")
    parser.add_argument("--output", required=True, help="Output JSONL file path")
    parser.add_argument("--model", default="Qwen/Qwen3.5-72B-Instruct")
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--no-headless", dest="headless", action="store_false")
    parser.add_argument("--n-per-task", type=int, default=5)
    parser.add_argument("--completion-bonus", type=float, default=10.0)
    parser.add_argument("--step-cost", type=float, default=0.1)
    parser.add_argument("--max-steps", type=int, default=50)
    args = parser.parse_args()

    task_ids = [t.strip() for t in args.task_ids.split(",")]
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    policy_cfg = LLMPolicyConfig(model=args.model, temperature=0.0)
    policy = LLMPolicy(policy_cfg)

    total_collected = 0
    total_attempted = 0
    all_rewards: list[float] = []

    with open(args.output, "w") as out_f:
        for task_id in task_ids:
            logger.info(f"Task {task_id}: collecting {args.n_per_task} safe demos")
            env_cfg = STWebAgentEnvConfig(
                task_id=task_id,
                headless=args.headless,
                completion_bonus=args.completion_bonus,
                step_cost=args.step_cost,
                max_steps=args.max_steps,
            )

            collected_this_task = 0
            for attempt in range(args.n_per_task * 3):
                if collected_this_task >= args.n_per_task:
                    break

                env = STWebAgentEnv(env_cfg)
                try:
                    rollouts = collect_rollouts(env, policy, n_steps=args.max_steps)
                    total_attempted += len(rollouts)

                    for traj in rollouts:
                        if traj.total_reward > 0:
                            record = _traj_to_record(traj, task_id)
                            out_f.write(json.dumps(record) + "\n")
                            out_f.flush()
                            collected_this_task += 1
                            all_rewards.append(traj.total_reward)
                            logger.info(
                                f"  ✓ Saved demo {collected_this_task}/{args.n_per_task} "
                                f"(reward={traj.total_reward:.2f}, steps={len(traj.transitions)})"
                            )
                        else:
                            logger.info(
                                f"  ✗ Episode failed (reward={traj.total_reward:.2f}) — skipping"
                            )
                finally:
                    env.close()

            total_collected += collected_this_task
            logger.info(
                f"Task {task_id}: {collected_this_task}/{args.n_per_task} demos collected"
            )

    if all_rewards:
        import statistics
        print(f"\nCollection complete:")
        print(f"  Demos collected : {total_collected}")
        print(f"  Episodes run    : {total_attempted}")
        print(f"  Success rate    : {total_collected/total_attempted:.1%}")
        print(f"  Mean reward     : {statistics.mean(all_rewards):.2f}")
        print(f"  Max reward      : {max(all_rewards):.2f}")
        print(f"  Min reward      : {min(all_rewards):.2f}")
        print(f"  Output          : {args.output}")
    else:
        print("No successful demos collected. Check your API key and task IDs.")
        sys.exit(1)


if __name__ == "__main__":
    main()
