"""
Adversarial ICRL on ST-WebAgentBench with an LLM (Qwen3.5 via Together AI) policy.

Usage:
    python experiments/run_stwebagent.py
    python experiments/run_stwebagent.py --config configs/experiment/stwebagent.yaml

Prerequisites:
    export TOGETHER_API_KEY=your-key-here
    pip install -e ".[stwebagent]"
    playwright install chromium
    python scripts/generate_safe_demos.py --task-ids 235 --output data/stwebagent_safe_demos.jsonl

What to watch in runs/stwebagent_235/metrics.jsonl:
  - unsafe_buffer_size  : grows as LLM skips confirmations, falls as constraint guides it back
  - constraint_loss     : drops as MLP learns to separate safe / unsafe trajectories
  - feasibility_gap     : safe_feasibility - unsafe_feasibility should increase
  - mean_reward         : converges toward safe-demo threshold (~8.2) not unsafe ceiling (~8.5)
"""
from __future__ import annotations

import argparse
import logging
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from icrl.utils.config import load_config
from icrl.utils.seeding import seed_everything
from icrl.envs.stwebagent import STWebAgentEnvConfig, STWebAgentEnv
from icrl.demos.stwebagent_demos import STWebAgentDemoConfig, STWebAgentDemoLoader
from icrl.embedders.sentence_transformer import (
    SentenceTransformerEmbedderConfig,
    SentenceTransformerEmbedder,
)
from icrl.constraints.mlp_constraint import MLPConstraintConfig, MLPConstraint
from icrl.policies.llm_policy import LLMPolicyConfig, LLMPolicy
from icrl.trainer.adversarial_detector import AdversarialDetectorConfig, AdversarialDetector
from icrl.trainer.icrl_trainer import ICRLConfig, ICRLTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Adversarial ICRL — ST-WebAgentBench")
    parser.add_argument(
        "--config",
        default="configs/experiment/stwebagent.yaml",
        help="Path to YAML experiment config",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = cfg.get("seed", 42)
    seed_everything(seed)

    # ── Environment ───────────────────────────────────────────────────
    env_cfg = STWebAgentEnvConfig(**cfg.get("env", {}))
    env = STWebAgentEnv(env_cfg)
    print(f"ST-WebAgentBench task: {env_cfg.task_id}")
    print(f"  Completion bonus: {env_cfg.completion_bonus}")
    print(f"  Step cost: {env_cfg.step_cost}")
    print(
        f"  Safe demo expected reward ≈ {env_cfg.completion_bonus - 18 * env_cfg.step_cost:.1f}"
        " (18 steps, including confirmations)"
    )
    print(
        f"  Unsafe ceiling ≈ {env_cfg.completion_bonus - 15 * env_cfg.step_cost:.1f}"
        " (15 steps, skipped confirmations)"
    )
    print()

    # ── Demonstrations ────────────────────────────────────────────────
    demo_cfg = STWebAgentDemoConfig(**cfg.get("demo", {}))
    demos = STWebAgentDemoLoader(demo_cfg).load()
    print(f"Loaded {demos}")

    # ── Embedder ──────────────────────────────────────────────────────
    emb_cfg = SentenceTransformerEmbedderConfig(**cfg.get("embedder", {}))
    embedder = SentenceTransformerEmbedder(emb_cfg)
    print(f"Embedder: {emb_cfg.model_name}  dim={embedder.embed_dim}")

    # ── Constraint ────────────────────────────────────────────────────
    con_cfg = MLPConstraintConfig(**cfg.get("constraint", {}))
    constraint = MLPConstraint(embedder, env, con_cfg)

    # ── Policy ────────────────────────────────────────────────────────
    pol_cfg = LLMPolicyConfig(**cfg.get("policy", {}))
    policy = LLMPolicy(pol_cfg)
    print(f"Policy: {pol_cfg.model}")

    # ── Detector ─────────────────────────────────────────────────────
    det_cfg = AdversarialDetectorConfig(**cfg.get("detector", {}))
    detector = AdversarialDetector(det_cfg)

    # ── Trainer ───────────────────────────────────────────────────────
    tr_cfg = ICRLConfig(**cfg.get("trainer", {}))

    trainer = ICRLTrainer(
        env=env,
        policy=policy,
        constraint=constraint,
        demos=demos,
        detector=detector,
        config=tr_cfg,
    )

    print("Running ICRL on ST-WebAgentBench")
    print(f"Metrics → {tr_cfg.log_dir}/metrics.jsonl\n")

    try:
        trainer.train()
    finally:
        env.close()

    print(f"\nDone. Metrics saved to {tr_cfg.log_dir}/metrics.jsonl")


if __name__ == "__main__":
    main()
