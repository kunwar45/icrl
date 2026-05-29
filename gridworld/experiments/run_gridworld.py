"""
Adversarial ICRL on the 4×4 GridWorld — spatial constraint recovery.

The hidden constraint:  agents must avoid cells (1,0) and (2,0).
Safe demos go via col 1  (5 steps, reward=15).
Optimal shortcut via col 0 (3 steps, reward=17) is implicitly unsafe.

The ICRL loop should recover: low feasibility on col-0 paths, high on col-1 paths.

Usage:
    python experiments/run_gridworld.py
    python experiments/run_gridworld.py --config configs/experiment/gridworld_smoke.yaml
    python experiments/run_gridworld.py --no-constraint   # plain PPO baseline
"""
from __future__ import annotations

import argparse
import logging
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from icrl.utils.config import load_config
from icrl.utils.seeding import seed_everything
from icrl.envs.gridworld import GridWorldConfig, GridWorldEnv
from icrl.demos.gridworld_demos import GridWorldDemoConfig, GridWorldDemoLoader
from icrl.embedders.mean_pool import MeanPoolEmbedderConfig, MeanPoolEmbedder
from icrl.constraints.mlp_constraint import MLPConstraintConfig, MLPConstraint
from icrl.policies.ppo_lagrangian import PPOLagConfig, PPOLagPolicy
from icrl.trainer.adversarial_detector import AdversarialDetectorConfig, AdversarialDetector
from icrl.trainer.icrl_trainer import ICRLConfig, ICRLTrainer
from icrl.utils.viz import _make_canonical_trajectories, print_training_report

import torch
from icrl.core.interfaces import BaseConstraint
from icrl.core.types import Trajectory

logging.basicConfig(
    level=logging.WARNING,          # suppress tqdm/trainer INFO spam; we use viz instead
    format="%(levelname)-8s  %(message)s",
)


class _VisualizingTrainer(ICRLTrainer):
    """ICRLTrainer subclass that prints a rich terminal report at each eval step."""

    def __init__(self, *args, canonical_safe, canonical_unsafe, threshold, **kwargs):
        super().__init__(*args, **kwargs)
        self._canonical_safe = canonical_safe
        self._canonical_unsafe = canonical_unsafe
        self._threshold = threshold

    def _log_summary(self, m: dict) -> None:
        in_pretrain = bool(m.get("in_pretrain", 0))
        print_training_report(
            iteration=m["iteration"],
            total_iters=self.config.n_iterations,
            in_pretrain=in_pretrain,
            metrics=m,
            recent_trajs=self._last_rollouts,
            threshold=self._threshold,
            constraint=self.constraint if not in_pretrain else None,
            canonical_safe=self._canonical_safe,
            canonical_unsafe=self._canonical_unsafe,
        )


class _NullConstraint(BaseConstraint):
    def feasibility(self, trajectory: Trajectory) -> torch.Tensor:
        return torch.tensor(1.0)

    def cost(self, trajectory: Trajectory) -> torch.Tensor:
        return torch.tensor(0.0)

    def update(self, safe, unsafe) -> dict:
        return {}

    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, state: dict) -> None:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Adversarial ICRL — GridWorld")
    parser.add_argument("--config", default="configs/experiment/gridworld_smoke.yaml")
    parser.add_argument("--no-constraint", action="store_true",
                        help="Disable constraint (plain PPO baseline)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(cfg.get("seed", 42))

    # ── Environment ───────────────────────────────────────────────────
    env = GridWorldEnv(GridWorldConfig(**cfg.get("env", {})))

    print(f"\n4×4 GridWorld — adversarial constraint recovery")
    print(f"  Safe path   (via col 1): reward = {env.safe_episode_reward():.1f}")
    print(f"  Unsafe path (via col 0): reward = {env.unsafe_episode_reward():.1f}")
    print(f"  Reward gap (unsafe − safe) = {env.unsafe_episode_reward() - env.safe_episode_reward():.1f}")
    print(f"  Adversarial threshold: {env.safe_episode_reward():.1f}\n")

    # ── Demonstrations ────────────────────────────────────────────────
    demos = GridWorldDemoLoader(GridWorldDemoConfig(**cfg.get("demo", {}))).load()
    print(f"Loaded {len(demos.safe)} safe demos  "
          f"(threshold = {demos.safe_reward_threshold:.1f})")

    # ── Canonical trajectories for constraint probing ─────────────────
    canonical_safe, canonical_unsafe = _make_canonical_trajectories(env)
    threshold = demos.safe_reward_threshold

    # ── Embedder ──────────────────────────────────────────────────────
    embedder = MeanPoolEmbedder(MeanPoolEmbedderConfig(**cfg.get("embedder", {})))
    embedder.build_vocab(demos.safe, env)
    print(f"Vocabulary: {len(embedder._vocab)} tokens\n")

    # ── Constraint ────────────────────────────────────────────────────
    constraint: BaseConstraint
    if args.no_constraint:
        constraint = _NullConstraint()
    else:
        constraint = MLPConstraint(embedder, env, MLPConstraintConfig(**cfg.get("constraint", {})))

    # ── Policy ────────────────────────────────────────────────────────
    policy = PPOLagPolicy(env.obs_dim, env.n_actions, PPOLagConfig(**cfg.get("policy", {})))

    # ── Detector ─────────────────────────────────────────────────────
    detector = AdversarialDetector(AdversarialDetectorConfig(**cfg.get("detector", {})))

    # ── Trainer ───────────────────────────────────────────────────────
    tr_cfg_dict = dict(cfg.get("trainer", {}))
    if args.no_constraint:
        tr_cfg_dict["log_dir"] = tr_cfg_dict.get("log_dir", "runs") + "_no_constraint"
        tr_cfg_dict["pretrain_iterations"] = tr_cfg_dict.get("n_iterations", 100)

    tr_cfg = ICRLConfig(**tr_cfg_dict)

    trainer = _VisualizingTrainer(
        env=env,
        policy=policy,
        constraint=constraint,
        demos=demos,
        detector=detector,
        config=tr_cfg,
        canonical_safe=canonical_safe,
        canonical_unsafe=canonical_unsafe,
        threshold=threshold,
    )

    mode = "BASELINE (plain PPO)" if args.no_constraint else "ICRL (adversarial constraint)"
    print(f"Running: {mode}")
    print(f"Metrics → {tr_cfg.log_dir}/metrics.jsonl\n")

    trainer.train()
    print(f"\nDone. Metrics saved to {tr_cfg.log_dir}/metrics.jsonl")


if __name__ == "__main__":
    main()
