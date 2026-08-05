"""
Adversarial ICRL training loop.

Four steps per iteration:
  1. Collect rollouts from current policy
  2. Adversarial detection: flag r(τ) > threshold as implicitly unsafe
  3. Constraint update: contrastive training on safe demos vs flagged unsafe
  4. Policy update: PPO-Lag using current constraint cost signal

The key insight (paper #3): since safe demos are near-optimal,
any trajectory exceeding their reward must have skipped required
safety steps.  No unsafe demonstrations are needed as input.
"""
from __future__ import annotations

import logging
import os
import random
from dataclasses import dataclass, field

import numpy as np
from tqdm import tqdm

from icrl.core.interfaces import BaseConstraint, BaseEnv, BasePolicy
from icrl.core.types import DemoDataset, Trajectory
from icrl.metrics.tracker import MetricsTracker
from icrl.trainer.adversarial_detector import AdversarialDetector
from icrl.trainer.rollout_buffer import collect_rollouts

logger = logging.getLogger(__name__)


@dataclass
class ICRLConfig:
    n_iterations: int = 100
    n_rollout_steps: int = 2048
    n_constraint_epochs: int = 5
    constraint_batch_size: int = 16
    min_unsafe_for_update: int = 3
    unsafe_buffer_max_size: int = 500
    eval_every: int = 10
    checkpoint_every: int = 25
    log_dir: str = "runs/icrl"
    seed: int = 42
    # Run unconstrained PPO first so the policy learns to complete the task.
    # After this many iterations the constraint is switched on.
    pretrain_iterations: int = 0


class ICRLTrainer:
    def __init__(
        self,
        env: BaseEnv,
        policy: BasePolicy,
        constraint: BaseConstraint,
        demos: DemoDataset,
        detector: AdversarialDetector,
        config: ICRLConfig,
    ):
        self.env = env
        self.policy = policy
        self.constraint = constraint
        self.detector = detector
        self.config = config
        self.demos = demos
        self.metrics = MetricsTracker(config.log_dir)

        self.detector.fit(demos)
        # Constraint is injected after pretrain phase; start unconstrained.
        if config.pretrain_iterations > 0:
            self.policy.set_constraint(None)
        else:
            self.policy.set_constraint(constraint)

        self._unsafe_buffer: list[Trajectory] = []
        self._last_rollouts: list[Trajectory] = []
        self._iteration = 0

        logger.info(
            "ICRLTrainer ready. "
            f"safe_demos={len(demos.safe)}  threshold={self.detector.threshold:.3f}  "
            f"pretrain_iters={config.pretrain_iterations}"
        )

    def train(self) -> None:
        for iteration in tqdm(range(self.config.n_iterations), desc="ICRL"):
            self._iteration = iteration
            in_pretrain = iteration < self.config.pretrain_iterations

            # Switch constraint on after pretrain phase
            if iteration == self.config.pretrain_iterations and self.config.pretrain_iterations > 0:
                self.policy.set_constraint(self.constraint)
                logger.info(f"[iter {iteration}] Pretrain complete — constraint activated.")

            # ── 1. Collect rollouts ───────────────────────────────────
            rollouts = collect_rollouts(self.env, self.policy, self.config.n_rollout_steps)
            self._last_rollouts = rollouts

            # ── 2 & 3. Adversarial detection + constraint update ──────
            c_metrics: dict = {}
            if not in_pretrain:
                _, newly_unsafe = self.detector.filter_unsafe(rollouts)
                self._unsafe_buffer.extend(newly_unsafe)
                if len(self._unsafe_buffer) > self.config.unsafe_buffer_max_size:
                    self._unsafe_buffer = self._unsafe_buffer[-self.config.unsafe_buffer_max_size :]

                if len(self._unsafe_buffer) >= self.config.min_unsafe_for_update:
                    c_metrics = self._update_constraint()
            else:
                newly_unsafe = []

            # ── 4. Policy update ──────────────────────────────────────
            p_metrics = self.policy.update(rollouts)

            # ── 5. Logging ────────────────────────────────────────────
            mean_reward = (
                float(np.mean([t.total_reward for t in rollouts])) if rollouts else 0.0
            )
            step_metrics = {
                "iteration": iteration,
                "in_pretrain": int(in_pretrain),
                "n_rollout_episodes": len(rollouts),
                "n_newly_unsafe": len(newly_unsafe),
                "unsafe_buffer_size": len(self._unsafe_buffer),
                "mean_reward": mean_reward,
                "safe_threshold": self.detector.threshold,
                **c_metrics,
                **p_metrics,
            }
            self.metrics.log(step_metrics, step=iteration)

            if iteration % self.config.eval_every == 0:
                self._log_summary(step_metrics)

            if iteration % self.config.checkpoint_every == 0 and iteration > 0:
                self._checkpoint(iteration)

        self.metrics.close()

    def _update_constraint(self) -> dict:
        metrics: dict = {}
        for _ in range(self.config.n_constraint_epochs):
            half = max(1, self.config.constraint_batch_size // 2)
            safe_batch = random.sample(self.demos.safe, min(half, len(self.demos.safe)))
            unsafe_batch = random.sample(
                self._unsafe_buffer, min(half, len(self._unsafe_buffer))
            )
            metrics = self.constraint.update(safe_batch, unsafe_batch)
        return metrics

    def _log_summary(self, m: dict) -> None:
        logger.info(
            f"[iter {m['iteration']:4d}] "
            f"reward={m['mean_reward']:.2f}  "
            f"unsafe_buf={m['unsafe_buffer_size']}  "
            f"c_loss={m.get('constraint_loss', '-')!s:.6}  "
            f"gap={m.get('feasibility_gap', '-')!s:.4}  "
            f"λ={m.get('lambda', '-')!s:.4}"
        )

    def _checkpoint(self, iteration: int) -> None:
        import torch

        os.makedirs(self.config.log_dir, exist_ok=True)
        path = os.path.join(self.config.log_dir, f"ckpt_{iteration:04d}.pt")
        torch.save(
            {
                "iteration": iteration,
                "policy": self.policy.state_dict(),
                "constraint": self.constraint.state_dict(),
            },
            path,
        )
        logger.debug(f"Checkpoint saved: {path}")
