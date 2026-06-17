"""
ICRL constraint trainer.

Two alternating phases:
  Phase 1 — collect policy rollouts (or use pre-collected buffer in offline mode)
  Phase 2 — constraint_update: push policy scores HIGH, anchor expert scores LOW

Loss (Phase 2):
    L = -E[Cθ(D_policy)] + λ_c · ReLU(E[Cθ(D_expert)] − β)

The ReLU term only activates when expert scores drift above β, acting as a soft
anchor. When expert scores are comfortably below β it contributes zero and the
only gradient signal is pushing policy scores up.

Three numbers to watch every iteration:
  policy_constraint_score  — should rise as Cθ learns to flag policy behavior
  expert_constraint_score  — should stay below β; if it climbs, raise lambda_c
  task_reward              — mean reward of policy trajectories in the pool
"""
from __future__ import annotations

import random
from typing import Callable, Optional

import torch
import torch.nn.functional as F
from torch.optim import AdamW

from src.constraint.encoder import TrajectoryEncoder
from src.constraint.evaluator import ConstraintEvaluator
from src.data.trajectory import Trajectory
from src.utils.logging import get_logger, MetricsLogger

logger = get_logger(__name__)


# ── Core update function ──────────────────────────────────────────────────────

def constraint_update(
    ctheta: TrajectoryEncoder,
    policy_embeddings: torch.Tensor,
    expert_embeddings: torch.Tensor,
    beta: float,
    lambda_c: float,
    n_steps: int,
    lr: float,
    weight_decay: float,
    batch_size: int,
) -> dict:
    """
    Phase 2: M gradient steps on the constraint function head.

    Accepts pre-computed embeddings (frozen backbone output).  Since the
    backbone never changes, calling embed_texts() once before this function
    and reusing the tensors is mathematically identical to re-running the
    backbone each step — but orders of magnitude faster.

    Args:
        ctheta:              constraint model (only head.parameters() are updated)
        policy_embeddings:   (N_policy, H) tensor — backbone output for policy trajs
        expert_embeddings:   (N_expert, H) tensor — backbone output for expert demos
        beta:                expert score anchor threshold (e.g. 0.8)
        lambda_c:            penalty weight for expert violation
        n_steps:             gradient steps
        lr / weight_decay:   optimizer hyperparams
        batch_size:          rows sampled from each embedding matrix per step

    Returns:
        dict with final-step policy_constraint_score, expert_constraint_score,
        constraint_loss (all floats)
    """
    optimizer = AdamW(ctheta.head.parameters(), lr=lr, weight_decay=weight_decay)
    ctheta.train()

    n_policy = policy_embeddings.size(0)
    n_expert = expert_embeddings.size(0)
    p_scores_last = e_scores_last = loss_last = 0.0

    for _ in range(n_steps):
        # Sample random row indices (with replacement)
        p_idx = torch.randint(0, n_policy, (min(batch_size, n_policy),))
        e_idx = torch.randint(0, n_expert, (min(batch_size, n_expert),))

        p_scores = ctheta.head(policy_embeddings[p_idx]).squeeze(-1)
        e_scores = ctheta.head(expert_embeddings[e_idx]).squeeze(-1)

        # Push policy scores HIGH (unsafe); keep expert scores BELOW β (safe)
        loss = -p_scores.mean() + lambda_c * F.relu(e_scores.mean() - beta)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        p_scores_last = p_scores.mean().item()
        e_scores_last = e_scores.mean().item()
        loss_last     = loss.item()

    ctheta.eval()
    return {
        "policy_constraint_score": p_scores_last,
        "expert_constraint_score": e_scores_last,
        "constraint_loss":         loss_last,
    }


# ── Outer training loop ───────────────────────────────────────────────────────

class ICRLTrainer:
    """
    Outer loop that alternates Phase 1 (rollout collection) and Phase 2
    (constraint_update).

    Offline mode  — pass pre-collected policy_trajs (no rollout_fn).
                    Use this for initial training with existing unsafe demos.
    Online mode   — pass rollout_fn; fresh trajectories are collected each
                    iteration and added to the policy pool.

    The policy pool is bounded by policy_buffer_size to prevent memory growth
    in long online runs (oldest trajectories are dropped).
    """

    def __init__(
        self,
        ctheta: TrajectoryEncoder,
        expert_trajs: list[Trajectory],
        policy_trajs: Optional[list[Trajectory]] = None,
        rollout_fn: Optional[Callable[[int], list[Trajectory]]] = None,
        *,
        beta: float = 0.8,
        lambda_c: float = 1.0,
        n_constraint_steps: int = 50,
        batch_size: int = 8,
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        rollouts_per_iter: int = 10,
        policy_buffer_size: int = 500,
        eval_every: int = 10,
        log_dir: str = "logs/icrl",
        run_name: str = "icrl",
    ):
        if policy_trajs is None and rollout_fn is None:
            raise ValueError("Provide either policy_trajs (offline) or rollout_fn (online).")

        self.ctheta             = ctheta
        self.expert_trajs       = expert_trajs
        self.policy_pool        = list(policy_trajs or [])
        self.rollout_fn         = rollout_fn
        self.beta               = beta
        self.lambda_c           = lambda_c
        self.n_constraint_steps = n_constraint_steps
        self.batch_size         = batch_size
        self.lr                 = lr
        self.weight_decay       = weight_decay
        self.rollouts_per_iter  = rollouts_per_iter
        self.policy_buffer_size = policy_buffer_size
        self.eval_every         = eval_every
        self.evaluator          = ConstraintEvaluator(ctheta)
        self.metrics_logger     = MetricsLogger(log_dir, run_name)

        logger.info(
            f"ICRLTrainer ready  "
            f"expert={len(expert_trajs)}  "
            f"policy_pool={len(self.policy_pool)}  "
            f"mode={'offline' if rollout_fn is None else 'online'}  "
            f"β={beta}  λ_c={lambda_c}"
        )

    def train(self, n_iterations: int) -> TrajectoryEncoder:
        # ── Pre-compute expert embeddings once (backbone is frozen) ───────────
        logger.info(f"Pre-computing expert embeddings ({len(self.expert_trajs)} demos)...")
        expert_embs = self.ctheta.embed_texts(
            [t.to_text() for t in self.expert_trajs]
        )
        logger.info(f"Expert embeddings ready: {tuple(expert_embs.shape)}")

        # ── Pre-compute offline policy embeddings (if provided) ───────────────
        policy_embs: torch.Tensor | None = None
        if self.policy_pool:
            logger.info(f"Pre-computing policy embeddings ({len(self.policy_pool)} demos)...")
            policy_embs = self.ctheta.embed_texts(
                [t.to_text() for t in self.policy_pool]
            )
            logger.info(f"Policy embeddings ready: {tuple(policy_embs.shape)}")

        for iteration in range(1, n_iterations + 1):
            # ── Phase 1: collect fresh rollouts (online mode only) ────────────
            if self.rollout_fn is not None:
                new_trajs = self.rollout_fn(self.rollouts_per_iter)
                new_embs  = self.ctheta.embed_texts([t.to_text() for t in new_trajs])
                self.policy_pool.extend(new_trajs)

                policy_embs = new_embs if policy_embs is None else \
                              torch.cat([policy_embs, new_embs], dim=0)

                # Cap buffer — drop oldest embeddings and trajectories together
                if len(self.policy_pool) > self.policy_buffer_size:
                    self.policy_pool = self.policy_pool[-self.policy_buffer_size:]
                    policy_embs      = policy_embs[-self.policy_buffer_size:]

            if policy_embs is None or policy_embs.size(0) == 0:
                logger.warning(f"[{iteration}] No policy trajectories — skipping.")
                continue

            # ── Phase 2: constraint update (head only, embeddings cached) ─────
            metrics = constraint_update(
                self.ctheta,
                policy_embeddings=policy_embs,
                expert_embeddings=expert_embs,
                beta=self.beta,
                lambda_c=self.lambda_c,
                n_steps=self.n_constraint_steps,
                lr=self.lr,
                weight_decay=self.weight_decay,
                batch_size=self.batch_size,
            )

            # ── Logging ───────────────────────────────────────────────────────
            rewards = [t.reward for t in self.policy_pool if t.reward is not None]
            metrics["task_reward"]       = sum(rewards) / len(rewards) if rewards else 0.0
            metrics["policy_pool_size"]  = len(self.policy_pool)

            logger.info(
                f"[{iteration:4d}/{n_iterations}]  "
                f"policy_score={metrics['policy_constraint_score']:.4f}  "
                f"expert_score={metrics['expert_constraint_score']:.4f}  "
                f"reward={metrics['task_reward']:.3f}  "
                f"loss={metrics['constraint_loss']:+.4f}"
            )
            self.metrics_logger.log(metrics, step=iteration)

            # ── Periodic evaluation ───────────────────────────────────────────
            if iteration % self.eval_every == 0:
                eval_m = self.evaluator.evaluate(self.expert_trajs, self.policy_pool)
                logger.info(
                    f"[{iteration:4d}] eval  "
                    f"AUROC={eval_m['auroc']:.3f}  "
                    f"sep={eval_m['separation']:+.3f}  "
                    f"unsafe={eval_m['unsafe_mean_score']:.3f}  "
                    f"safe={eval_m['safe_mean_score']:.3f}"
                )
                self.metrics_logger.log({f"eval_{k}": v for k, v in eval_m.items()},
                                         step=iteration)

        return self.ctheta
