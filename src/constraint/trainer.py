"""
Adversarial ICRL training loop.

Round structure:
  1. Train adversarial policy πadv for N PPO steps (maximize task reward)
  2. Score πadv rollouts with C_θ; compute constraint loss
  3. Update C_θ to push πadv scores below 0.5, safe demo scores above 0.5
  4. Log: reward gap, AUROC, constraint loss
"""
import torch
import torch.nn as nn
from torch.optim import AdamW
from omegaconf import DictConfig
from src.constraint.encoder import TrajectoryEncoder
from src.constraint.evaluator import ConstraintEvaluator
from src.data.trajectory import Trajectory
from src.utils.logging import get_logger

logger = get_logger(__name__)


class ICRLTrainer:
    def __init__(
        self,
        cfg: DictConfig,
        constraint_model: TrajectoryEncoder,
        safe_trajectories: list[Trajectory],
    ):
        self.cfg = cfg
        self.model = constraint_model
        self.safe_trajs = safe_trajectories
        self.optimizer = AdamW(
            self.model.head.parameters(),
            lr=cfg.constraint.training.learning_rate,
            weight_decay=cfg.constraint.training.weight_decay,
        )
        self.loss_fn = nn.BCELoss()
        self.evaluator = ConstraintEvaluator(constraint_model)

    def train(self) -> TrajectoryEncoder:
        for round_idx in range(self.cfg.constraint.training.n_adversarial_rounds):
            logger.info(f"=== ICRL Round {round_idx + 1} ===")

            adv_trajectories = self._get_adversarial_trajectories(round_idx)
            loss = self._update_constraint(adv_trajectories)

            if (round_idx + 1) % self.cfg.constraint.evaluation.eval_every_n_rounds == 0:
                if adv_trajectories:
                    metrics = self.evaluator.evaluate(self.safe_trajs, adv_trajectories)
                    logger.info(f"Round {round_idx + 1}: {metrics}")
                    # TODO: log to W&B

        return self.model

    def _update_constraint(self, adv_trajectories: list[Trajectory]) -> float:
        if not adv_trajectories:
            return 0.0

        self.model.train()

        safe_texts = [t.to_text() for t in self.safe_trajs]
        safe_labels = torch.ones(len(safe_texts))

        adv_texts = [t.to_text() for t in adv_trajectories]
        adv_labels = torch.zeros(len(adv_texts))

        all_texts = safe_texts + adv_texts
        all_labels = torch.cat([safe_labels, adv_labels]).to(
            next(self.model.parameters()).device
        )

        self.optimizer.zero_grad()
        scores = self.model(all_texts)
        loss = self.loss_fn(scores, all_labels)
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def _get_adversarial_trajectories(self, round_idx: int) -> list[Trajectory]:
        # TODO (Kunwar): implement adversarial PPO rollout here.
        return []
