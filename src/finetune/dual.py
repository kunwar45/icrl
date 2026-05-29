"""
Dual variable λ for the Lagrangian constrained RLHF objective.

Update rule (gradient ascent on the dual):
    λ ← max(0, λ + αλ · (mean_batch[1 - C_θ(τ)] − ε))
"""
import torch
from omegaconf import DictConfig


class DualVariable:
    def __init__(self, cfg: DictConfig):
        self.lambda_val = cfg.finetune.constraint.lambda_init
        self.lr = cfg.finetune.constraint.lambda_lr
        self.epsilon = cfg.finetune.constraint.epsilon
        self.lambda_max = cfg.finetune.constraint.lambda_max
        self._history: list[float] = []

    def update(self, constraint_scores: torch.Tensor) -> float:
        violation_rate = (1.0 - constraint_scores.mean().item())
        gradient = violation_rate - self.epsilon
        self.lambda_val = max(0.0, min(
            self.lambda_max,
            self.lambda_val + self.lr * gradient
        ))
        self._history.append(self.lambda_val)
        return self.lambda_val

    @property
    def value(self) -> float:
        return self.lambda_val

    def state_dict(self) -> dict:
        return {"lambda": self.lambda_val, "history": self._history}

    def load_state_dict(self, d: dict):
        self.lambda_val = d["lambda"]
        self._history = d["history"]
