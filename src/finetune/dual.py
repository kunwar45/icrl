"""
Dual variable λ for the Lagrangian constrained RLHF objective.

Convention: HIGH C_θ(τ) = high cost = unsafe (matches constraint_update loss).

Update rule (gradient ascent on the dual):
    λ ← clamp(λ + αλ · (E[C_θ(τ)] − ε), 0, λ_max)

λ rises when mean cost E[C_θ] exceeds the budget ε (policy is being unsafe),
and falls when mean cost drops below ε (policy is safely within budget).
"""
import torch
from omegaconf import DictConfig


class DualVariable:
    def __init__(self, cfg: DictConfig):
        self.lambda_val = cfg.finetune.constraint.lambda_init
        self.lr         = cfg.finetune.constraint.lambda_lr
        self.epsilon    = cfg.finetune.constraint.epsilon
        self.lambda_max = cfg.finetune.constraint.lambda_max
        self._history: list[float] = []

    def update(self, constraint_scores: torch.Tensor) -> float:
        """
        Args:
            constraint_scores: C_θ(τ) for each trajectory in the batch.
                               HIGH = unsafe/costly, LOW = safe.
        """
        # Mean cost; compare against budget ε
        mean_cost = constraint_scores.mean().item()
        gradient  = mean_cost - self.epsilon
        self.lambda_val = max(0.0, min(
            self.lambda_max,
            self.lambda_val + self.lr * gradient,
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
        self._history   = d["history"]
