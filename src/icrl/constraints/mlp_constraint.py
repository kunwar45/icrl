"""
MLP-based constraint function.

C_θ(τ) = sigmoid(MLP(concat(embed(τ), reward_norm, length_norm)))

Feasibility ∈ [0,1]: high = safe, low = unsafe.
Cost = −log(feasibility): fed to PPO-Lag as episode-level cost signal.

The reward_norm and length_norm features are concatenated to the embedding
because the adversarial principle depends on reward level: the MLP should
learn that "high reward + unusual action sequence = unsafe".
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from icrl.constraints.losses import contrastive_loss, margin_loss
from icrl.core.interfaces import BaseConstraint, BaseEmbedder, BaseEnv
from icrl.core.types import Trajectory


@dataclass
class MLPConstraintConfig:
    hidden_dim: int = 64
    lr: float = 1e-3
    loss: str = "contrastive"    # "contrastive" | "margin"
    margin: float = 0.3
    weight_decay: float = 1e-4


class MLPConstraint(BaseConstraint):
    def __init__(
        self,
        embedder: BaseEmbedder,
        env: BaseEnv,
        config: MLPConstraintConfig,
    ):
        self.embedder = embedder
        self.env = env
        self.config = config

        input_dim = embedder.embed_dim + 2  # +2: normalised reward, normalised length
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 2, 1),
            nn.Sigmoid(),
        )
        # Only include embedder params when it exposes trainable weights (e.g. MeanPoolEmbedder).
        # Frozen embedders like SentenceTransformerEmbedder have no learnable params.
        embedder_params = list(embedder.parameters()) if isinstance(embedder, nn.Module) else []
        self.optimizer = optim.Adam(
            list(self.mlp.parameters()) + embedder_params,
            lr=config.lr,
            weight_decay=config.weight_decay,
        )

        # Running normalisation stats — updated each constraint training step
        self._reward_mean = 0.0
        self._reward_std = 1.0
        self._length_mean = 10.0
        self._length_std = 5.0

    # ------------------------------------------------------------------
    # Featurisation
    # ------------------------------------------------------------------

    def _featurize(self, trajectory: Trajectory) -> torch.Tensor:
        emb = self.embedder.embed_trajectory(trajectory, self.env)
        r_norm = torch.tensor(
            [(trajectory.total_reward - self._reward_mean) / (self._reward_std + 1e-8)],
            dtype=torch.float32,
        )
        l_norm = torch.tensor(
            [(len(trajectory) - self._length_mean) / (self._length_std + 1e-8)],
            dtype=torch.float32,
        )
        return torch.cat([emb, r_norm, l_norm], dim=0)

    def _update_normalisation(self, trajectories: list[Trajectory]) -> None:
        rewards = [t.total_reward for t in trajectories]
        lengths = [len(t) for t in trajectories]
        self._reward_mean = float(np.mean(rewards))
        self._reward_std = float(np.std(rewards)) + 1e-8
        self._length_mean = float(np.mean(lengths))
        self._length_std = float(np.std(lengths)) + 1e-8

    # ------------------------------------------------------------------
    # BaseConstraint interface
    # ------------------------------------------------------------------

    def feasibility(self, trajectory: Trajectory) -> torch.Tensor:
        feats = self._featurize(trajectory)
        return self.mlp(feats).squeeze()

    def cost(self, trajectory: Trajectory) -> torch.Tensor:
        return -torch.log(self.feasibility(trajectory) + 1e-8)

    def update(
        self,
        safe_trajectories: list[Trajectory],
        unsafe_trajectories: list[Trajectory],
    ) -> dict[str, float]:
        if not safe_trajectories or not unsafe_trajectories:
            return {}

        self._update_normalisation(safe_trajectories + unsafe_trajectories)

        safe_feats = torch.stack([self._featurize(t) for t in safe_trajectories])
        unsafe_feats = torch.stack([self._featurize(t) for t in unsafe_trajectories])

        safe_scores = self.mlp(safe_feats).squeeze(-1)
        unsafe_scores = self.mlp(unsafe_feats).squeeze(-1)

        if self.config.loss == "contrastive":
            loss = contrastive_loss(safe_scores, unsafe_scores)
        elif self.config.loss == "margin":
            loss = margin_loss(safe_scores, unsafe_scores, self.config.margin)
        else:
            raise ValueError(f"Unknown loss: {self.config.loss!r}")

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            "constraint_loss": float(loss.item()),
            "safe_feasibility_mean": float(safe_scores.mean().item()),
            "unsafe_feasibility_mean": float(unsafe_scores.mean().item()),
            "feasibility_gap": float((safe_scores.mean() - unsafe_scores.mean()).item()),
        }

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        return {
            "mlp": self.mlp.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "embedder": self.embedder.state_dict(),
            "norm": {
                "reward_mean": self._reward_mean,
                "reward_std": self._reward_std,
                "length_mean": self._length_mean,
                "length_std": self._length_std,
            },
        }

    def load_state_dict(self, state: dict) -> None:
        self.mlp.load_state_dict(state["mlp"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.embedder.load_state_dict(state["embedder"])
        n = state["norm"]
        self._reward_mean = n["reward_mean"]
        self._reward_std = n["reward_std"]
        self._length_mean = n["length_mean"]
        self._length_std = n["length_std"]
