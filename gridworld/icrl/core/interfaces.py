"""Abstract base classes defining contracts between ICRL components.

Keeping these narrow and typed makes every component swappable:
swap the env, the policy, the constraint, or the embedder independently.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

import torch

from icrl.core.types import Trajectory


class BaseEnv(ABC):
    """Gymnasium-compatible environment with text representations for embedding."""

    @abstractmethod
    def reset(self, seed: Optional[int] = None) -> tuple[Any, dict]:
        ...

    @abstractmethod
    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict]:
        ...

    @abstractmethod
    def obs_repr(self, obs: Any) -> str:
        """Text representation of an observation — used for trajectory embedding."""
        ...

    @abstractmethod
    def action_repr(self, action: Any) -> str:
        """Text representation of an action — used for trajectory embedding."""
        ...

    @property
    def obs_dim(self) -> int:
        """Dimensionality of the numeric observation vector (policy network input).

        LLM-based policies that don't use a numeric obs vector return -1.
        """
        return -1

    @property
    def n_actions(self) -> int:
        """Number of discrete actions. LLM policies return -1 (open text action space)."""
        return -1


class BaseEmbedder(ABC):
    """Maps a full trajectory (sequence of obs-action pairs) to a fixed vector.

    Operates on text representations so it generalises across action/obs types:
    grid positions, DOM trees, API calls, tool responses — all become text.
    """

    @abstractmethod
    def embed_trajectory(self, trajectory: Trajectory, env: BaseEnv) -> torch.Tensor:
        """Returns shape [embed_dim]."""
        ...

    @abstractmethod
    def embed_batch(
        self, trajectories: list[Trajectory], env: BaseEnv
    ) -> torch.Tensor:
        """Returns shape [B, embed_dim]."""
        ...

    @property
    @abstractmethod
    def embed_dim(self) -> int:
        ...


class BaseConstraint(ABC):
    """C_θ: trajectory → feasibility score in [0, 1].  High = feasible (safe)."""

    @abstractmethod
    def feasibility(self, trajectory: Trajectory) -> torch.Tensor:
        """Scalar tensor in [0, 1]."""
        ...

    @abstractmethod
    def cost(self, trajectory: Trajectory) -> torch.Tensor:
        """-log(feasibility).  Used as episode cost signal for PPO-Lag."""
        ...

    @abstractmethod
    def update(
        self,
        safe_trajectories: list[Trajectory],
        unsafe_trajectories: list[Trajectory],
    ) -> dict[str, float]:
        """One contrastive training step.  Returns loss metrics."""
        ...

    @abstractmethod
    def state_dict(self) -> dict:
        ...

    @abstractmethod
    def load_state_dict(self, state: dict) -> None:
        ...


class BasePolicy(ABC):
    """Agent policy π_φ: obs → action."""

    @abstractmethod
    def act(self, obs: Any, deterministic: bool = False) -> Any:
        ...

    @abstractmethod
    def update(self, trajectories: list[Trajectory]) -> dict[str, float]:
        """Policy gradient update.  Returns loss metrics."""
        ...

    @abstractmethod
    def set_constraint(self, constraint: Optional[BaseConstraint]) -> None:
        """Inject constraint so the policy can use cost signals during training."""
        ...

    @abstractmethod
    def state_dict(self) -> dict:
        ...

    @abstractmethod
    def load_state_dict(self, state: dict) -> None:
        ...
