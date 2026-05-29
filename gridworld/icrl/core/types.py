"""Core data structures shared across all ICRL components."""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


@dataclass
class Transition:
    """Single (obs, action, reward, next_obs, done) environment step."""

    obs: Any
    action: Any
    reward: float
    next_obs: Any
    done: bool
    cost: float = 0.0
    info: dict = field(default_factory=dict)


@dataclass
class Trajectory:
    """Complete episode as an ordered sequence of transitions."""

    transitions: list[Transition]
    total_reward: float
    total_cost: float
    episode_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    metadata: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.transitions)

    @property
    def is_safe(self) -> Optional[bool]:
        """None = unknown (learner rollout); True/False = labeled demo."""
        return self.metadata.get("is_safe", None)

    @classmethod
    def from_transitions(
        cls,
        transitions: list[Transition],
        metadata: Optional[dict] = None,
    ) -> "Trajectory":
        total_r = sum(t.reward for t in transitions)
        total_c = sum(t.cost for t in transitions)
        return cls(
            transitions=transitions,
            total_reward=total_r,
            total_cost=total_c,
            metadata=metadata or {},
        )


@dataclass
class DemoDataset:
    """Container for demonstration trajectories used in ICRL training."""

    safe: list[Trajectory]
    unsafe: list[Trajectory] = field(default_factory=list)

    @property
    def safe_reward_threshold(self) -> float:
        """Max total reward across safe demos — the adversarial pivot point.

        Any learner trajectory exceeding this must have skipped required
        safety steps (e.g., confirmation) to achieve the higher reward.
        """
        if not self.safe:
            raise ValueError("No safe demonstrations loaded.")
        return max(t.total_reward for t in self.safe)

    @property
    def safe_reward_mean(self) -> float:
        return float(np.mean([t.total_reward for t in self.safe]))

    @property
    def safe_reward_std(self) -> float:
        return float(np.std([t.total_reward for t in self.safe]))

    def __repr__(self) -> str:
        return (
            f"DemoDataset(safe={len(self.safe)}, unsafe={len(self.unsafe)}, "
            f"threshold={self.safe_reward_threshold:.2f})"
        )
