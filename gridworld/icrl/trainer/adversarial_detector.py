"""
Adversarial detector: the core insight of paper #3.

Safe demonstrations are near-optimal under the true (unknown) constraints.
Therefore any trajectory achieving higher reward than those demos must have
done so by skipping required safety steps (e.g., confirmation, verification).
This class converts that insight into an automated labelling rule.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from icrl.core.types import DemoDataset, Trajectory


@dataclass
class AdversarialDetectorConfig:
    mode: str = "max"           # "max" | "mean_plus_std" | "percentile"
    percentile: float = 95.0    # used when mode="percentile"
    std_multiplier: float = 1.0  # used when mode="mean_plus_std"
    min_reward_gap: float = 0.01  # must beat threshold by at least this much


class AdversarialDetector:
    def __init__(self, config: AdversarialDetectorConfig):
        self.config = config
        self._threshold: Optional[float] = None
        self._n_safe_demos: int = 0

    def fit(self, demo_dataset: DemoDataset) -> float:
        """Compute and cache the reward threshold from safe demonstrations."""
        rewards = [t.total_reward for t in demo_dataset.safe]
        self._n_safe_demos = len(rewards)

        if self.config.mode == "max":
            self._threshold = float(max(rewards))
        elif self.config.mode == "mean_plus_std":
            self._threshold = float(
                np.mean(rewards) + self.config.std_multiplier * np.std(rewards)
            )
        elif self.config.mode == "percentile":
            self._threshold = float(np.percentile(rewards, self.config.percentile))
        else:
            raise ValueError(f"Unknown mode: {self.config.mode!r}")

        return self._threshold

    @property
    def threshold(self) -> float:
        if self._threshold is None:
            raise RuntimeError("Call fit() before using AdversarialDetector.")
        return self._threshold

    def is_unsafe(self, trajectory: Trajectory) -> bool:
        """True if this trajectory beat the safe-demo threshold — flagged unsafe."""
        return trajectory.total_reward > self.threshold + self.config.min_reward_gap

    def filter_unsafe(
        self, trajectories: list[Trajectory]
    ) -> tuple[list[Trajectory], list[Trajectory]]:
        """Split into (safe_enough, flagged_as_unsafe)."""
        safe, unsafe = [], []
        for t in trajectories:
            (unsafe if self.is_unsafe(t) else safe).append(t)
        return safe, unsafe

    def stats(self) -> dict:
        return {
            "threshold": self.threshold,
            "mode": self.config.mode,
            "n_safe_demos": self._n_safe_demos,
        }
