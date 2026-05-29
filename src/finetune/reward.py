"""Task reward model wrapper. Plug-in point — Kunwar will provide the implementation."""
from src.data.trajectory import Trajectory


class RewardModel:
    def score(self, trajectory: Trajectory) -> float:
        # TODO (Kunwar): implement task completion reward
        raise NotImplementedError
