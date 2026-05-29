"""Probing dataset construction from model checkpoints."""
from src.data.trajectory import Trajectory


def build_probing_dataset(
    trajectories: list[Trajectory],
    constraint_scores: list[float],
) -> tuple[list[Trajectory], list[float]]:
    """
    Pair trajectories with their C_θ scores for probe training.
    Returns (trajectories, scores) ready for LayerWiseProbes.fit().
    """
    assert len(trajectories) == len(constraint_scores)
    return trajectories, constraint_scores
