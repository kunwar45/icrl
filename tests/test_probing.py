"""Probing tests using toy data and a tiny dummy model."""
import numpy as np
import pytest
from unittest.mock import MagicMock
from src.data.trajectory import Trajectory, Step


def make_trajectory(traj_id: str = "t001", action: str = "GET /x") -> Trajectory:
    return Trajectory(
        trajectory_id=traj_id,
        task_type="delete_record",
        task_instance_id="i001",
        steps=[Step(step_idx=0, action=action, observation="200 OK")],
        is_safe=True,
        source="test",
    )


def test_probing_dataset_construction():
    from src.probing.dataset import build_probing_dataset
    trajs = [make_trajectory(f"t{i}") for i in range(5)]
    scores = [float(i) / 5 for i in range(5)]
    out_trajs, out_scores = build_probing_dataset(trajs, scores)
    assert len(out_trajs) == 5
    assert len(out_scores) == 5


def test_probing_dataset_length_mismatch():
    from src.probing.dataset import build_probing_dataset
    trajs = [make_trajectory()]
    scores = [0.5, 0.6]
    with pytest.raises(AssertionError):
        build_probing_dataset(trajs, scores)
