"""Encoder tests that run without loading a real LLM (uses mock backbone)."""
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch
from omegaconf import OmegaConf
from src.data.trajectory import Trajectory, Step


def make_trajectory():
    return Trajectory(
        trajectory_id="enc_test_001",
        task_type="delete_record",
        task_instance_id="i001",
        steps=[Step(step_idx=0, action="DELETE /x", observation="200 OK")],
        is_safe=True,
        source="test",
    )


def test_trajectory_to_text_nonempty():
    traj = make_trajectory()
    text = traj.to_text()
    assert len(text) > 0
    assert "DELETE /x" in text
    assert "200 OK" in text


def test_trajectory_to_text_format():
    traj = make_trajectory()
    text = traj.to_text()
    assert text.startswith("[ACTION]")
    assert "[OBS]" in text
