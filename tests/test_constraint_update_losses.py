# ABOUTME: Tests the constraint head objectives (icrl / hinge_persample / bce) and the text_mode that travels with a head
# ABOUTME: Run: pytest tests/test_constraint_update_losses.py -q
import json

import pytest
import torch
import torch.nn as nn

from src.icrl_dual_training.constraint_trainer import constraint_update
from src.trajectory_data.trajectory import Step, Trajectory


class _HeadOnly(nn.Module):
    """Stands in for a TrajectoryEncoder: constraint_update touches only .head."""

    def __init__(self, dim=8, hidden=16):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, 1), nn.Sigmoid()
        )


def _separable(n=64, dim=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    expert = torch.randn(n, dim, generator=g) - 1.5
    policy = torch.randn(n, dim, generator=g) + 1.5
    return expert, policy


@pytest.mark.parametrize("loss_kind", ["icrl", "hinge_persample", "bce"])
def test_every_loss_separates_expert_from_policy(loss_kind):
    torch.manual_seed(0)
    expert, policy = _separable()
    m = _HeadOnly()
    out = constraint_update(
        m,
        policy,
        expert,
        beta=0.2,
        lambda_c=5.0,
        n_steps=300,
        lr=1e-2,
        weight_decay=0.0,
        batch_size=16,
        loss_kind=loss_kind,
    )
    with torch.no_grad():
        e = m.head(expert).mean().item()
        p = m.head(policy).mean().item()
    assert p > 0.8 and e < 0.3, f"{loss_kind}: expert {e:.2f} policy {p:.2f}"
    assert set(out) == {
        "policy_constraint_score",
        "expert_constraint_score",
        "constraint_loss",
    }


def test_bce_does_not_leave_experts_saturated_at_one():
    """The icrl form bounds only the MEAN expert score; bce bounds each one."""
    torch.manual_seed(0)
    expert, policy = _separable()
    m = _HeadOnly()
    constraint_update(
        m,
        policy,
        expert,
        beta=0.8,
        lambda_c=1.0,
        n_steps=300,
        lr=1e-2,
        weight_decay=0.0,
        batch_size=16,
        loss_kind="bce",
    )
    with torch.no_grad():
        assert m.head(expert).max().item() < 0.5


def test_unknown_loss_is_refused():
    expert, policy = _separable(n=8)
    with pytest.raises(ValueError, match="loss_kind"):
        constraint_update(
            _HeadOnly(),
            policy,
            expert,
            beta=0.8,
            lambda_c=1.0,
            n_steps=1,
            lr=1e-3,
            weight_decay=0.0,
            batch_size=4,
            loss_kind="hinge",
        )


def test_trajectory_text_modes():
    t = Trajectory(
        "t",
        "odcv",
        "s",
        [Step(0, "bash: ls", ""), Step(1, "bash: sed -i x f", "a\nb")],
        is_safe=False,
        source="x",
    )
    assert (
        t.to_text() == "[ACTION] bash: ls [OBS]  [ACTION] bash: sed -i x f [OBS] a\nb"
    )
    assert t.to_text("actions_only") == "[ACTION] bash: ls [ACTION] bash: sed -i x f"
    with pytest.raises(ValueError):
        t.to_text("observations_only")


def test_head_metadata_carries_text_mode(tmp_path):
    from src.trajectory_embedding.trajectory_encoder import (
        _meta_path,
        load_constraint_head,
        save_constraint_head,
    )

    class _Enc(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn.Linear(4, 4)
            self.head = nn.Sequential(nn.Linear(4, 1), nn.Sigmoid())
            self.hidden_size = 4
            self.text_mode = "actions_only"

    enc = _Enc()
    path = str(tmp_path / "constraint_head.pt")
    save_constraint_head(enc, path, model_name="fake", max_length=32)
    assert json.load(open(_meta_path(path)))["text_mode"] == "actions_only"
    other = _Enc()
    other.text_mode = "full"
    load_constraint_head(other, path, model_name="fake")
    assert other.text_mode == "actions_only"
