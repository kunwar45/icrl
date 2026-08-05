"""Unit tests for constraint loss functions."""
import torch
import pytest
from icrl.constraints.losses import contrastive_loss, margin_loss, info_nce_loss


def test_contrastive_loss_perfect_separation():
    """When safe≈1 and unsafe≈0, loss should be near zero."""
    safe = torch.tensor([0.99, 0.98, 0.97])
    unsafe = torch.tensor([0.01, 0.02, 0.03])
    loss = contrastive_loss(safe, unsafe)
    assert float(loss) < 0.1


def test_contrastive_loss_no_separation():
    """When both are 0.5, loss should be high."""
    safe = torch.tensor([0.5, 0.5])
    unsafe = torch.tensor([0.5, 0.5])
    loss_confused = contrastive_loss(safe, unsafe)
    loss_good = contrastive_loss(torch.tensor([0.99, 0.99]), torch.tensor([0.01, 0.01]))
    assert float(loss_confused) > float(loss_good)


def test_contrastive_loss_gradients():
    """Loss should produce non-zero gradients for parameters."""
    import torch.nn as nn
    net = nn.Linear(4, 1)
    x_safe = torch.randn(3, 4)
    x_unsafe = torch.randn(3, 4)
    safe_scores = torch.sigmoid(net(x_safe)).squeeze()
    unsafe_scores = torch.sigmoid(net(x_unsafe)).squeeze()
    loss = contrastive_loss(safe_scores, unsafe_scores)
    loss.backward()
    assert net.weight.grad is not None
    assert net.weight.grad.abs().sum() > 0


def test_margin_loss_satisfied():
    """When safe > unsafe + margin, loss is zero."""
    safe = torch.tensor([0.9])
    unsafe = torch.tensor([0.5])
    loss = margin_loss(safe, unsafe, margin=0.3)
    assert float(loss) == pytest.approx(0.0, abs=1e-6)


def test_margin_loss_violated():
    """When safe < unsafe + margin, loss is positive."""
    safe = torch.tensor([0.5])
    unsafe = torch.tensor([0.5])
    loss = margin_loss(safe, unsafe, margin=0.3)
    assert float(loss) > 0.0


def test_info_nce_loss_shape():
    """InfoNCE should be a scalar."""
    safe_emb = torch.randn(4, 8)
    unsafe_emb = torch.randn(6, 8)
    loss = info_nce_loss(safe_emb, unsafe_emb)
    assert loss.shape == ()
    assert float(loss) >= 0.0
