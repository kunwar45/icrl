"""
Convention: HIGH C_θ = high cost = unsafe.
λ rises when mean cost E[C_θ] > ε (policy is unsafe),
λ falls when mean cost E[C_θ] < ε (policy is within budget).
"""
from src.finetune.dual import DualVariable
from omegaconf import OmegaConf
import torch


def make_cfg(epsilon=0.1, lambda_init=0.5, lambda_lr=0.01, lambda_max=10.0):
    return OmegaConf.create({
        "finetune": {
            "constraint": {
                "epsilon": epsilon,
                "lambda_init": lambda_init,
                "lambda_lr": lambda_lr,
                "lambda_max": lambda_max,
            }
        }
    })


def test_dual_increases_on_violation():
    dual = DualVariable(make_cfg())
    initial = dual.value
    # All scores = 1.0 (HIGH = all unsafe) → mean cost 1.0 >> ε=0.1 → λ should rise
    scores = torch.ones(8)
    dual.update(scores)
    assert dual.value > initial


def test_dual_decreases_when_safe():
    dual = DualVariable(make_cfg(lambda_init=2.0))
    initial = dual.value
    # All scores = 0.0 (LOW = all safe) → mean cost 0.0 << ε=0.1 → λ should fall
    scores = torch.zeros(8)
    dual.update(scores)
    assert dual.value < initial


def test_dual_clamped_at_zero():
    dual = DualVariable(make_cfg(lambda_init=0.001))
    # Safe scores → gradient pushes below 0, should clamp at 0
    scores = torch.zeros(8)
    dual.update(scores)
    assert dual.value >= 0.0


def test_dual_clamped_at_max():
    dual = DualVariable(make_cfg(lambda_init=9.99, lambda_lr=1.0))
    # Unsafe scores → gradient pushes above λ_max, should clamp at 10.0
    scores = torch.ones(8)
    dual.update(scores)
    assert dual.value <= 10.0


def test_state_dict_roundtrip():
    dual = DualVariable(make_cfg())
    dual.update(torch.ones(4))
    state = dual.state_dict()
    dual2 = DualVariable(make_cfg())
    dual2.load_state_dict(state)
    assert dual2.value == dual.value
