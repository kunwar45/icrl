# ABOUTME: Tests for GRAFT's RETAIN/FORGET MLP adapters, per-reply gradient routing, the widened-checkpoint fold and the split-moment AdamW optimizer, on a tiny randomly initialised Qwen2 model
# ABOUTME: Run: pytest tests/test_gradient_routing_adapters.py -q
import copy

import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from src.lagrangian_finetuning.gradient_routing_adapters import (
    adapter_state_dict,
    adapters_of,
    add_dual_mlp_adapters,
    decoder_layers,
    forget_scaled,
    layer_adapter,
    load_adapter_state_dict,
    parameter_pairs,
    routing,
    routing_weights,
    widened_mlp_weights,
)
from src.lagrangian_finetuning.split_moment_adamw import SplitMomentAdamW

WIDTH = 4


def tiny_config(extra_width: int = 0) -> Qwen2Config:
    return Qwen2Config(
        vocab_size=50,
        hidden_size=16,
        intermediate_size=32 + extra_width,
        num_hidden_layers=3,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=32,
    )


@pytest.fixture()
def base():
    torch.manual_seed(0)
    return Qwen2ForCausalLM(tiny_config()).double().eval()


def with_trained_adapters(base):
    """A copy of `base` with adapters whose output weights are non-zero, as after training."""
    model = copy.deepcopy(base)
    add_dual_mlp_adapters(model, WIDTH)
    torch.manual_seed(1)
    for a in adapters_of(model):
        for branch in (a.retain, a.forget):
            torch.nn.init.normal_(branch.down, std=0.3)
    return model


def ids(batch=3, length=6):
    torch.manual_seed(2)
    return torch.randint(0, 50, (batch, length))


def per_reply_loss(model, x):
    """A loss that is a sum of per-reply terms, like the GRPO loss once advantages are fixed."""
    logits = model(input_ids=x).logits
    coefficients = torch.tensor([1.0, -2.0, 0.5], dtype=logits.dtype)[: x.shape[0]]
    return (logits.pow(2).mean(dim=(1, 2)) * coefficients).sum()


def test_new_adapters_do_not_change_the_model(base):
    x = ids()
    before = base(input_ids=x).logits
    model = copy.deepcopy(base)
    add_dual_mlp_adapters(model, WIDTH)
    assert torch.allclose(model(input_ids=x).logits, before, atol=1e-12)
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert trainable and all(".retain." in n or ".forget." in n for n in trainable)


def test_routing_weights_follow_algorithm_1():
    flagged = torch.tensor([0, 1, 1, 0])
    is_anchor = torch.tensor([0, 0, 1, 1])
    retain, forget, forget_on = routing_weights(flagged, is_anchor)
    # target clean, target flagged, anchor (a flag on an anchor reply does not route), anchor
    assert retain.tolist() == [1.0, 0.0, 1.0, 1.0]
    assert forget.tolist() == [1.0, 2.0, 1.0, 1.0]
    assert forget_on.tolist() == [1.0, 1.0, 0.0, 0.0]


def test_routing_leaves_the_forward_value_unchanged(base):
    model = with_trained_adapters(base)
    x = ids()
    plain = model(input_ids=x).logits
    ones = torch.ones(3)
    with routing(
        model, torch.tensor([1.0, 0.0, 1.0]), torch.tensor([1.0, 2.0, 1.0]), ones
    ):
        routed = model(input_ids=x).logits
    assert torch.allclose(plain, routed, atol=1e-10)


def test_routed_and_unrouted_gradients_match_per_reply_gradients(base):
    model = with_trained_adapters(base)
    x = ids()
    pairs = parameter_pairs(model)
    coefficients = [1.0, -2.0, 0.5]

    def single(b, forget_scale):
        """Gradient of reply b's own loss term, computed alone."""
        model.zero_grad()
        with forget_scaled(model, forget_scale):
            logits = model(input_ids=x[b : b + 1]).logits
        (logits.pow(2).mean() * coefficients[b]).backward()
        return {
            which: [
                w.grad.clone() if w.grad is not None else torch.zeros_like(w)
                for w, _ in ps
            ]
            for which, ps in pairs.items()
        }

    # Reply 0: target, not flagged. Reply 1: target, flagged. Reply 2: anchor (FORGET off).
    g0, g1, g2 = single(0, 1.0), single(1, 1.0), single(2, 0.0)

    model.zero_grad()
    flagged, is_anchor = torch.tensor([0, 1, 0]), torch.tensor([0, 0, 1])
    with routing(model, *routing_weights(flagged, is_anchor)):
        per_reply_loss(model, x).backward()

    for i, (w, z) in enumerate(pairs["retain"]):
        unrouted = g0["retain"][i] + g1["retain"][i] + g2["retain"][i]
        routed = (
            g0["retain"][i] + g2["retain"][i]
        )  # the flagged reply contributes nothing
        assert torch.allclose(w.grad, unrouted, atol=1e-9)
        assert torch.allclose(w.grad + z.grad, routed, atol=1e-9)
    for i, (w, z) in enumerate(pairs["forget"]):
        unrouted = (
            g0["forget"][i] + g1["forget"][i]
        )  # the anchor reply never touches FORGET
        routed = g0["forget"][i] + 2.0 * g1["forget"][i]
        assert torch.allclose(w.grad, unrouted, atol=1e-9)
        assert torch.allclose(w.grad + z.grad, routed, atol=1e-9)
    assert any(g.abs().max() > 1e-6 for g in g1["retain"]), "test would pass trivially"


def test_a_fully_flagged_batch_gives_retain_exactly_zero_routed_gradient(base):
    model = with_trained_adapters(base)
    x = ids()
    with routing(model, *routing_weights(torch.ones(3), torch.zeros(3))):
        per_reply_loss(model, x).backward()
    for w, z in parameter_pairs(model)["retain"]:
        assert torch.allclose(w.grad + z.grad, torch.zeros_like(w), atol=1e-10)


def widened_copy(base, model, forget):
    """A plain Qwen2 model, 2*WIDTH wider, holding base + adapters via widened_mlp_weights."""
    wide = Qwen2ForCausalLM(tiny_config(extra_width=2 * WIDTH)).double().eval()
    state = adapter_state_dict(model)
    source = base.state_dict()
    target = wide.state_dict()
    for name in target:
        if ".mlp." not in name:
            target[name] = source[name].clone()
    for i, layer in enumerate(decoder_layers(base)):
        gate, up, down = widened_mlp_weights(
            layer.mlp.gate_proj.weight.data,
            layer.mlp.up_proj.weight.data,
            layer.mlp.down_proj.weight.data,
            layer_adapter(state, i),
            WIDTH,
            forget,
        )
        target[f"model.layers.{i}.mlp.gate_proj.weight"] = gate
        target[f"model.layers.{i}.mlp.up_proj.weight"] = up
        target[f"model.layers.{i}.mlp.down_proj.weight"] = down
    wide.load_state_dict(target)
    return wide


def test_widened_checkpoint_equals_the_adapter_model(base):
    model = with_trained_adapters(base)
    x = ids()
    both = model(input_ids=x).logits
    with forget_scaled(model, 0.0):
        retain_only = model(input_ids=x).logits
    assert not torch.allclose(both, retain_only, atol=1e-6)
    assert torch.allclose(
        widened_copy(base, model, True)(input_ids=x).logits, both, atol=1e-9
    )
    assert torch.allclose(
        widened_copy(base, model, False)(input_ids=x).logits, retain_only, atol=1e-9
    )


def test_widened_weights_without_adapters_equal_the_base_model(base):
    mlp = decoder_layers(base)[0].mlp
    gate, up, down = widened_mlp_weights(
        mlp.gate_proj.weight.data,
        mlp.up_proj.weight.data,
        mlp.down_proj.weight.data,
        None,
        WIDTH,
        True,
    )
    assert gate.shape == (32 + 2 * WIDTH, 16) and down.shape == (16, 32 + 2 * WIDTH)
    assert down[:, 32:].abs().max() == 0


def test_adapter_state_round_trip(base):
    model = with_trained_adapters(base)
    state = adapter_state_dict(model)
    assert len(state) == 6 * 3 and not any("shadow" in k for k in state)
    other = copy.deepcopy(base)
    add_dual_mlp_adapters(other, WIDTH)
    load_adapter_state_dict(other, state)
    x = ids()
    assert torch.allclose(
        other(input_ids=x).logits, model(input_ids=x).logits, atol=1e-12
    )
    with pytest.raises(ValueError):
        load_adapter_state_dict(other, dict(list(state.items())[:-1]))


def pair(value=1.0):
    w = torch.nn.Parameter(torch.full((2,), value))
    z = torch.nn.Parameter(torch.zeros(2))
    return w, z


def test_split_moment_step_uses_routed_gradient_for_direction_and_unrouted_for_scale():
    w, z = pair()
    opt = SplitMomentAdamW([{"pairs": [(w, z)]}], lr=0.1, betas=(0.9, 0.99), eps=0.0)
    w.grad = torch.tensor([1.0, 1.0])  # unrouted
    z.grad = torch.tensor([1.0, -1.0])  # routed = [2, 0]: doubled, and masked
    opt.step()
    # First step: m_hat = routed, v_hat = unrouted^2, so the update is routed / |unrouted|.
    assert torch.allclose(w.data, torch.tensor([1.0 - 0.1 * 2.0, 1.0]))
    # Plain Adam would have normalised the doubled gradient back to a step of 0.1.


def test_split_moment_unrouted_scale_weight_decay_and_clipping():
    w, z = pair()
    opt = SplitMomentAdamW(
        [{"pairs": [(w, z)], "unrouted_scale": 2.0}], lr=0.1, eps=0.0, weight_decay=0.5
    )
    w.grad = torch.tensor([1.0, 1.0])
    z.grad = torch.zeros(2)
    opt.step()
    # decay: 1 * (1 - 0.1 * 0.5) = 0.95; update: 1 / (2 * 1) = 0.5, times lr
    assert torch.allclose(w.data, torch.full((2,), 0.95 - 0.05))

    w, z = pair()
    opt = SplitMomentAdamW([{"pairs": [(w, z)]}], lr=0.1, eps=0.0, max_grad_norm=1.0)
    w.grad = torch.tensor([3.0, 4.0])
    z.grad = torch.zeros(2)
    assert opt.routed_grad_norm() == pytest.approx(5.0)
    opt.step()
    # Both gradients are scaled by the same factor, so the first step is still lr * sign.
    assert torch.allclose(w.data, torch.full((2,), 0.9), atol=1e-5)
    opt.zero_grad()
    assert w.grad is None and z.grad is None
