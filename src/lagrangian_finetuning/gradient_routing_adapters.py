# ABOUTME: GRAFT's RETAIN and FORGET MLP adapters (Ward et al. 2026, arXiv 2610.17062): two extra SwiGLU branches on every layer's MLP of a frozen model, with per-reply gradient weights (routing) and a per-reply switch that turns FORGET off
# ABOUTME: Use: adapters = add_dual_mlp_adapters(model, width=64); with routing(model, retain_weight, forget_weight, forget_on): loss.backward(); write_widened_checkpoint(...) folds the adapters into a plain checkpoint
"""
Our reimplementation of the adapter and routing mechanism of GRAFT. The mechanism
follows the paper (Section 3, Appendix G) and the authors' MIT-licensed code at
https://github.com/jnward/GRAFT (rl/gradient_routing.py); no code is copied.

Architecture (paper, Appendix G). Each Transformer block computes

    MLP(x) + g_R(x) + g_F(x),    g(x) = W_down( SiLU(W_gate x) * W_up x )

with the base model frozen. W_gate and W_up start Kaiming-uniform and W_down starts
at zero, so the adapters change nothing at the start of training.

Routing (paper, Algorithm 1 lines 8-9). Every reply has a gradient weight for each
adapter: RETAIN gets 0 on flagged replies and 1 otherwise; FORGET gets 2 on flagged
replies and 1 otherwise. The weight scales only that reply's contribution to the
adapter's own parameters. The forward value and the gradient that flows on to
earlier layers are unchanged. This is done with a zero-valued term

    (w - 1) * ( g'(stopgrad(x)) - stopgrad(g'(stopgrad(x))) )

where g' uses the same weights plus a "shadow" parameter that is always zero. After
one backward pass, for each adapter weight W with shadow Z:

    W.grad           = the unrouted gradient (every weight set to 1)
    W.grad + Z.grad  = the routed gradient

The split-moment optimizer (split_moment_adamw.py) needs both, and this gives both
from a single backward pass.

Anchor replies are sampled and trained with FORGET off (`forget_on` = 0).

Deployment. Two parallel branches of width n are the same function as widening the
base MLP's hidden layer by 2n, so `widened_mlp_weights` folds the adapters into a
plain checkpoint that vLLM serves natively. With `forget=False` the FORGET branch's
down-projection is written as zero: that is the deployed model.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F

SHADOW_SUFFIX = "_shadow"


class RoutingState:
    """Per-reply routing for the next forward pass, shared by all of a model's adapters."""

    def __init__(self) -> None:
        self.active = False
        self.retain_weight: torch.Tensor | None = None
        self.forget_weight: torch.Tensor | None = None
        self.forget_on: torch.Tensor | None = None
        # Used when no routing is set (plain forward, e.g. generation): scales FORGET's output.
        self.default_forget_scale = 1.0


class SwiGLUBranch(nn.Module):
    """One adapter at one layer: a bias-free SwiGLU of the given width, plus zero shadows."""

    def __init__(
        self, hidden: int, width: int, dtype: torch.dtype, device: torch.device
    ) -> None:
        super().__init__()
        kw = {"dtype": dtype, "device": device}
        self.gate = nn.Parameter(torch.empty(width, hidden, **kw))
        self.up = nn.Parameter(torch.empty(width, hidden, **kw))
        self.down = nn.Parameter(torch.zeros(hidden, width, **kw))
        nn.init.kaiming_uniform_(self.gate, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.up, a=math.sqrt(5))
        # Always zero. They only collect the routing part of the gradient.
        self.gate_shadow = nn.Parameter(torch.zeros(width, hidden, **kw))
        self.up_shadow = nn.Parameter(torch.zeros(width, hidden, **kw))
        self.down_shadow = nn.Parameter(torch.zeros(hidden, width, **kw))

    def _swiglu(self, x: torch.Tensor, gate, up, down) -> torch.Tensor:
        # The adapter weights may be kept in float32 while the frozen model runs in bfloat16.
        gate, up, down = gate.to(x.dtype), up.to(x.dtype), down.to(x.dtype)
        return F.linear(F.silu(F.linear(x, gate)) * F.linear(x, up), down)

    def core(self, x: torch.Tensor) -> torch.Tensor:
        return self._swiglu(x, self.gate, self.up, self.down)

    def shadow_core(self, x_detached: torch.Tensor) -> torch.Tensor:
        return self._swiglu(
            x_detached,
            self.gate.detach() + self.gate_shadow,
            self.up.detach() + self.up_shadow,
            self.down.detach() + self.down_shadow,
        )

    def routed(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """core(x) in value and in input gradient; the parameter gradient is scaled by `weight`."""
        shadow = self.shadow_core(x.detach())
        return self.core(x) + (weight - 1.0) * (shadow - shadow.detach())

    def pairs(self) -> list[tuple[nn.Parameter, nn.Parameter]]:
        return [
            (self.gate, self.gate_shadow),
            (self.up, self.up_shadow),
            (self.down, self.down_shadow),
        ]


class DualMLPAdapter(nn.Module):
    """A frozen base MLP with a RETAIN and a FORGET branch added in parallel."""

    def __init__(
        self,
        base_mlp: nn.Module,
        hidden: int,
        width: int,
        state: RoutingState,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.base = base_mlp
        for p in self.base.parameters():
            p.requires_grad_(False)
        p0 = next(self.base.parameters())
        dtype = dtype or p0.dtype
        self.retain = SwiGLUBranch(hidden, width, dtype, p0.device)
        self.forget = SwiGLUBranch(hidden, width, dtype, p0.device)
        self.width = width
        # Not a submodule or buffer: one object shared by every layer's adapter.
        object.__setattr__(self, "state", state)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        s: RoutingState = self.state
        if not s.active:
            out = out + self.retain.core(x)
            if s.default_forget_scale != 0.0:
                out = out + s.default_forget_scale * self.forget.core(x)
            return out
        dtype = out.dtype
        retain = self.retain.routed(x, s.retain_weight.to(dtype))
        forget = self.forget.routed(x, s.forget_weight.to(dtype)) * s.forget_on.to(
            dtype
        )
        return out + retain + forget


def add_dual_mlp_adapters(
    model: nn.Module, width: int, dtype: torch.dtype | None = None
) -> list[DualMLPAdapter]:
    """Freeze `model` and wrap every decoder layer's MLP. Returns the adapters, in layer order.

    `dtype` is the dtype the adapter weights are stored in (default: the model's). Training a
    bfloat16 model passes float32, so small optimizer steps are not lost to rounding.
    """
    for p in model.parameters():
        p.requires_grad_(False)
    state = RoutingState()
    hidden = model.config.hidden_size
    adapters = []
    for layer in decoder_layers(model):
        layer.mlp = DualMLPAdapter(layer.mlp, hidden, width, state, dtype)
        adapters.append(layer.mlp)
    model.gradient_routing_state = state
    return adapters


def decoder_layers(model: nn.Module) -> nn.ModuleList:
    inner = getattr(model, "model", model)
    return inner.layers


def adapters_of(model: nn.Module) -> list[DualMLPAdapter]:
    return [m for m in model.modules() if isinstance(m, DualMLPAdapter)]


def routing_state(model: nn.Module) -> RoutingState:
    return adapters_of(model)[0].state


def set_routing(
    model: nn.Module,
    retain_weight: torch.Tensor,
    forget_weight: torch.Tensor,
    forget_on: torch.Tensor,
) -> None:
    """Set per-reply routing for the following forward AND backward passes; each tensor is (batch,).

    It must stay set until the backward pass is over: with gradient checkpointing the layers are
    run again during backward. Call clear_routing afterwards.
    """
    s = routing_state(model)
    s.retain_weight = retain_weight.float().view(-1, 1, 1)
    s.forget_weight = forget_weight.float().view(-1, 1, 1)
    s.forget_on = forget_on.float().view(-1, 1, 1)
    s.active = True


def clear_routing(model: nn.Module) -> None:
    s = routing_state(model)
    s.active = False
    s.retain_weight = s.forget_weight = s.forget_on = None


@contextmanager
def routing(
    model: nn.Module,
    retain_weight: torch.Tensor,
    forget_weight: torch.Tensor,
    forget_on: torch.Tensor,
) -> Iterator[None]:
    """set_routing for the block, cleared on exit. The backward pass must happen inside the block."""
    set_routing(model, retain_weight, forget_weight, forget_on)
    try:
        yield
    finally:
        clear_routing(model)


@contextmanager
def forget_scaled(model: nn.Module, scale: float) -> Iterator[None]:
    """Plain forward passes inside the block use FORGET scaled by `scale` (0 = RETAIN only)."""
    s = routing_state(model)
    previous = s.default_forget_scale
    s.default_forget_scale = scale
    try:
        yield
    finally:
        s.default_forget_scale = previous


def routing_weights(
    flagged: torch.Tensor, is_anchor: torch.Tensor, forget_amplification: float = 2.0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Algorithm 1 lines 8-9 as per-reply tensors: (retain_weight, forget_weight, forget_on).

    Target, not flagged: both adapters, weight 1. Target, flagged: RETAIN 0, FORGET doubled.
    Anchor: sampled and trained with FORGET off, so only RETAIN is updated, weight 1.
    """
    flagged = flagged.bool() & ~is_anchor.bool()
    retain_weight = (~flagged).float()
    forget_weight = torch.where(flagged, forget_amplification, 1.0).float()
    forget_on = (~is_anchor.bool()).float()
    return retain_weight, forget_weight, forget_on


def parameter_pairs(
    model: nn.Module,
) -> dict[str, list[tuple[nn.Parameter, nn.Parameter]]]:
    """{"retain": [(weight, shadow), ...], "forget": [...]} over all layers, for the optimizer."""
    out: dict[str, list] = {"retain": [], "forget": []}
    for a in adapters_of(model):
        out["retain"] += a.retain.pairs()
        out["forget"] += a.forget.pairs()
    return out


def adapter_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """The trained adapter weights only (no base weights, no shadows), keyed by layer."""
    out = {}
    for i, a in enumerate(adapters_of(model)):
        for which, branch in (("retain", a.retain), ("forget", a.forget)):
            for name in ("gate", "up", "down"):
                out[f"layers.{i}.{which}.{name}"] = getattr(branch, name).detach().cpu()
    return out


def load_adapter_state_dict(model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    adapters = adapters_of(model)
    expected = 6 * len(adapters)
    if len(state) != expected:
        raise ValueError(
            f"adapter state has {len(state)} tensors, the model needs {expected}"
        )
    with torch.no_grad():
        for i, a in enumerate(adapters):
            for which, branch in (("retain", a.retain), ("forget", a.forget)):
                for name in ("gate", "up", "down"):
                    p = getattr(branch, name)
                    p.copy_(state[f"layers.{i}.{which}.{name}"].to(p.device, p.dtype))


def save_adapters(model: nn.Module, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(adapter_state_dict(model), tmp)
    tmp.replace(path)


def widened_mlp_weights(
    base_gate: torch.Tensor,
    base_up: torch.Tensor,
    base_down: torch.Tensor,
    adapter: dict[str, torch.Tensor] | None,
    width: int,
    forget: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One layer's (gate, up, down) for a plain MLP that equals base + RETAIN (+ FORGET).

    Hidden units are ordered [base, RETAIN, FORGET]. `adapter` holds that layer's six tensors
    keyed "retain.gate" ... "forget.down"; None gives adapters that are all zero. With
    forget=False the FORGET units keep their input weights but their output weights are zero.
    """
    hidden = base_gate.shape[1]
    kw = {"dtype": base_gate.dtype, "device": base_gate.device}
    if adapter is None:
        adapter = {
            f"{w}.{n}": torch.zeros(
                (hidden, width) if n == "down" else (width, hidden), **kw
            )
            for w in ("retain", "forget")
            for n in ("gate", "up", "down")
        }
    a = {k: v.to(**kw) for k, v in adapter.items()}
    forget_down = a["forget.down"] if forget else torch.zeros_like(a["forget.down"])
    gate = torch.cat([base_gate, a["retain.gate"], a["forget.gate"]], dim=0)
    up = torch.cat([base_up, a["retain.up"], a["forget.up"]], dim=0)
    down = torch.cat([base_down, a["retain.down"], forget_down], dim=1)
    return gate, up, down


def layer_adapter(
    state: dict[str, torch.Tensor], layer: int
) -> dict[str, torch.Tensor]:
    prefix = f"layers.{layer}."
    return {k[len(prefix) :]: v for k, v in state.items() if k.startswith(prefix)}


def write_widened_checkpoint(
    base_model_dir: str | Path,
    out_dir: str | Path,
    width: int,
    adapter_state: dict[str, torch.Tensor] | None = None,
    forget: bool = True,
    dtype: torch.dtype = torch.bfloat16,
) -> Path:
    """Write a plain checkpoint whose MLPs are 2*width wider and hold the adapters.

    adapter_state=None writes the starting point (adapters all zero), which is what the sampling
    engine loads before training. forget=False writes the deployed model (RETAIN only).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out_dir = Path(out_dir)
    model = AutoModelForCausalLM.from_pretrained(str(base_model_dir), torch_dtype=dtype)
    for i, layer in enumerate(decoder_layers(model)):
        mlp = layer.mlp
        per_layer = None if adapter_state is None else layer_adapter(adapter_state, i)
        gate, up, down = widened_mlp_weights(
            mlp.gate_proj.weight.data,
            mlp.up_proj.weight.data,
            mlp.down_proj.weight.data,
            per_layer,
            width,
            forget,
        )
        for name, weight in (("gate_proj", gate), ("up_proj", up), ("down_proj", down)):
            old = getattr(mlp, name)
            new = nn.Linear(
                weight.shape[1], weight.shape[0], bias=False, dtype=weight.dtype
            )
            new.weight.data = weight.contiguous()
            setattr(mlp, name, new)
            del old
    model.config.intermediate_size = model.config.intermediate_size + 2 * width
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir))
    AutoTokenizer.from_pretrained(str(base_model_dir)).save_pretrained(str(out_dir))
    return out_dir
