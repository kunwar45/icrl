# ABOUTME: GRAFT's split-moment AdamW (Ward et al. 2026, Eq. 1): the step direction uses the routed gradient, the per-parameter scale (second moment) uses the unrouted gradient, so routing coefficients 0 and 2 are not cancelled by Adam's normalisation
# ABOUTME: Use: SplitMomentAdamW([{"pairs": retain_pairs}, {"pairs": forget_pairs, "unrouted_scale": N / n_target}], lr=..., max_grad_norm=0.2); pairs come from gradient_routing_adapters.parameter_pairs(model)
"""
Our reimplementation of the optimizer in GRAFT (paper Section 3, Eq. 1; the authors'
MIT-licensed version is rl/split_moment.py in https://github.com/jnward/GRAFT; no
code is copied).

Each trained weight W has a zero "shadow" parameter Z (see
gradient_routing_adapters.py). After a backward pass:

    unrouted gradient  g_bar = W.grad
    routed gradient    g     = W.grad + Z.grad

The update is AdamW with

    m <- beta1 * m + (1 - beta1) * g
    v <- beta2 * v + (1 - beta2) * (unrouted_scale * g_bar)^2

`unrouted_scale` handles the paper's different averaging for the two adapters: the
RETAIN unrouted gradient is averaged over all N replies (scale 1), the FORGET one
over the target replies only (scale N / n_target).

Gradient clipping is done here, on the norm of the routed gradient over all trained
weights, with the same factor applied to both gradients. The Trainer's own clipping
would measure the norm of (W.grad, Z.grad) instead, so it must be turned off.
"""

from __future__ import annotations

import torch


class SplitMomentAdamW(torch.optim.Optimizer):
    def __init__(
        self,
        groups: list[dict],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        max_grad_norm: float | None = None,
    ) -> None:
        self._shadow: dict[int, torch.nn.Parameter] = {}
        param_groups = []
        for g in groups:
            g = dict(g)
            pairs = g.pop("pairs")
            for weight, shadow in pairs:
                self._shadow[id(weight)] = shadow
            g["params"] = [weight for weight, _ in pairs]
            g.setdefault("unrouted_scale", 1.0)
            param_groups.append(g)
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(param_groups, defaults)
        self.max_grad_norm = max_grad_norm

    def _gradients(self, p: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor] | None:
        """(routed, unrouted) for one weight, or None if it has no gradient."""
        if p.grad is None:
            return None
        shadow = self._shadow[id(p)]
        routed = p.grad if shadow.grad is None else p.grad + shadow.grad
        return routed, p.grad

    def routed_grad_norm(self) -> float:
        total = 0.0
        for group in self.param_groups:
            for p in group["params"]:
                grads = self._gradients(p)
                if grads is not None:
                    total += float(grads[0].float().pow(2).sum())
        return total**0.5

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        clip = 1.0
        if self.max_grad_norm:
            norm = self.routed_grad_norm()
            if norm > self.max_grad_norm:
                clip = self.max_grad_norm / (norm + 1e-6)
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for p in group["params"]:
                grads = self._gradients(p)
                if grads is None:
                    continue
                routed = grads[0].float() * clip
                unrouted = grads[1].float() * (clip * group["unrouted_scale"])
                state = self.state[p]
                if not state:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(routed)
                    state["v"] = torch.zeros_like(routed)
                state["step"] += 1
                t = state["step"]
                state["m"].mul_(beta1).add_(routed, alpha=1 - beta1)
                state["v"].mul_(beta2).addcmul_(unrouted, unrouted, value=1 - beta2)
                m_hat = state["m"] / (1 - beta1**t)
                v_hat = state["v"] / (1 - beta2**t)
                update = m_hat / (v_hat.sqrt() + group["eps"])
                if group["weight_decay"]:
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(update.to(p.dtype), alpha=-group["lr"])
        return loss

    def zero_grad(self, set_to_none: bool = True) -> None:
        super().zero_grad(set_to_none=set_to_none)
        for shadow in self._shadow.values():
            if shadow.grad is not None:
                if set_to_none:
                    shadow.grad = None
                else:
                    shadow.grad.zero_()
