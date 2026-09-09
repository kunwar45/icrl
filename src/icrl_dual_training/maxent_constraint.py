# ABOUTME: Maximum-entropy inverse constraint inference (Malik et al. 2021) for agent trajectories: a per-step feasibility function trained by the expert-minus-nominal likelihood gradient, with a sparsity regulariser and positive-unlabelled negative selection
# ABOUTME: Used by scripts/train_constraint.py (method: maxent_icrl) and by the alternating ODCV loop; not a CLI
"""
The constraint-inference half of ICRL, in the form the field actually uses.

Reference: Liu, Xu, Liu, Gaurav, Ganapathi Subramanian and Poupart, "A
Comprehensive Survey on Inverse Constrained Reinforcement Learning"
(arXiv:2409.07569), Section 4.2, which states the continuous-domain maximum
entropy method of Malik et al. (2021).

**Model.** Rather than one score for a whole trajectory, learn a *feasibility
function* over single steps,

    phi_omega(s, a) in (0, 1)   the probability that taking action a in state s is safe,

so that the probability a trajectory is safe factorises over its steps and the
cost is a sum of step costs (survey Section 4.2):

    P(tau safe) = prod_{(s,a) in tau} phi_omega(s, a)
    c_omega(tau) = - sum_{(s,a) in tau} log phi_omega(s, a)     (>= 0)

This is the form the CMDP wants: a cost that accumulates over a trajectory and
attributes to individual actions, which a single trajectory-level score cannot.

**Objective.** Under the maximum-entropy policy induced by that constraint
(survey Eq. 20), the feasibility function is fitted by maximising the likelihood
of the expert demonstrations. Its gradient (survey Eq. 21) is a difference of two
expectations, expert minus nominal:

    grad_omega log p(D_E) = sum_{tau_E in D_E} beta grad_omega log prod phi_omega(s_E, a_E)
                          - E_{tau ~ pi} [ beta grad_omega log prod phi_omega(s, a) ]

so the update raises feasibility on what the expert did and lowers it on what the
*current* policy does. The nominal half must therefore be resampled from the
policy every round: that alternation is what makes the method inverse
constrained RL rather than a one-shot classifier, and it is precisely what a
static safe-versus-unsafe fit leaves out.

**Regularisation.** Marking everything the expert never visited as infeasible is
a trivial maximiser, so Malik et al. add a sparsity term (survey Eq. 22) that
penalises calling trajectories infeasible; `sparsity_coeff` scales it.

**Overlap.** The plain update treats every nominal trajectory as infeasible. That
is false here: an ODCV policy produces clean and violating rollouts from the same
prompt, so a large share of the nominal pool is genuinely safe and the two terms
fight (the survey names this failure and points to positive-unlabelled learning,
Peng and Billard 2024). `negative_selection="pucl"` therefore keeps only the
nominal trajectories the current model already considers least feasible, which is
the reliable-negative step of that method.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


class StepFeasibilityHead(nn.Module):
    """phi_omega over per-step embeddings: an MLP ending in a sigmoid.

    Deliberately the same shape as the trajectory-level head so the two can be
    compared on equal capacity; the difference that matters is what a row is (one
    step here, one whole trajectory there) and how rows are aggregated.
    """

    def __init__(self, input_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, step_embeddings: torch.Tensor) -> torch.Tensor:
        """Feasibility in (0, 1) for each step."""
        return torch.sigmoid(self.net(step_embeddings).squeeze(-1))


@dataclass
class TrajectoryBatch:
    """Per-step embeddings of a set of trajectories, flattened with an index.

    ``embeddings`` is (total_steps, dim) and ``owner[i]`` is the index of the
    trajectory step i belongs to, so log-feasibility can be summed per trajectory
    with one scatter rather than a Python loop.
    """

    embeddings: torch.Tensor
    owner: torch.Tensor
    n_trajectories: int
    log_prob_ratio: torch.Tensor | None = None  # for importance weighting

    def to(self, device) -> "TrajectoryBatch":
        return TrajectoryBatch(
            self.embeddings.to(device),
            self.owner.to(device),
            self.n_trajectories,
            None if self.log_prob_ratio is None else self.log_prob_ratio.to(device),
        )


def trajectory_log_feasibility(
    head: StepFeasibilityHead, batch: TrajectoryBatch, eps: float = 1e-6
) -> torch.Tensor:
    """sum_t log phi(s_t, a_t) per trajectory; the negative of the CMDP cost."""
    phi = head(batch.embeddings).clamp(eps, 1 - eps)
    out = torch.zeros(
        batch.n_trajectories, device=phi.device, dtype=phi.dtype
    ).index_add_(0, batch.owner, torch.log(phi))
    return out


def trajectory_cost(
    head: StepFeasibilityHead,
    batch: TrajectoryBatch,
    normalise_by_length: bool = False,
    eps: float = 1e-6,
) -> torch.Tensor:
    """c(tau) = -sum_t log phi. Optionally per step, which removes the length
    confound at the price of departing from the CMDP's additive cost."""
    cost = -trajectory_log_feasibility(head, batch, eps)
    if normalise_by_length:
        counts = torch.bincount(batch.owner, minlength=batch.n_trajectories).clamp(
            min=1
        )
        cost = cost / counts.to(cost.dtype)
    return cost


def importance_weights(batch: TrajectoryBatch, clip: float = 10.0) -> torch.Tensor:
    """Weights for nominal trajectories sampled under an earlier policy iterate.

    Malik et al. reweight the nominal expectation because the buffer holds
    trajectories from previous rounds. Without stored log-probabilities every
    trajectory weighs the same, which is correct only for the round that produced
    them; the buffer is kept short for that reason.
    """
    if batch.log_prob_ratio is None:
        return torch.ones(batch.n_trajectories, device=batch.embeddings.device)
    return torch.exp(
        batch.log_prob_ratio.clamp(max=torch.log(torch.tensor(clip)))
    ).clamp(0.0, clip)


def select_reliable_negatives(
    head: StepFeasibilityHead,
    nominal: TrajectoryBatch,
    keep_fraction: float,
) -> torch.Tensor:
    """Indices of the nominal trajectories the model currently finds least feasible.

    The positive-unlabelled step: the nominal pool is unlabelled, not negative, so
    pushing all of it down fights the expert term wherever the policy already
    behaves. Keeping the lowest-log-feasibility fraction approximates the reliable
    negatives of Peng and Billard (2024) without needing a judge in the loop.
    """
    if keep_fraction >= 1.0:
        return torch.arange(nominal.n_trajectories, device=nominal.embeddings.device)
    with torch.no_grad():
        logf = trajectory_log_feasibility(head, nominal)
        counts = torch.bincount(nominal.owner, minlength=nominal.n_trajectories).clamp(
            min=1
        )
        per_step = logf / counts.to(
            logf.dtype
        )  # length-normalised, so long != negative
    k = max(1, int(round(keep_fraction * nominal.n_trajectories)))
    return torch.topk(-per_step, k).indices


def subset(batch: TrajectoryBatch, keep: torch.Tensor) -> TrajectoryBatch:
    """Restrict a batch to a set of trajectory indices, renumbering owners."""
    keep_sorted, _ = torch.sort(keep)
    mapping = torch.full(
        (batch.n_trajectories,), -1, dtype=torch.long, device=batch.owner.device
    )
    mapping[keep_sorted] = torch.arange(len(keep_sorted), device=batch.owner.device)
    mask = mapping[batch.owner] >= 0
    return TrajectoryBatch(
        embeddings=batch.embeddings[mask],
        owner=mapping[batch.owner[mask]],
        n_trajectories=len(keep_sorted),
        log_prob_ratio=(
            None if batch.log_prob_ratio is None else batch.log_prob_ratio[keep_sorted]
        ),
    )


@dataclass
class MaxEntConstraintConfig:
    """Hyperparameters of the inverse constraint inference step."""

    beta: float = 1.0
    """Scales the likelihood terms (survey Eq. 20); the reward/cost balance."""
    sparsity_coeff: float = 0.05
    """Weight on the regulariser that penalises calling trajectories infeasible."""
    negative_selection: str = "pucl"
    """'all' for the plain Malik update, 'pucl' to keep only reliable negatives."""
    pucl_keep_fraction: float = 0.5
    """Share of the nominal pool used as negatives when negative_selection='pucl'."""
    pucl_k: int = 2
    """Neighbours in the PUCL distance rule; the paper recommends 1-3."""
    batch_steps: int = 256
    """Batch size for the step-level PUCL loss."""
    lr: float = 1e-4
    weight_decay: float = 0.01
    n_steps: int = 200
    batch_trajectories: int = 16
    normalise_cost_by_length: bool = False
    """Report and train on per-step cost. Off by default: the CMDP cost is additive."""
    grad_clip: float = 1.0
    history: list = field(default_factory=list)


def maxent_constraint_update(
    head: StepFeasibilityHead,
    expert: TrajectoryBatch,
    nominal: TrajectoryBatch,
    cfg: MaxEntConstraintConfig,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict:
    """One inverse-constraint-inference phase: fit phi to the expert-nominal gap.

    Maximises the expert log-likelihood of survey Eq. 21, which in loss form is

        L = - beta * E_expert[ sum_t log phi ]
            + beta * E_nominal[ w * sum_t log phi ]
            + sparsity_coeff * E_{expert + nominal}[ 1 - prod_t phi ]

    The first two terms are the two-player gap that gives ICRL its min-max shape;
    the third is Malik's sparsity regulariser, which stops the trivial solution of
    declaring everything the expert never did infeasible.
    """
    device = next(head.parameters()).device
    expert, nominal = expert.to(device), nominal.to(device)
    opt = optimizer or torch.optim.AdamW(
        head.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    if cfg.negative_selection == "pucl":
        nominal_used = subset(
            nominal, select_reliable_negatives(head, nominal, cfg.pucl_keep_fraction)
        )
    elif cfg.negative_selection == "all":
        nominal_used = nominal
    else:
        raise ValueError(
            f"unknown negative_selection {cfg.negative_selection!r}: use 'all' or 'pucl'"
        )

    weights = importance_weights(nominal_used)
    stats = {}
    for step in range(cfg.n_steps):
        e_idx = torch.randint(
            0,
            expert.n_trajectories,
            (min(cfg.batch_trajectories, expert.n_trajectories),),
        )
        n_idx = torch.randint(
            0,
            nominal_used.n_trajectories,
            (min(cfg.batch_trajectories, nominal_used.n_trajectories),),
        )
        e_batch, n_batch = (
            subset(expert, e_idx.to(device)),
            subset(nominal_used, n_idx.to(device)),
        )

        e_logf = trajectory_log_feasibility(head, e_batch)
        n_logf = trajectory_log_feasibility(head, n_batch)
        w = weights[n_idx.to(device)]
        w = w / w.mean().clamp(min=1e-6)

        likelihood = cfg.beta * (e_logf.mean() - (w * n_logf).mean())
        # survey Eq. 22: penalise declaring trajectories infeasible, over both pools
        infeasibility = (1 - torch.exp(e_logf)).mean() + (1 - torch.exp(n_logf)).mean()
        loss = -likelihood + cfg.sparsity_coeff * infeasibility

        opt.zero_grad()
        loss.backward()
        if cfg.grad_clip:
            nn.utils.clip_grad_norm_(head.parameters(), cfg.grad_clip)
        opt.step()

        if step == cfg.n_steps - 1 or step % max(1, cfg.n_steps // 5) == 0:
            with torch.no_grad():
                stats = {
                    "step": step,
                    "loss": float(loss),
                    "expert_log_feasibility": float(e_logf.mean()),
                    "nominal_log_feasibility": float(n_logf.mean()),
                    "gap": float(e_logf.mean() - n_logf.mean()),
                    "mean_expert_cost": float(-e_logf.mean()),
                    "mean_nominal_cost": float(-n_logf.mean()),
                }
                cfg.history.append(stats)
    stats["n_expert"] = expert.n_trajectories
    stats["n_nominal_pool"] = nominal.n_trajectories
    stats["n_nominal_used"] = nominal_used.n_trajectories
    return stats


def step_feasibility_report(
    head: StepFeasibilityHead, batch: TrajectoryBatch, top_k: int = 10
) -> list[tuple[int, float]]:
    """The least feasible steps in a batch, as (row index, phi).

    The point of a per-step constraint over a trajectory-level score: it says
    *which action* was the violation, which is what makes a learned constraint
    inspectable and what the trajectory-level head could never provide.
    """
    with torch.no_grad():
        phi = head(batch.embeddings.to(next(head.parameters()).device))
    k = min(top_k, phi.numel())
    vals, idx = torch.topk(-phi, k)
    return [(int(i), float(-v)) for v, i in zip(vals, idx)]


# ─────────────────────────────────────────────────────────────────────────────
# Positive-Unlabelled Constraint Learning (Peng and Billard, IEEE RA-L 2024,
# arXiv:2408.01622), the two-step method the ICRL survey names as the answer to
# a constraint that "mistakenly classifies demonstrations as infeasible".
#
# Our situation is exactly the one it was written for. The plain expert-minus-
# policy update of Malik et al. treats every policy rollout as infeasible, but
# our policy is already clean on roughly four fifths of its rollouts, so that
# term fights the expert term and the resulting constraint over-flags: on 625
# judged traces its positive label is 37% pure and its precision ceiling is 52%.
#
# PUCL instead calls the policy's rollouts UNLABELLED. Step 1 picks out the
# subset that is reliably infeasible using a distance rule in feature space --
# not the model's own scores, which would bootstrap from noise. Step 2 fits the
# feasibility classifier by ordinary cross-entropy on the demonstrations against
# only that subset, plus a memory buffer of negatives confirmed in earlier
# rounds.
# ─────────────────────────────────────────────────────────────────────────────


def standardise_together(
    expert: torch.Tensor, nominal: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Zero-mean, unit-variance over the merged sets (PUCL Sec. III-B).

    The distance rule needs comparable dimensions; the paper standardises the
    combined data before scoring because the features are heterogeneous.
    """
    both = torch.cat([expert, nominal], dim=0)
    mu, sd = both.mean(0, keepdim=True), both.std(0, keepdim=True).clamp(min=1e-6)
    return (expert - mu) / sd, (nominal - mu) / sd


def knn_distance_to_demonstrations(
    expert_steps: torch.Tensor, nominal_steps: torch.Tensor, k: int = 2, chunk: int = 2048
) -> torch.Tensor:
    """PUCL Eq. 2: mean distance from each unlabelled step to its k nearest
    demonstration steps. A step far from everything the expert ever did is the
    one that is plausibly infeasible."""
    out = []
    kk = min(k, expert_steps.shape[0])
    for i in range(0, nominal_steps.shape[0], chunk):
        d = torch.cdist(nominal_steps[i : i + chunk], expert_steps)
        out.append(d.topk(kk, dim=1, largest=False).values.mean(dim=1))
    return torch.cat(out)


def reliable_infeasible_steps(
    expert: TrajectoryBatch,
    nominal: TrajectoryBatch,
    percentile: float = 0.3,
    k: int = 2,
    expand: bool = True,
) -> torch.Tensor:
    """PUCL Eqs. 2-4: indices of the nominal STEPS that are reliably infeasible.

    `percentile` is the share of the unlabelled pool kept, which the paper calls
    the user's belief about the size of the true infeasible region: lower keeps
    more and yields a more conservative constraint. `expand` adds, from every
    policy trajectory, the step closest to the selected set, which the paper
    includes because a demonstration hugging the true boundary otherwise leaves
    a gap between the learned and the true one.
    """
    e_std, n_std = standardise_together(expert.embeddings, nominal.embeddings)
    scores = knn_distance_to_demonstrations(e_std, n_std, k=k)
    n_keep = max(1, int(round(percentile * scores.numel())))
    idx = torch.topk(scores, n_keep).indices
    if not expand or idx.numel() == 0:
        return idx
    # Eq. 4: one extra step per trajectory, the one nearest the selected set
    keep = torch.zeros(scores.numel(), dtype=torch.bool, device=scores.device)
    keep[idx] = True
    rel = n_std[idx]
    extra = []
    for t in range(nominal.n_trajectories):
        rows = (nominal.owner == t).nonzero(as_tuple=True)[0]
        rows = rows[~keep[rows]]
        if rows.numel() == 0:
            continue
        d = torch.cdist(n_std[rows], rel).min(dim=1).values
        extra.append(rows[int(d.argmin())])
    if extra:
        idx = torch.cat([idx, torch.stack(extra)])
    return torch.unique(idx)


def pucl_constraint_update(
    head: StepFeasibilityHead,
    expert: TrajectoryBatch,
    nominal: TrajectoryBatch,
    cfg: MaxEntConstraintConfig,
    memory: torch.Tensor | None = None,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[dict, torch.Tensor]:
    """PUCL Eq. 5: cross-entropy on demonstrations against reliable negatives.

    Returns the training statistics and the memory buffer of reliable-infeasible
    step embeddings, to be carried into the next round as the paper prescribes.
    """
    device = next(head.parameters()).device
    expert, nominal = expert.to(device), nominal.to(device)
    opt = optimizer or torch.optim.AdamW(
        head.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    idx = reliable_infeasible_steps(
        expert, nominal, percentile=cfg.pucl_keep_fraction, k=cfg.pucl_k
    )
    neg = nominal.embeddings[idx]
    if memory is not None and memory.numel():
        neg = torch.cat([neg, memory.to(device)])
    pos = expert.embeddings

    for step in range(cfg.n_steps):
        p_idx = torch.randint(0, pos.shape[0], (min(cfg.batch_steps, pos.shape[0]),))
        n_idx = torch.randint(0, neg.shape[0], (min(cfg.batch_steps, neg.shape[0]),))
        # zeta = feasibility, so demonstrations are pushed to 1 and reliable
        # negatives to 0; identical in form to the supervised loss, but the
        # negative set is a selected subset rather than everything the policy did
        zp = head(pos[p_idx.to(device)]).clamp(1e-6, 1 - 1e-6)
        zn = head(neg[n_idx.to(device)]).clamp(1e-6, 1 - 1e-6)
        loss = -(torch.log(zp).mean() + torch.log(1 - zn).mean())
        opt.zero_grad()
        loss.backward()
        if cfg.grad_clip:
            nn.utils.clip_grad_norm_(head.parameters(), cfg.grad_clip)
        opt.step()

    with torch.no_grad():
        stats = {
            "loss": float(loss),
            "n_expert_steps": int(pos.shape[0]),
            "n_unlabelled_steps": int(nominal.embeddings.shape[0]),
            "n_reliable_negative_steps": int(idx.numel()),
            "n_memory_steps": 0 if memory is None else int(memory.shape[0]),
            "mean_phi_expert": float(head(pos).mean()),
            "mean_phi_reliable_negative": float(head(neg).mean()),
        }
    return stats, nominal.embeddings[idx].detach().cpu()
