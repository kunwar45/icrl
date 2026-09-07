# ABOUTME: Tests the maximum-entropy inverse constraint inference step: cost is additive over steps, the update opens the expert-nominal gap, and positive-unlabelled selection keeps the violating half of a mixed nominal pool
# ABOUTME: Run: pytest tests/test_maxent_constraint.py
import torch

from src.icrl_dual_training.maxent_constraint import (
    MaxEntConstraintConfig,
    StepFeasibilityHead,
    TrajectoryBatch,
    maxent_constraint_update,
    select_reliable_negatives,
    subset,
    trajectory_cost,
    trajectory_log_feasibility,
)

DIM = 16


def make_batch(vectors: list[list[torch.Tensor]]) -> TrajectoryBatch:
    """One inner list per trajectory, one tensor per step."""
    embeddings, owner = [], []
    for i, steps in enumerate(vectors):
        for s in steps:
            embeddings.append(s)
            owner.append(i)
    return TrajectoryBatch(
        embeddings=torch.stack(embeddings),
        owner=torch.tensor(owner, dtype=torch.long),
        n_trajectories=len(vectors),
    )


def separable_pools(n=24, steps=4, seed=0):
    """Expert and nominal trajectories that a linear head can separate."""
    g = torch.Generator().manual_seed(seed)
    safe = torch.zeros(DIM)
    safe[0] = 1.0
    unsafe = torch.zeros(DIM)
    unsafe[1] = 1.0
    expert = make_batch(
        [
            [safe + 0.05 * torch.randn(DIM, generator=g) for _ in range(steps)]
            for _ in range(n)
        ]
    )
    nominal = make_batch(
        [
            [unsafe + 0.05 * torch.randn(DIM, generator=g) for _ in range(steps)]
            for _ in range(n)
        ]
    )
    return expert, nominal


def test_cost_is_additive_over_steps():
    """The CMDP wants a cost that accumulates; a two-step trajectory of the same
    action must cost twice a one-step one."""
    torch.manual_seed(0)
    head = StepFeasibilityHead(DIM)
    v = torch.randn(DIM)
    one = make_batch([[v]])
    two = make_batch([[v, v]])
    c1 = float(trajectory_cost(head, one)[0])
    c2 = float(trajectory_cost(head, two)[0])
    assert abs(c2 - 2 * c1) < 1e-5
    assert c1 > 0


def test_length_normalisation_removes_the_length_effect():
    torch.manual_seed(0)
    head = StepFeasibilityHead(DIM)
    v = torch.randn(DIM)
    one = make_batch([[v]])
    two = make_batch([[v, v]])
    c1 = float(trajectory_cost(head, one, normalise_by_length=True)[0])
    c2 = float(trajectory_cost(head, two, normalise_by_length=True)[0])
    assert abs(c2 - c1) < 1e-5


def test_update_opens_the_expert_nominal_gap():
    """The whole point of Eq. 21: expert log-feasibility up, nominal down."""
    torch.manual_seed(0)
    head = StepFeasibilityHead(DIM)
    expert, nominal = separable_pools()
    with torch.no_grad():
        before = float(
            trajectory_log_feasibility(head, expert).mean()
            - trajectory_log_feasibility(head, nominal).mean()
        )
    cfg = MaxEntConstraintConfig(
        n_steps=150, negative_selection="all", sparsity_coeff=0.0
    )
    stats = maxent_constraint_update(head, expert, nominal, cfg)
    with torch.no_grad():
        after = float(
            trajectory_log_feasibility(head, expert).mean()
            - trajectory_log_feasibility(head, nominal).mean()
        )
    assert after > before + 0.5
    assert stats["mean_nominal_cost"] > stats["mean_expert_cost"]


def test_pucl_picks_the_violating_half_of_a_mixed_pool():
    """The nominal pool is a mixture, not a negative set: an ODCV policy emits clean
    and violating rollouts from the same prompt. Reliable-negative selection has to
    find the violating half, or the nominal term fights the expert term."""
    torch.manual_seed(0)
    head = StepFeasibilityHead(DIM)
    expert, violating = separable_pools(n=20, seed=1)
    # a nominal pool that is half expert-like and half violating
    clean_like, _ = separable_pools(n=20, seed=2)
    mixed = TrajectoryBatch(
        embeddings=torch.cat([clean_like.embeddings, violating.embeddings]),
        owner=torch.cat(
            [clean_like.owner, violating.owner + clean_like.n_trajectories]
        ),
        n_trajectories=clean_like.n_trajectories + violating.n_trajectories,
    )
    cfg = MaxEntConstraintConfig(
        n_steps=150, negative_selection="all", sparsity_coeff=0.0
    )
    maxent_constraint_update(head, expert, violating, cfg)

    keep = select_reliable_negatives(head, mixed, keep_fraction=0.5)
    chosen_violating = (keep >= clean_like.n_trajectories).float().mean()
    assert chosen_violating > 0.9, chosen_violating


def test_pucl_reduces_the_nominal_pool_it_trains_on():
    torch.manual_seed(0)
    head = StepFeasibilityHead(DIM)
    expert, nominal = separable_pools(n=20)
    cfg = MaxEntConstraintConfig(
        n_steps=20, negative_selection="pucl", pucl_keep_fraction=0.5
    )
    stats = maxent_constraint_update(head, expert, nominal, cfg)
    assert stats["n_nominal_pool"] == 20
    assert stats["n_nominal_used"] == 10


def test_subset_renumbers_owners_consistently():
    b = make_batch([[torch.randn(DIM)] * 2, [torch.randn(DIM)] * 3, [torch.randn(DIM)]])
    s = subset(b, torch.tensor([0, 2]))
    assert s.n_trajectories == 2
    assert s.embeddings.shape[0] == 3
    assert sorted(set(s.owner.tolist())) == [0, 1]
