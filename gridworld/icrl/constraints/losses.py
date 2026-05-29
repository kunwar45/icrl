"""
Constraint loss functions for the contrastive ICRL update.

All functions take feasibility scores (scalars in [0,1]):
  - safe_scores:   feasibility of safe/positive trajectories  (target = 1)
  - unsafe_scores: feasibility of unsafe/negative trajectories (target = 0)

The default contrastive loss is the offline version of the ICRL gradient:
  ∇L = β·∇log∏C(safe) − E_π[β·∇log∏C(learner)]
where D_unsafe replaces the expectation (adversarial detector makes it offline).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def contrastive_loss(
    safe_scores: torch.Tensor,
    unsafe_scores: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Binary cross-entropy: safe→1, unsafe→0.
    L = −mean(log C(safe)) − mean(log(1 − C(unsafe)))
    """
    loss_safe = -torch.log(safe_scores.clamp(eps, 1.0)).mean()
    loss_unsafe = -torch.log((1.0 - unsafe_scores).clamp(eps, 1.0)).mean()
    return loss_safe + loss_unsafe


def margin_loss(
    safe_scores: torch.Tensor,
    unsafe_scores: torch.Tensor,
    margin: float = 0.3,
) -> torch.Tensor:
    """Pairwise margin: safe_score must exceed every unsafe_score by ≥ margin.
    Useful when safe and unsafe rewards overlap significantly.
    L = mean(max(0, margin − C(safe_i) + C(unsafe_j)))
    """
    diff = margin - safe_scores.unsqueeze(1) + unsafe_scores.unsqueeze(0)
    return F.relu(diff).mean()


def info_nce_loss(
    safe_embeddings: torch.Tensor,
    unsafe_embeddings: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """InfoNCE for representation-level contrastive learning.
    safe_embeddings: [N, D], unsafe_embeddings: [M, D]
    Each safe embedding is an anchor; unsafe embeddings are negatives.
    """
    safe_norm = F.normalize(safe_embeddings, dim=-1)
    unsafe_norm = F.normalize(unsafe_embeddings, dim=-1)
    logits = safe_norm @ unsafe_norm.T / temperature        # [N, M]
    labels = torch.zeros(len(safe_norm), dtype=torch.long, device=safe_norm.device)
    return F.cross_entropy(logits, labels)
