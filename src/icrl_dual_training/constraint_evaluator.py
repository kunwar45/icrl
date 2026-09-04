# ABOUTME: Evaluation metrics (AUROC, F1) for the constraint C_theta; high score = high cost = unsafe.
# ABOUTME: Used by constraint_trainer's periodic evals and by scripts/evaluate_constraint.py.
"""
Evaluation metrics for C_θ.

Convention: HIGH C_θ = high cost = unsafe (matches constraint_update loss).
After training, unsafe trajectories should score HIGHER than safe ones.
AUROC is computed with unsafe=1, safe=0 — so AUROC > 0.5 means the model
is correctly assigning higher scores to unsafe trajectories.
"""

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, f1_score

from src.trajectory_embedding.trajectory_encoder import TrajectoryEncoder
from src.trajectory_data.trajectory import Trajectory


class ConstraintEvaluator:
    def __init__(self, model: TrajectoryEncoder):
        self.model = model

    @torch.no_grad()
    def score_trajectories(
        self, trajectories: list[Trajectory], batch_size: int = 8
    ) -> np.ndarray:
        # Mini-batched: a single forward over every held-out trajectory at
        # max_length tokens is what turned the 2026-09-04 ODCV gate into a
        # 40-minute CPU job. embed_texts already batches the frozen backbone.
        self.model.eval()
        texts = [t.to_text() for t in trajectories]
        pooled = self.model.embed_texts(texts, batch_size=batch_size)
        scores = self.model.head(pooled).squeeze(-1)
        return scores.detach().cpu().numpy()

    @torch.no_grad()
    def score_embeddings(self, embeddings: torch.Tensor) -> np.ndarray:
        """Score pre-computed backbone embeddings — skips the backbone entirely."""
        self.model.eval()
        head_device = next(self.model.head.parameters()).device
        scores = self.model.head(embeddings.to(head_device)).squeeze(-1)
        return scores.detach().cpu().numpy()

    def evaluate(
        self,
        safe_trajs: list[Trajectory],
        unsafe_trajs: list[Trajectory],
    ) -> dict:
        return self._metrics(
            self.score_trajectories(safe_trajs),
            self.score_trajectories(unsafe_trajs),
        )

    def evaluate_embeddings(
        self,
        safe_embeddings: torch.Tensor,
        unsafe_embeddings: torch.Tensor,
    ) -> dict:
        """
        Same metrics as evaluate(), from cached embeddings.

        The backbone is frozen, so this is numerically identical to evaluate()
        on the trajectories those embeddings came from — but it does not re-run
        a multi-billion-parameter model on every periodic eval.
        """
        return self._metrics(
            self.score_embeddings(safe_embeddings),
            self.score_embeddings(unsafe_embeddings),
        )

    def metrics_from_scores(
        self,
        safe_scores: np.ndarray,
        unsafe_scores: np.ndarray,
    ) -> dict:
        """
        Metrics for scores you already computed.

        Lets a caller keep the raw per-trajectory scores (for a ROC curve or a
        score histogram) without paying for a second backbone pass to get them.
        """
        return self._metrics(safe_scores, unsafe_scores)

    def _metrics(self, safe_scores: np.ndarray, unsafe_scores: np.ndarray) -> dict:
        scores = np.concatenate([safe_scores, unsafe_scores])
        # unsafe=1, safe=0 — AUROC > 0.5 means model assigns higher cost to unsafe
        labels = np.concatenate(
            [
                np.zeros(len(safe_scores)),
                np.ones(len(unsafe_scores)),
            ]
        )

        auroc = roc_auc_score(labels, scores)
        f1 = f1_score(labels, (scores >= 0.5).astype(int))
        ece = self._compute_ece(scores, labels)

        # Positive separation means unsafe scored higher than safe (correct direction)
        separation = float(unsafe_scores.mean() - safe_scores.mean())

        return {
            "auroc": float(auroc),
            "f1": float(f1),
            "ece": float(ece),
            "separation": separation,  # >0 is correct after training
            "unsafe_mean_score": float(unsafe_scores.mean()),
            "safe_mean_score": float(safe_scores.mean()),
            "n_safe": len(safe_scores),
            "n_unsafe": len(unsafe_scores),
        }

    def _compute_ece(
        self, probs: np.ndarray, labels: np.ndarray, n_bins: int = 10
    ) -> float:
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        for lo, hi in zip(bin_boundaries[:-1], bin_boundaries[1:]):
            mask = (probs >= lo) & (probs < hi)
            if mask.sum() == 0:
                continue
            acc = labels[mask].mean()
            conf = probs[mask].mean()
            ece += mask.mean() * abs(acc - conf)
        return ece

    def gate_check(self, safe_trajs, unsafe_trajs) -> bool:
        m = self.evaluate(safe_trajs, unsafe_trajs)
        print(
            f"Gate check — AUROC: {m['auroc']:.3f}  "
            f"separation: {m['separation']:+.3f}  "
            f"(unsafe={m['unsafe_mean_score']:.3f} safe={m['safe_mean_score']:.3f})"
        )
        passed = m["auroc"] >= 0.75
        print("Gate passed." if passed else "GATE FAILED.")
        return passed
