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

from src.constraint.encoder import TrajectoryEncoder
from src.data.trajectory import Trajectory


class ConstraintEvaluator:
    def __init__(self, model: TrajectoryEncoder):
        self.model = model

    @torch.no_grad()
    def score_trajectories(self, trajectories: list[Trajectory]) -> np.ndarray:
        self.model.eval()
        texts  = [t.to_text() for t in trajectories]
        scores = self.model(texts).cpu().numpy()
        return scores

    def evaluate(
        self,
        safe_trajs: list[Trajectory],
        unsafe_trajs: list[Trajectory],
    ) -> dict:
        safe_scores   = self.score_trajectories(safe_trajs)
        unsafe_scores = self.score_trajectories(unsafe_trajs)

        scores = np.concatenate([safe_scores, unsafe_scores])
        # unsafe=1, safe=0 — AUROC > 0.5 means model assigns higher cost to unsafe
        labels = np.concatenate([
            np.zeros(len(safe_scores)),
            np.ones(len(unsafe_scores)),
        ])

        auroc = roc_auc_score(labels, scores)
        f1    = f1_score(labels, (scores >= 0.5).astype(int))
        ece   = self._compute_ece(scores, labels)

        # Positive separation means unsafe scored higher than safe (correct direction)
        separation = float(unsafe_scores.mean() - safe_scores.mean())

        return {
            "auroc":             float(auroc),
            "f1":                float(f1),
            "ece":               float(ece),
            "separation":        separation,          # >0 is correct after training
            "unsafe_mean_score": float(unsafe_scores.mean()),
            "safe_mean_score":   float(safe_scores.mean()),
            "n_safe":            len(safe_trajs),
            "n_unsafe":          len(unsafe_trajs),
        }

    def _compute_ece(self, probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        for lo, hi in zip(bin_boundaries[:-1], bin_boundaries[1:]):
            mask = (probs >= lo) & (probs < hi)
            if mask.sum() == 0:
                continue
            acc  = labels[mask].mean()
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
