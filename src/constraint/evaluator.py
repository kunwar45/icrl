"""Evaluation metrics for the constraint function C_θ."""
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
        texts = [t.to_text() for t in trajectories]
        scores = self.model(texts).cpu().numpy()
        return scores

    def evaluate(
        self,
        safe_trajs: list[Trajectory],
        unsafe_trajs: list[Trajectory],
    ) -> dict:
        safe_scores = self.score_trajectories(safe_trajs)
        unsafe_scores = self.score_trajectories(unsafe_trajs)

        scores = np.concatenate([safe_scores, unsafe_scores])
        labels = np.concatenate([
            np.ones(len(safe_scores)),
            np.zeros(len(unsafe_scores))
        ])

        auroc = roc_auc_score(labels, scores)
        f1 = f1_score(labels, (scores >= 0.5).astype(int))
        ece = self._compute_ece(scores, labels)

        return {
            "auroc": float(auroc),
            "f1": float(f1),
            "ece": float(ece),
            "reward_gap": float(safe_scores.mean() - unsafe_scores.mean()),
            "n_safe": len(safe_trajs),
            "n_unsafe": len(unsafe_trajs),
        }

    def _compute_ece(self, probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
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
        metrics = self.evaluate(safe_trajs, unsafe_trajs)
        print(f"Gate check — AUROC: {metrics['auroc']:.3f}, ECE: {metrics['ece']:.3f}")
        passes = metrics["auroc"] >= 0.75 and metrics["ece"] <= 0.10
        if not passes:
            print("GATE FAILED. Do not proceed to fine-tuning.")
        else:
            print("Gate passed.")
        return passes
