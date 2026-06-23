"""
Supervised constraint trainer for sycophancy.

Replaces ICRLTrainer's adversarial loop with direct supervised binary
classification.  We have clean is_safe labels, so we train C_θ directly:

    Loss = BCE(C_θ(conv), is_unsafe_label)

Convention (matches the main project): HIGH C_θ = unsafe (sycophantic).

After training:
  - safe conversations  (resisted pressure) → C_θ ≈ 0
  - unsafe conversations (flipped answer)   → C_θ ≈ 1

Three numbers to watch every epoch:
  train_loss    — should decrease
  val_auroc     — should rise; gate is 0.75
  separation    — unsafe_mean_score − safe_mean_score; should be positive
"""
from __future__ import annotations

import random
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.optim import AdamW

# Shared infrastructure from the parent project — no changes needed there
from src.constraint.encoder import TrajectoryEncoder
from src.constraint.evaluator import ConstraintEvaluator
from src.utils.logging import get_logger, MetricsLogger

from sycophancy.src.data.conversation import Conversation

logger = get_logger(__name__)


def _split(items: list, val_frac: float = 0.2, seed: int = 42) -> Tuple[list, list]:
    rng = random.Random(seed)
    shuffled = list(items)
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_frac))
    return shuffled[n_val:], shuffled[:n_val]


class SupervisedConstraintTrainer:
    """
    Stage 2: train C_θ as a supervised binary classifier on labeled conversations.

    safe   (is_safe=True)  → label 0  (low cost, not sycophantic)
    unsafe (is_safe=False) → label 1  (high cost, sycophantic)

    Only the MLP head is updated; the frozen backbone is used to pre-compute
    embeddings once before the training loop.
    """

    def __init__(
        self,
        ctheta: TrajectoryEncoder,
        conversations: List[Conversation],
        *,
        val_frac: float = 0.2,
        n_epochs: int = 20,
        batch_size: int = 16,
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        log_dir: str = "logs/sycophancy/constraint",
        run_name: str = "constraint",
    ):
        self.ctheta = ctheta
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.evaluator = ConstraintEvaluator(ctheta)
        self.metrics_logger = MetricsLogger(log_dir, run_name)

        train_convs, val_convs = _split(conversations, val_frac=val_frac)
        self.train_convs = train_convs
        self.val_convs   = val_convs

        logger.info(
            "SupervisedConstraintTrainer  train=%d  val=%d  "
            "safe=%d  unsafe=%d",
            len(train_convs),
            len(val_convs),
            sum(1 for c in conversations if c.is_safe),
            sum(1 for c in conversations if not c.is_safe),
        )

    def train(self) -> TrajectoryEncoder:
        # Pre-compute all embeddings once (backbone is frozen)
        logger.info("Pre-computing train embeddings (%d conversations)...", len(self.train_convs))
        train_embs   = self.ctheta.embed_texts([c.to_text() for c in self.train_convs])
        train_labels = torch.tensor(
            [0.0 if c.is_safe else 1.0 for c in self.train_convs],
            dtype=torch.float32,
            device=train_embs.device,
        )

        logger.info("Pre-computing val embeddings (%d conversations)...", len(self.val_convs))
        val_embs   = self.ctheta.embed_texts([c.to_text() for c in self.val_convs])
        val_labels = torch.tensor(
            [0.0 if c.is_safe else 1.0 for c in self.val_convs],
            dtype=torch.float32,
            device=val_embs.device,
        )

        optimizer = AdamW(self.ctheta.head.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        n_train = train_embs.size(0)
        best_auroc = 0.0

        for epoch in range(1, self.n_epochs + 1):
            self.ctheta.train()
            perm = torch.randperm(n_train, device=train_embs.device)
            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, n_train, self.batch_size):
                idx = perm[start : start + self.batch_size]
                embs   = train_embs[idx]
                labels = train_labels[idx]

                scores = self.ctheta.head(embs).squeeze(-1)
                loss   = F.binary_cross_entropy(scores, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches  += 1

            # ── Validation ────────────────────────────────────────────────────
            self.ctheta.eval()
            with torch.no_grad():
                val_scores = self.ctheta.head(val_embs).squeeze(-1).cpu().numpy()

            val_labels_np = val_labels.cpu().numpy()
            try:
                auroc = float(roc_auc_score(val_labels_np, val_scores))
            except ValueError:
                auroc = float("nan")

            safe_mask   = val_labels_np == 0
            unsafe_mask = val_labels_np == 1
            sep = (
                float(val_scores[unsafe_mask].mean() - val_scores[safe_mask].mean())
                if safe_mask.any() and unsafe_mask.any()
                else float("nan")
            )

            metrics = {
                "epoch":        epoch,
                "train_loss":   epoch_loss / max(n_batches, 1),
                "val_auroc":    auroc,
                "separation":   sep,
                "unsafe_score": float(val_scores[unsafe_mask].mean()) if unsafe_mask.any() else float("nan"),
                "safe_score":   float(val_scores[safe_mask].mean()) if safe_mask.any() else float("nan"),
            }
            self.metrics_logger.log(metrics, step=epoch)

            logger.info(
                "[%3d/%d]  loss=%.4f  val_auroc=%.3f  sep=%+.3f"
                "  unsafe=%.3f  safe=%.3f",
                epoch, self.n_epochs,
                metrics["train_loss"], auroc, sep,
                metrics["unsafe_score"], metrics["safe_score"],
            )

            if auroc > best_auroc:
                best_auroc = auroc

        logger.info("Training done.  Best val AUROC: %.3f", best_auroc)
        return self.ctheta

    def gate_check(self) -> bool:
        """Return True iff val AUROC >= 0.75 on the current model state."""
        self.ctheta.eval()
        safe_convs   = [c for c in self.val_convs if     c.is_safe]
        unsafe_convs = [c for c in self.val_convs if not c.is_safe]

        # Reuse parent evaluator (safe=expert, unsafe=policy convention)
        m = self.evaluator.evaluate(safe_convs, unsafe_convs)
        print(
            f"Gate check — AUROC: {m['auroc']:.3f}  "
            f"separation: {m['separation']:+.3f}  "
            f"(unsafe={m['unsafe_mean_score']:.3f} safe={m['safe_mean_score']:.3f})"
        )
        passed = m["auroc"] >= 0.75
        print("Gate passed." if passed else "GATE FAILED — AUROC below 0.75.")
        return passed
