#!/usr/bin/env python3
"""
Stage 2 — Train C_θ on the sycophancy conversation dataset.

This is supervised binary classification, not ICRL adversarial training.
We have clean is_safe labels, so we skip the adversarial loop entirely.

Loss: BCE(C_θ(conv), is_unsafe_label)

Checkpoint gate: val AUROC ≥ 0.75 required before proceeding to Stage 3.

Usage:
    python sycophancy/scripts/train_constraint.py \\
        +constraint=supervised_default +compute=local run_name=syco_constraint_v1

    # Smaller run on CPU / M1:
    python sycophancy/scripts/train_constraint.py \\
        +constraint=supervised_default +compute=local run_name=syco_constraint_v1 \\
        constraint.encoder.model_name=Qwen/Qwen2.5-1.5B \\
        constraint.encoder.max_length=512 \\
        constraint.training.batch_size=4
"""
from __future__ import annotations

import os
import sys

# ── Path setup ─────────────────────────────────────────────────────────────────
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
_SYCO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for _p in [_ROOT, _SYCO]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
import hydra
from omegaconf import DictConfig

from src.constraint.encoder import TrajectoryEncoder
from src.models.loader import load_model_and_tokenizer
from src.utils.compute import seed_everything, setup_accelerator
from src.utils.config import resolve_paths

from sycophancy.src.constraint.trainer import SupervisedConstraintTrainer
from sycophancy.src.data.conversation import load_conversations


@hydra.main(config_path="../configs", config_name="base", version_base=None)
def main(cfg: DictConfig):
    seed_everything(cfg.seed)
    resolve_paths(cfg)
    setup_accelerator(cfg)

    # ── Load labeled conversations ────────────────────────────────────────────
    data_dir   = os.path.join(cfg.paths.data_root, "conversations")
    safe_path   = os.path.join(data_dir, "safe.jsonl")
    unsafe_path = os.path.join(data_dir, "unsafe.jsonl")

    safe_convs   = load_conversations(safe_path)
    unsafe_convs = load_conversations(unsafe_path)
    all_convs    = safe_convs + unsafe_convs

    print(f"Loaded {len(safe_convs)} safe + {len(unsafe_convs)} unsafe = {len(all_convs)} total")

    # ── Build encoder ─────────────────────────────────────────────────────────
    print(f"Loading backbone: {cfg.constraint.encoder.model_name}")
    backbone, tokenizer = load_model_and_tokenizer(
        cfg.constraint.encoder.model_name,
        cfg,
        causal_lm=False,
    )
    ctheta = TrajectoryEncoder(
        model=backbone,
        tokenizer=tokenizer,
        max_length=cfg.constraint.encoder.max_length,
        head_hidden=cfg.constraint.encoder.head_hidden,
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    trainer = SupervisedConstraintTrainer(
        ctheta=ctheta,
        conversations=all_convs,
        val_frac=cfg.constraint.training.val_frac,
        n_epochs=cfg.constraint.training.n_epochs,
        batch_size=cfg.constraint.training.batch_size,
        lr=cfg.constraint.training.lr,
        weight_decay=cfg.constraint.training.weight_decay,
        log_dir=os.path.join(cfg.paths.log_dir, cfg.run_name),
        run_name=cfg.run_name,
    )

    trained = trainer.train()

    # ── Gate check ────────────────────────────────────────────────────────────
    passed = trainer.gate_check()

    # ── Save head ─────────────────────────────────────────────────────────────
    ckpt_path = os.path.join(cfg.paths.checkpoint_dir, cfg.run_name, "constraint_head.pt")
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    torch.save(trained.head.state_dict(), ckpt_path)
    print(f"Constraint head saved: {ckpt_path}")

    if not passed:
        print("Gate check failed — val AUROC below 0.75.  "
              "Do NOT proceed to Stage 3.  Investigate data quality or increase n_epochs.")
        sys.exit(1)

    print("Gate passed.  Proceed to Stage 3: run_finetune.py")


if __name__ == "__main__":
    main()
