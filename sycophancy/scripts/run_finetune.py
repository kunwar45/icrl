#!/usr/bin/env python3
"""
Stage 3 — Lagrangian constrained PPO fine-tuning for sycophancy resistance.

Mirrors scripts/run_finetune.py from the main project.

Run at multiple epsilon values to sweep the constraint budget:
    python sycophancy/scripts/run_finetune.py +finetune=lagrangian_ppo +compute=local \\
        run_name=syco_finetune_eps0.05 finetune.constraint.epsilon=0.05

    python sycophancy/scripts/run_finetune.py +finetune=lagrangian_ppo +compute=local \\
        run_name=syco_finetune_eps0.10 finetune.constraint.epsilon=0.10

    python sycophancy/scripts/run_finetune.py +finetune=lagrangian_ppo +compute=local \\
        run_name=syco_finetune_eps0.20 finetune.constraint.epsilon=0.20

The question pool is loaded from data/conversations/eval_pool.jsonl (held out from
constraint training).  Build it with:
    python sycophancy/scripts/build_dataset.py --preview  # verify data first
    # Then generate a separate eval pool:
    python sycophancy/scripts/build_dataset.py --n_examples 200 --seed 99 --out eval_pool
"""
from __future__ import annotations

import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
_SYCO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for _p in [_ROOT, _SYCO]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import json
import torch
import hydra
from omegaconf import DictConfig

from src.constraint.encoder import TrajectoryEncoder
from src.models.loader import load_model_and_tokenizer
from src.utils.compute import seed_everything, setup_accelerator
from src.utils.config import resolve_paths

from sycophancy.src.finetune.lagrangian import SycophancyPPOTrainer


def _load_question_pool(path: str) -> list[dict]:
    """Load question pool JSONL (keys: question, correct_answer, wrong_answers)."""
    pool = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                pool.append(json.loads(line))
    return pool


@hydra.main(config_path="../configs", config_name="base", version_base=None)
def main(cfg: DictConfig):
    seed_everything(cfg.seed)
    resolve_paths(cfg)
    setup_accelerator(cfg)

    # ── Load frozen constraint model ──────────────────────────────────────────
    backbone, tokenizer = load_model_and_tokenizer(
        cfg.constraint.encoder.model_name,
        cfg,
        causal_lm=False,
    )
    constraint_model = TrajectoryEncoder(
        model=backbone,
        tokenizer=tokenizer,
        max_length=cfg.constraint.encoder.max_length,
        head_hidden=cfg.constraint.encoder.head_hidden,
    )
    ckpt = torch.load(cfg.finetune.constraint.constraint_model_path, map_location="cpu")
    constraint_model.head.load_state_dict(ckpt)
    for p in constraint_model.parameters():
        p.requires_grad_(False)

    # ── Load question pool ────────────────────────────────────────────────────
    pool_path = os.path.join(cfg.paths.data_root, "conversations", "eval_pool.jsonl")
    question_pool = _load_question_pool(pool_path)
    print(f"Question pool: {len(question_pool)} items")

    # ── Train ─────────────────────────────────────────────────────────────────
    trainer = SycophancyPPOTrainer(
        cfg=cfg,
        constraint_model=constraint_model,
        question_pool=question_pool,
    )
    trainer.train()


if __name__ == "__main__":
    main()
