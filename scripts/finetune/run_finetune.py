#!/usr/bin/env python3
"""
Block B: Lagrangian constrained fine-tuning.

Requires a trained constraint head (scripts/constraint/train_constraint.py) that has
passed the held-out gate (scripts/constraint/eval_constraint.py).

Usage:
    python scripts/finetune/run_finetune.py +constraint=icrl_default \
        +finetune=lagrangian_ppo +compute=local \
        run_name=finetune_eps0.1 finetune.constraint.epsilon=0.1 \
        constraint.head_path=checkpoints/constraint_v1/constraint_head.pt

    # No browser / no CRM — exercises the whole loop against the mock env:
    python scripts/finetune/run_finetune.py +constraint=icrl_default \
        +finetune=lagrangian_ppo +compute=local run_name=smoke \
        finetune.env.backend=mock finetune.ppo.steps=3
"""
# Make `src.*` importable when run as `python scripts/<name>.py` from anywhere.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent.parent))

import os
import sys

import torch
import hydra
from omegaconf import DictConfig, OmegaConf

from src.constraint.encoder import TrajectoryEncoder, load_constraint_head
from src.finetune.lagrangian import LagrangianPPOTrainer
from src.finetune.reward import build_reward_model
from src.finetune.rollout import build_env_provider
from src.models.loader import load_model_and_tokenizer
from src.utils.compute import seed_everything, setup_accelerator
from src.utils.logging import quiet_third_party_logs
from src.utils.config import resolve_paths, constraint_head_path


def load_constraint_model(cfg: DictConfig) -> TrajectoryEncoder:
    """Rebuild C_theta and load its trained head."""
    head_path = OmegaConf.select(cfg, "finetune.constraint.constraint_model_path") \
        or constraint_head_path(cfg)
    if not os.path.exists(head_path):
        print(f"No constraint head at {head_path}.\n"
              f"Run scripts/constraint/train_constraint.py first, or set "
              f"finetune.constraint.constraint_model_path.", file=sys.stderr)
        sys.exit(1)

    backbone, tokenizer = load_model_and_tokenizer(
        cfg.constraint.encoder.model_name, cfg, causal_lm=False,
    )
    model = TrajectoryEncoder(
        model=backbone,
        tokenizer=tokenizer,
        max_length=cfg.constraint.encoder.max_length,
        head_hidden=cfg.constraint.encoder.head_hidden,
    )
    load_constraint_head(model, head_path,
                         model_name=cfg.constraint.encoder.model_name)
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    print(f"Constraint head loaded: {head_path}")
    return model


@hydra.main(config_path="../../configs", config_name="base", version_base=None)
def main(cfg: DictConfig):
    seed_everything(cfg.seed)
    resolve_paths(cfg)
    quiet_third_party_logs()
    accelerator = setup_accelerator(cfg)

    constraint_model = load_constraint_model(cfg)
    constraint_model = accelerator.prepare(constraint_model)

    reward_model = build_reward_model(cfg)
    task_env = build_env_provider(cfg)
    print(f"Rollout env: {type(task_env).__name__} "
          f"({len(task_env.load_tasks())} tasks)")

    trainer = LagrangianPPOTrainer(cfg, constraint_model, reward_model, task_env)
    trainer.train()


if __name__ == "__main__":
    main()
