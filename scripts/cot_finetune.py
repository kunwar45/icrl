#!/usr/bin/env python3
"""
CoT fine-tuning: thin wrapper around run_finetune.py pointing the main
Lagrangian PPO pipeline at CoT reasoning trace data.

Usage:
    python scripts/cot_finetune.py +compute=local +cot=finetune \
        run_name=cot_finetune_eps0.1 \
        finetune.constraint.epsilon=0.1 \
        finetune.constraint.constraint_model_path=checkpoints/cot_constraint_v1/constraint_model.pt

For multi-seed / multi-epsilon sweeps, use sweep.py:
    python scripts/sweep.py --mode local --script scripts/cot_finetune.py
"""
import torch
import hydra
from omegaconf import DictConfig

from src.constraint.encoder import TrajectoryEncoder
from src.finetune.lagrangian import LagrangianPPOTrainer
from src.utils.compute import seed_everything, setup_accelerator
from src.utils.config import resolve_paths
from src.utils.logging import get_logger

logger = get_logger(__name__)


@hydra.main(config_path="../configs", config_name="base", version_base=None)
def main(cfg: DictConfig):
    seed_everything(cfg.seed)
    resolve_paths(cfg)
    accelerator = setup_accelerator(cfg)

    # Load frozen constraint model trained on CoT data
    constraint_model = TrajectoryEncoder(cfg)
    constraint_model.head.load_state_dict(
        torch.load(cfg.finetune.constraint.constraint_model_path, map_location="cpu")
    )
    for p in constraint_model.parameters():
        p.requires_grad = False
    constraint_model = accelerator.prepare(constraint_model)

    # TODO (Kunwar): plug in answer-correctness reward model for GSM8K / ARC
    reward_model = None
    # TODO (Kunwar): plug in task environment that generates CoT episodes
    task_env = None

    trainer = LagrangianPPOTrainer(cfg, constraint_model, reward_model, task_env)
    trainer.train()


if __name__ == "__main__":
    main()
