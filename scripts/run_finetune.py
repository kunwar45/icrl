#!/usr/bin/env python3
"""
Block B: Lagrangian constrained PPO fine-tuning.

Usage:
    python scripts/run_finetune.py +compute=gcp run_name=finetune_eps0.1 \
        finetune.constraint.epsilon=0.1
"""
import torch
import hydra
from omegaconf import DictConfig
from src.constraint.encoder import TrajectoryEncoder
from src.finetune.lagrangian import LagrangianPPOTrainer
from src.utils.compute import seed_everything, setup_accelerator
from src.utils.config import resolve_paths


@hydra.main(config_path="../configs", config_name="base", version_base=None)
def main(cfg: DictConfig):
    seed_everything(cfg.seed)
    resolve_paths(cfg)
    accelerator = setup_accelerator(cfg)

    constraint_model = TrajectoryEncoder(cfg)
    constraint_model.head.load_state_dict(
        torch.load(cfg.finetune.constraint.constraint_model_path, map_location="cpu")
    )
    for p in constraint_model.parameters():
        p.requires_grad = False
    constraint_model = accelerator.prepare(constraint_model)

    # TODO (Kunwar): plug in reward model and task environment
    reward_model = None
    task_env = None

    trainer = LagrangianPPOTrainer(cfg, constraint_model, reward_model, task_env)
    trainer.train()


if __name__ == "__main__":
    main()
