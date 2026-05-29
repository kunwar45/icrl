#!/usr/bin/env python3
"""
Block A: Train C_θ via adversarial ICRL.

Usage:
    python scripts/train_constraint.py +compute=local run_name=constraint_v1
    python scripts/train_constraint.py +compute=carleton run_name=constraint_v1
"""
import os
import torch
import hydra
from omegaconf import DictConfig
from src.constraint.encoder import TrajectoryEncoder
from src.constraint.trainer import ICRLTrainer
from src.data.trajectory import load_trajectories
from src.utils.compute import seed_everything, setup_accelerator
from src.utils.config import resolve_paths


@hydra.main(config_path="../configs", config_name="base", version_base=None)
def main(cfg: DictConfig):
    seed_everything(cfg.seed)
    resolve_paths(cfg)
    accelerator = setup_accelerator(cfg)

    safe_trajs = []
    for task_type in cfg.constraint.task_types:
        path = os.path.join(cfg.paths.data_root, "demos", f"{task_type}_safe.jsonl")
        safe_trajs.extend(load_trajectories(path))
    print(f"Loaded {len(safe_trajs)} safe trajectories")

    constraint_model = TrajectoryEncoder(cfg)
    constraint_model = accelerator.prepare(constraint_model)

    trainer = ICRLTrainer(cfg, constraint_model, safe_trajs)
    trained_model = trainer.train()

    ckpt_path = os.path.join(cfg.paths.checkpoint_dir, cfg.run_name, "constraint_model.pt")
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    torch.save(trained_model.head.state_dict(), ckpt_path)
    print(f"Constraint model saved: {ckpt_path}")


if __name__ == "__main__":
    main()
