#!/usr/bin/env python3
"""
Block A gate check: AUROC >= 0.75 before fine-tuning.

Usage:
    python scripts/eval_constraint.py +compute=local run_name=constraint_v1 \
        constraint.model_path=checkpoints/constraint_v1/constraint_model.pt
"""
import torch
import hydra
from omegaconf import DictConfig
from src.constraint.encoder import TrajectoryEncoder
from src.constraint.evaluator import ConstraintEvaluator
from src.data.trajectory import load_trajectories
from src.utils.config import resolve_paths


@hydra.main(config_path="../configs", config_name="base", version_base=None)
def main(cfg: DictConfig):
    resolve_paths(cfg)

    constraint_model = TrajectoryEncoder(cfg)
    constraint_model.head.load_state_dict(
        torch.load(cfg.constraint.model_path, map_location="cpu")
    )

    evaluator = ConstraintEvaluator(constraint_model)

    import os
    safe_trajs = load_trajectories(os.path.join(cfg.paths.data_root, "eval", "safe_held_out.jsonl"))
    unsafe_trajs = load_trajectories(os.path.join(cfg.paths.data_root, "eval", "unsafe_held_out.jsonl"))

    passed = evaluator.gate_check(safe_trajs, unsafe_trajs)
    metrics = evaluator.evaluate(safe_trajs, unsafe_trajs)
    print(metrics)

    if not passed:
        exit(1)


if __name__ == "__main__":
    main()
