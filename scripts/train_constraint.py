#!/usr/bin/env python3
"""
Block A: Train C_θ via ICRL.

Offline mode (default): uses pre-collected safe + unsafe demos.
  - Expert demos : data/demos/safe.jsonl
  - Policy demos : data/demos/unsafe.jsonl

Online mode: pass a rollout_fn to collect fresh policy trajectories each
iteration (wire up run_episode() when the PPO loop is connected).

Usage:
    python scripts/train_constraint.py +constraint=icrl_default +compute=local \
        run_name=constraint_v1

    # Smaller encoder for local testing on M1:
    python scripts/train_constraint.py +constraint=icrl_default +compute=local \
        run_name=constraint_v1 constraint.encoder.model_name=Qwen/Qwen2.5-1.5B \
        constraint.encoder.max_length=512 constraint.training.batch_size=4
"""
import os
import torch
import hydra
from omegaconf import DictConfig

from src.constraint.encoder import TrajectoryEncoder
from src.constraint.trainer import ICRLTrainer
from src.constraint.evaluator import ConstraintEvaluator
from src.data.trajectory import load_trajectories
from src.models.loader import load_model_and_tokenizer
from src.utils.compute import seed_everything, setup_accelerator
from src.utils.config import resolve_paths


@hydra.main(config_path="../configs", config_name="base", version_base=None)
def main(cfg: DictConfig):
    seed_everything(cfg.seed)
    resolve_paths(cfg)
    accelerator = setup_accelerator(cfg)

    # ── Load demos ────────────────────────────────────────────────────────────
    demos_dir = os.path.join(cfg.paths.data_root, "demos")
    expert_trajs = load_trajectories(os.path.join(demos_dir, "safe.jsonl"))
    policy_trajs = load_trajectories(os.path.join(demos_dir, "unsafe.jsonl"))
    print(f"Expert demos : {len(expert_trajs)}")
    print(f"Policy demos : {len(policy_trajs)}")

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
    ctheta = accelerator.prepare(ctheta)

    # ── Train ─────────────────────────────────────────────────────────────────
    trainer = ICRLTrainer(
        ctheta=ctheta,
        expert_trajs=expert_trajs,
        policy_trajs=policy_trajs,      # offline mode: fixed policy pool
        beta=cfg.constraint.icrl.beta,
        lambda_c=cfg.constraint.icrl.lambda_c,
        n_constraint_steps=cfg.constraint.training.n_constraint_steps,
        batch_size=cfg.constraint.training.batch_size,
        lr=cfg.constraint.training.lr,
        weight_decay=cfg.constraint.training.weight_decay,
        eval_every=cfg.constraint.training.eval_every,
        log_dir=os.path.join(cfg.paths.log_dir, cfg.run_name),
        run_name=cfg.run_name,
    )

    trained = trainer.train(n_iterations=cfg.constraint.training.n_iterations)

    # ── Gate check ────────────────────────────────────────────────────────────
    evaluator = ConstraintEvaluator(trained)
    metrics = evaluator.evaluate(expert_trajs, policy_trajs)
    print(f"\nFinal metrics: {metrics}")
    passed = evaluator.gate_check(expert_trajs, policy_trajs)

    # ── Save head ─────────────────────────────────────────────────────────────
    ckpt_path = os.path.join(cfg.paths.checkpoint_dir, cfg.run_name, "constraint_head.pt")
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    torch.save(trained.head.state_dict(), ckpt_path)
    print(f"Constraint head saved: {ckpt_path}")

    if not passed:
        print("Gate check failed — AUROC below threshold.")
        exit(1)


if __name__ == "__main__":
    main()
