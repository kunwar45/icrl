#!/usr/bin/env python3
# ABOUTME: Trains the constraint function C_theta via ICRL, contrasting expert vs unsafe demo trajectories
# ABOUTME: Run: python scripts/train_constraint.py +icrl_dual_training=constraint_default +compute=local run_name=...
"""
Block A: Train C_θ via ICRL.

Offline mode (default): uses the *train* split of pre-collected demos.
  - Expert demos : <data_root>/train/safe.jsonl
  - Policy demos : <data_root>/train/unsafe.jsonl
Produce those with `python scripts/make_demo_splits.py`; the held-out split is what
scripts/evaluate_constraint.py scores, so the gate check stays honest.

Online mode: pass a rollout_fn to collect fresh policy trajectories each
iteration (wire up run_episode() when the PPO loop is connected).

Usage:
    python scripts/make_demo_splits.py
    python scripts/train_constraint.py +icrl_dual_training=constraint_default +compute=local \
        run_name=constraint_v1

    # Smaller encoder for local testing on M1:
    python scripts/train_constraint.py +icrl_dual_training=constraint_default +compute=local \
        run_name=constraint_v1 constraint.encoder.model_name=Qwen/Qwen2.5-1.5B \
        constraint.encoder.max_length=512 constraint.training.batch_size=4

    # Reuse embeddings from scripts/embed_trajectories.py (skips the backbone):
    python scripts/train_constraint.py +icrl_dual_training=constraint_default +compute=local \
        run_name=constraint_v1 \
        constraint.encoder.safe_embeddings_path=embeddings/safe.pt \
        constraint.encoder.unsafe_embeddings_path=embeddings/unsafe.pt
"""
# Make `src.*` importable when run as `python scripts/<name>.py` from anywhere.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import json
import os
import sys

import torch
import hydra
from omegaconf import DictConfig

from src.trajectory_embedding.trajectory_encoder import TrajectoryEncoder, save_constraint_head
from src.icrl_dual_training.constraint_trainer import ICRLTrainer
from src.icrl_dual_training.constraint_evaluator import ConstraintEvaluator
from src.trajectory_data.trajectory import load_trajectories
from src.models.model_loader import load_model_and_tokenizer
from src.utils.device_setup import seed_everything, setup_accelerator
from src.utils.logging import quiet_third_party_logs
from src.utils.config import resolve_paths, data_path, constraint_head_path


def _load_embeddings(path, expected_label: str):
    """Load an embed_trajectories.py bundle and return its (N, H) tensor."""
    if not path:
        return None
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Embedding bundle not found: {path}\n"
            f"Build it with: python scripts/embed_trajectories.py "
            f"--jsonl <demos.jsonl> --label {expected_label} --output {path}"
        )
    bundle = torch.load(path, weights_only=False)
    if bundle.get("label") != expected_label:
        raise ValueError(
            f"{path} holds '{bundle.get('label')}' embeddings but was passed as "
            f"'{expected_label}'."
        )
    return bundle["embeddings"]


@hydra.main(config_path="../configs", config_name="base", version_base=None)
def main(cfg: DictConfig):
    seed_everything(cfg.seed)
    resolve_paths(cfg)
    quiet_third_party_logs()
    accelerator = setup_accelerator(cfg)

    # ── Load demos (train split) ──────────────────────────────────────────────
    expert_path = data_path(cfg, cfg.data.train_safe)
    policy_path = data_path(cfg, cfg.data.train_unsafe)
    for p in (expert_path, policy_path):
        if not os.path.exists(p):
            print(f"Missing {p}. Run: python scripts/make_demo_splits.py", file=sys.stderr)
            sys.exit(1)

    expert_trajs = load_trajectories(expert_path)
    policy_trajs = load_trajectories(policy_path)
    print(f"Expert demos : {len(expert_trajs)}  ({expert_path})")
    print(f"Policy demos : {len(policy_trajs)}  ({policy_path})")

    # ── Cached embeddings (optional) ──────────────────────────────────────────
    expert_embs = _load_embeddings(cfg.constraint.encoder.safe_embeddings_path, "safe")
    policy_embs = _load_embeddings(cfg.constraint.encoder.unsafe_embeddings_path, "unsafe")

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
        expert_embeddings=expert_embs,
        policy_embeddings=policy_embs,
    )

    trained = trainer.train(n_iterations=cfg.constraint.training.n_iterations)

    # ── Train-set metrics (not the gate — see evaluate_constraint.py) ─────────────
    evaluator = ConstraintEvaluator(trained)
    if expert_embs is not None and policy_embs is not None:
        # Identical numbers without a second pass over the frozen backbone.
        metrics = evaluator.evaluate_embeddings(
            trainer.expert_embeddings, trainer.policy_embeddings,
        )
    else:
        metrics = evaluator.evaluate(expert_trajs, policy_trajs)
    print(f"\nTrain-split metrics: {metrics}")

    # ── Save head ─────────────────────────────────────────────────────────────
    ckpt_path = constraint_head_path(cfg)
    save_constraint_head(
        trained, ckpt_path,
        model_name=cfg.constraint.encoder.model_name,
        max_length=cfg.constraint.encoder.max_length,
    )
    print(f"Constraint head saved: {ckpt_path}")

    metrics_path = os.path.join(os.path.dirname(ckpt_path), "train_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Train metrics saved: {metrics_path}")

    # The real gate runs on the held-out split:
    #   python scripts/evaluate_constraint.py ... run_name=<same run_name>
    if metrics["auroc"] < cfg.constraint.evaluation.auroc_gate:
        print(f"NOTE: train AUROC {metrics['auroc']:.3f} is below the "
              f"{cfg.constraint.evaluation.auroc_gate} gate — the held-out gate "
              f"will almost certainly fail too.")


if __name__ == "__main__":
    main()
