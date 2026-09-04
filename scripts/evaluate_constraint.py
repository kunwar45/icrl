#!/usr/bin/env python3
# ABOUTME: Gate check for the trained constraint head: AUROC on held-out demos; exit 1 means do not fine-tune
# ABOUTME: Run: python scripts/evaluate_constraint.py +icrl_dual_training=constraint_default +compute=local run_name=...
"""
Block A gate check: AUROC >= constraint.evaluation.auroc_gate on the HELD-OUT
split (task instances C_theta never trained on).

Usage:
    python scripts/evaluate_constraint.py +icrl_dual_training=constraint_default +compute=local \
        run_name=constraint_v1

    # Explicit head checkpoint:
    python scripts/evaluate_constraint.py +icrl_dual_training=constraint_default +compute=local \
        run_name=constraint_v1 \
        constraint.head_path=checkpoints/constraint_v1/constraint_head.pt

Exit code 1 means the gate failed — do not proceed to fine-tuning.
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

from src.trajectory_embedding.trajectory_encoder import (
    TrajectoryEncoder,
    load_constraint_head,
)
from src.icrl_dual_training.constraint_evaluator import ConstraintEvaluator
from src.trajectory_data.trajectory import load_trajectories
from src.models.model_loader import load_model_and_tokenizer
from src.utils.device_setup import seed_everything
from src.utils.logging import quiet_third_party_logs
from src.utils.config import resolve_paths, data_path, constraint_head_path, run_dir


@hydra.main(config_path="../configs", config_name="base", version_base=None)
def main(cfg: DictConfig):
    seed_everything(cfg.seed)
    resolve_paths(cfg)
    quiet_third_party_logs()

    head_path = constraint_head_path(cfg)
    if not os.path.exists(head_path):
        print(
            f"No constraint head at {head_path}. Run scripts/train_constraint.py first.",
            file=sys.stderr,
        )
        sys.exit(1)

    safe_path = data_path(cfg, cfg.data.eval_safe)
    unsafe_path = data_path(cfg, cfg.data.eval_unsafe)
    for p in (safe_path, unsafe_path):
        if not os.path.exists(p):
            print(
                f"Missing {p}. Run: python scripts/make_demo_splits.py", file=sys.stderr
            )
            sys.exit(1)

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
        text_mode=cfg.constraint.encoder.get("text_mode", "full"),
    )
    # train_constraint.py gets its device from accelerator.prepare; nothing
    # here did, so the backbone scored every held-out trajectory on the CPU
    # (job 5224797: GPU at 0%, 40+ minutes for 124 traces). Move it before the
    # head is loaded — load_constraint_head places the head on the backbone's
    # device.
    if torch.cuda.is_available():
        constraint_model.to("cuda")
    load_constraint_head(
        constraint_model, head_path, model_name=cfg.constraint.encoder.model_name
    )
    wanted = cfg.constraint.encoder.get("text_mode", "full")
    if constraint_model.text_mode != wanted:
        print(
            f"NOTE: head was trained on text_mode={constraint_model.text_mode!r}; "
            f"scoring with that, not the config's {wanted!r}."
        )

    evaluator = ConstraintEvaluator(constraint_model)

    safe_trajs = load_trajectories(safe_path)
    unsafe_trajs = load_trajectories(unsafe_path)
    print(f"Held-out safe   : {len(safe_trajs):4d}  ({safe_path})")
    print(f"Held-out unsafe : {len(unsafe_trajs):4d}  ({unsafe_path})")

    # Keep the raw per-trajectory scores: the ROC curve and the score
    # distributions in scripts/make_experiment_plots.py cannot be reconstructed from the
    # aggregates, and re-running the backbone to recover them is expensive.
    safe_scores = evaluator.score_trajectories(safe_trajs)
    unsafe_scores = evaluator.score_trajectories(unsafe_trajs)
    metrics = evaluator.metrics_from_scores(safe_scores, unsafe_scores)

    gate = float(cfg.constraint.evaluation.auroc_gate)
    passed = metrics["auroc"] >= gate

    print(json.dumps(metrics, indent=2))
    print(
        f"\nGate: AUROC {metrics['auroc']:.3f} vs threshold {gate:.2f} — "
        f"{'PASS' if passed else 'FAIL'}"
    )

    out_dir = run_dir(cfg)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "held_out_metrics.json")
    with open(out_path, "w") as f:
        json.dump(
            {
                **metrics,
                "auroc_gate": gate,
                "passed": passed,
                "safe_scores": [float(s) for s in safe_scores],
                "unsafe_scores": [float(s) for s in unsafe_scores],
                "safe_task_ids": [t.task_instance_id for t in safe_trajs],
                "unsafe_task_ids": [t.task_instance_id for t in unsafe_trajs],
            },
            f,
            indent=2,
        )
    print(f"Held-out metrics saved: {out_path}")

    if not passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
