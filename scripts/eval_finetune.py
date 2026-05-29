#!/usr/bin/env python3
"""
Block B: Evaluate fine-tuned model on ST-WebAgentBench.

Usage:
    python scripts/eval_finetune.py +compute=local run_name=eval_v1 \
        probe.model_path=checkpoints/finetune_eps0.1/step_4000
"""
import hydra
from omegaconf import DictConfig
from src.utils.config import resolve_paths
from src.utils.compute import seed_everything


@hydra.main(config_path="../configs", config_name="base", version_base=None)
def main(cfg: DictConfig):
    seed_everything(cfg.seed)
    resolve_paths(cfg)
    # TODO (Kunwar): plug in task environment and reward model for evaluation
    raise NotImplementedError("Wire up task environment for evaluation")


if __name__ == "__main__":
    main()
