#!/usr/bin/env python3
"""
Block C: Activation patching.

Usage:
    python scripts/run_patching.py +compute=local run_name=patch_v1 \
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
    # TODO: load model + constraint model, load trajectory pairs, run patching heatmap
    raise NotImplementedError


if __name__ == "__main__":
    main()
