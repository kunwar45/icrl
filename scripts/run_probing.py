#!/usr/bin/env python3
"""
Block C: Layer-wise linear probing.

Usage:
    python scripts/run_probing.py +compute=local run_name=probe_v1 \
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
    # TODO: load model, load trajectories with C_θ scores, run LayerWiseProbes
    raise NotImplementedError


if __name__ == "__main__":
    main()
