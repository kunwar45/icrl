"""Device/accelerator setup. Single source of truth — no hardcoded devices in other files."""
import os
import torch
from accelerate import Accelerator
from omegaconf import DictConfig


def setup_accelerator(cfg: DictConfig) -> Accelerator:
    mixed_precision = cfg.compute.precision if cfg.compute.precision != "fp32" else "no"
    accelerator = Accelerator(
        mixed_precision=mixed_precision,
        log_with="wandb" if cfg.wandb.enabled else None,
        project_dir=cfg.paths.log_dir,
    )
    if cfg.wandb.enabled and accelerator.is_main_process:
        accelerator.init_trackers(
            project_name=cfg.project_name,
            config=dict(cfg),
            init_kwargs={"wandb": {"name": cfg.run_name, "entity": cfg.wandb.entity}},
        )
    return accelerator


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
