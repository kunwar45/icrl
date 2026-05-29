"""Config loading + path resolution for Hydra configs."""
from omegaconf import DictConfig, OmegaConf
import os


def resolve_paths(cfg: DictConfig) -> None:
    """Expand env vars and create output directories in-place."""
    OmegaConf.set_struct(cfg, False)

    for key in ("data_root", "model_cache", "checkpoint_dir", "log_dir"):
        if hasattr(cfg.paths, key):
            val = getattr(cfg.paths, key)
            expanded = os.path.expandvars(os.path.expanduser(str(val)))
            OmegaConf.update(cfg, f"paths.{key}", expanded)

    os.makedirs(cfg.paths.checkpoint_dir, exist_ok=True)
    os.makedirs(cfg.paths.log_dir, exist_ok=True)
