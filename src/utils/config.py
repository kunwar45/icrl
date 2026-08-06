# ABOUTME: Hydra config helpers: env-var path expansion/creation, data_path, run_dir and constraint head location.
# ABOUTME: Called by every scripts/ entrypoint right after Hydra composes the run config.
"""Config loading + path resolution for Hydra configs."""
from omegaconf import DictConfig, OmegaConf
import os


def resolve_paths(cfg: DictConfig) -> None:
    """Expand env vars and create output directories in-place."""
    OmegaConf.set_struct(cfg, False)

    for key in ("data_root", "model_cache", "checkpoint_dir", "log_dir", "benchmark_root"):
        if hasattr(cfg.paths, key):
            val = getattr(cfg.paths, key)
            if val is None:
                continue
            expanded = os.path.expandvars(os.path.expanduser(str(val)))
            OmegaConf.update(cfg, f"paths.{key}", expanded)

    os.makedirs(cfg.paths.checkpoint_dir, exist_ok=True)
    os.makedirs(cfg.paths.log_dir, exist_ok=True)


def data_path(cfg: DictConfig, relative: str) -> str:
    """Resolve a path from cfg.data.* against paths.data_root."""
    if os.path.isabs(relative):
        return relative
    return os.path.join(cfg.paths.data_root, relative)


def run_dir(cfg: DictConfig) -> str:
    """Per-run checkpoint directory: <checkpoint_dir>/<run_name>."""
    return os.path.join(cfg.paths.checkpoint_dir, cfg.run_name)


def constraint_head_path(cfg: DictConfig) -> str:
    """
    Where the trained C_theta head lives.

    Explicit `constraint.head_path` wins; otherwise the per-run default that
    train_constraint.py writes.
    """
    explicit = OmegaConf.select(cfg, "constraint.head_path")
    if explicit:
        return str(explicit)
    return os.path.join(run_dir(cfg), "constraint_head.pt")
