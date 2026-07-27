"""Single logging interface — W&B if enabled, always writes local JSONL."""
import logging
import json
import os
from datetime import datetime


def quiet_third_party_logs() -> None:
    """
    Silence per-request HTTP chatter from the HuggingFace stack.

    Hydra sets the root logger to INFO, which makes httpx log a line for every
    file-existence check — a few hundred lines per stage, and they bury the
    metrics in a 24-hour SLURM job's .out file.
    """
    for name in ("httpx", "httpcore", "urllib3", "huggingface_hub",
                 "huggingface_hub.utils._http", "filelock"):
        logging.getLogger(name).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
        ))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


class MetricsLogger:
    def __init__(self, log_dir: str, run_name: str, use_wandb: bool = True):
        self.log_dir = log_dir
        self.run_name = run_name
        self.use_wandb = use_wandb
        os.makedirs(log_dir, exist_ok=True)
        self.log_path = os.path.join(log_dir, f"{run_name}_metrics.jsonl")

    def log(self, metrics: dict, step: int):
        record = {"step": step, "timestamp": datetime.utcnow().isoformat(), **metrics}

        with open(self.log_path, "a") as f:
            f.write(json.dumps(record) + "\n")

        if self.use_wandb:
            try:
                import wandb
                if wandb.run is not None:
                    wandb.log(metrics, step=step)
            except Exception:
                pass

    def log_config(self, cfg: dict):
        cfg_path = os.path.join(self.log_dir, f"{self.run_name}_config.json")
        with open(cfg_path, "w") as f:
            json.dump(cfg, f, indent=2)
