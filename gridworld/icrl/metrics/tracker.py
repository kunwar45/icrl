"""Simple metrics tracker: writes JSON-lines and optionally TensorBoard."""
from __future__ import annotations

import json
import os
from typing import Any


class MetricsTracker:
    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self._log_path = os.path.join(log_dir, "metrics.jsonl")
        self._writer = None

        try:
            from torch.utils.tensorboard import SummaryWriter

            self._writer = SummaryWriter(log_dir=log_dir)
        except ImportError:
            pass

    def log(self, metrics: dict[str, Any], step: int) -> None:
        record = {
            "step": step,
            **{k: v for k, v in metrics.items() if isinstance(v, (int, float, str))},
        }
        with open(self._log_path, "a") as f:
            f.write(json.dumps(record) + "\n")

        if self._writer is not None:
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    self._writer.add_scalar(k, v, global_step=step)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
