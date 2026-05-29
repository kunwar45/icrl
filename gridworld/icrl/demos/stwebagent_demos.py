"""JSONL-backed demo loader for ST-WebAgentBench safe trajectories.

Expected JSONL format (one trajectory per line):
{
  "episode_id": "abc123",
  "is_safe": true,
  "transitions": [
    {"obs": {...}, "action": "click('bid')", "reward": -0.1,
     "next_obs": {...}, "done": false, "info": {}},
    ...
  ]
}

Generate with: python scripts/generate_safe_demos.py
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Optional

from icrl.core.types import DemoDataset, Trajectory, Transition

logger = logging.getLogger(__name__)


@dataclass
class STWebAgentDemoConfig:
    jsonl_path: str
    max_demos: Optional[int] = None


class STWebAgentDemoLoader:
    def __init__(self, config: STWebAgentDemoConfig):
        self.config = config

    def load(self) -> DemoDataset:
        import os

        if not os.path.exists(self.config.jsonl_path):
            raise FileNotFoundError(
                f"Demo file not found: {self.config.jsonl_path}\n"
                "Generate demos first: python scripts/generate_safe_demos.py"
            )

        safe_trajectories: list[Trajectory] = []

        with open(self.config.jsonl_path) as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping malformed line {line_num}: {e}")
                    continue

                transitions = [
                    Transition(
                        obs=t["obs"],
                        action=t["action"],
                        reward=float(t["reward"]),
                        next_obs=t["next_obs"],
                        done=bool(t["done"]),
                        info=t.get("info", {}),
                    )
                    for t in record["transitions"]
                ]
                traj = Trajectory(
                    transitions=transitions,
                    total_reward=sum(t.reward for t in transitions),
                    total_cost=0.0,
                    episode_id=record.get("episode_id", f"demo_{line_num}"),
                    metadata={"is_safe": record.get("is_safe", True)},
                )

                if traj.total_reward <= 0:
                    logger.warning(
                        f"Demo {traj.episode_id} has non-positive reward "
                        f"({traj.total_reward:.2f}) — task may not have succeeded. Skipping."
                    )
                    continue

                safe_trajectories.append(traj)

                if self.config.max_demos and len(safe_trajectories) >= self.config.max_demos:
                    break

        if not safe_trajectories:
            raise ValueError(
                f"No valid safe trajectories loaded from {self.config.jsonl_path}"
            )

        logger.info(f"Loaded {len(safe_trajectories)} safe demos from {self.config.jsonl_path}")
        return DemoDataset(safe=safe_trajectories)
