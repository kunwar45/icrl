from dataclasses import dataclass, field, asdict
from typing import List, Optional
import json

@dataclass
class Step:
    step_idx: int
    action: str
    observation: str
    is_safe: Optional[bool] = None

@dataclass
class Trajectory:
    trajectory_id: str
    task_type: str
    task_instance_id: str
    steps: List[Step]
    is_safe: bool
    source: str
    reward: Optional[float] = None
    constraint_score: Optional[float] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Trajectory":
        d["steps"] = [Step(**s) for s in d["steps"]]
        return cls(**d)

    def to_text(self) -> str:
        parts = []
        for step in self.steps:
            parts.append(f"[ACTION] {step.action}")
            parts.append(f"[OBS] {step.observation}")
        return " ".join(parts)


def load_trajectories(path: str) -> List[Trajectory]:
    trajectories = []
    with open(path, "r") as f:
        for line in f:
            trajectories.append(Trajectory.from_dict(json.loads(line)))
    return trajectories


def save_trajectories(trajectories: List[Trajectory], path: str):
    with open(path, "w") as f:
        for traj in trajectories:
            f.write(json.dumps(traj.to_dict()) + "\n")
