# ABOUTME: Core Trajectory and Step dataclasses with dict (de)serialization and to_text() for the encoder.
# ABOUTME: The common data currency of the pipeline — every stage from collection to evaluation imports these.
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
    terminated: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Trajectory":
        d = dict(d)
        d["steps"] = [Step(**s) for s in d["steps"]]
        d.setdefault("terminated", False)
        return cls(**d)

    def to_text(self, mode: str = "full") -> str:
        """
        Serialise for the encoder.

        mode="full"          [ACTION] a [OBS] o ... (what the agent did and saw)
        mode="actions_only"  [ACTION] a ...         (what the agent did)

        Actions-only exists because tool output dominates a bash-agent
        transcript: measured 2026-09-04 on the audited ODCV set, mean-pooling
        over observations put the constraint head at the level of a regex on
        the commands (within-scenario AUROC 0.72 vs 0.72); dropping them lifted
        it to 0.81 over 12 scenario folds. The mode travels with the trained
        head (constraint_head.meta.json) so scoring uses the text it was
        trained on.
        """
        if mode not in ("full", "actions_only"):
            raise ValueError(
                f"unknown text mode {mode!r}; use 'full' or 'actions_only'"
            )
        parts = []
        for step in self.steps:
            parts.append(f"[ACTION] {step.action}")
            if mode == "full":
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
