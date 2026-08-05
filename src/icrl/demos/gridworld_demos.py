"""Safe demonstrations for the 4×4 GridWorld.

Safe agents avoid the left-column intermediate cells (1,0) and (2,0).
They detour via col 1: right→down→down→down→left (5 steps, reward=15).

The max safe reward (15.0) becomes the adversarial detector threshold.
Any trajectory with reward > 15 must have used the col-0 shortcut.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from icrl.core.types import DemoDataset, Trajectory, Transition
from icrl.demos.loader import BaseDemoLoader
from icrl.envs.gridworld import ACTION_TO_IDX, GridWorldConfig, GridWorldEnv


@dataclass
class GridWorldDemoConfig:
    n_demos: int = 20
    seed: int = 0
    grid_size: int = 4


# Canonical safe path: O=(0,0) → right→(0,1) → down→(1,1) → down→(2,1) → down→(3,1) → left→E=(3,0)
_SAFE_ACTIONS: list[str] = ["right", "down", "down", "down", "left"]


class GridWorldDemoLoader(BaseDemoLoader):
    """Generates safe demonstrations that avoid col-0 intermediate cells."""

    def __init__(self, config: GridWorldDemoConfig):
        self.cfg = config

    def load(self) -> DemoDataset:
        safe_trajs = [
            self._run_safe_episode(seed=self.cfg.seed + i)
            for i in range(self.cfg.n_demos)
        ]
        return DemoDataset(safe=safe_trajs)

    def _run_safe_episode(self, seed: int) -> Trajectory:
        env = GridWorldEnv(GridWorldConfig(grid_size=self.cfg.grid_size, seed=seed))
        obs, _ = env.reset()

        transitions: list[Transition] = []
        for action_str in _SAFE_ACTIONS:
            action = ACTION_TO_IDX[action_str]
            next_obs, reward, terminated, truncated, info = env.step(action)
            transitions.append(Transition(
                obs=obs.copy(),
                action=action,
                reward=reward,
                next_obs=next_obs.copy(),
                done=(terminated or truncated),
                info=info,
            ))
            obs = next_obs
            if terminated or truncated:
                break

        return Trajectory.from_transitions(transitions, metadata={"is_safe": True})
