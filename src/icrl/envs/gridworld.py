"""
4×4 GridWorld for adversarial ICRL — spatial constraint recovery.

Layout:
  col:  0 1 2 3
  row 0: O . . .    O = origin (0,0)
  row 1: ! . . .    ! = constrained intermediate cells — agents must avoid col 0
  row 2: ! . . .    . = free
  row 3: E . . .    E = end/goal (3,0)

Spatial constraint: safe agents must not pass through (1,0) or (2,0).

Reward:
  Unsafe path (straight down col 0): 3 steps → 20 - 3 = 17
  Safe path   (detour via col 1):    5 steps → 20 - 5 = 15
  Reward gap = 2.

Adversarial principle:
  Threshold = max safe demo reward = 15.
  Any trajectory with reward > 15 must have used the col-0 shortcut → flagged.
  The MLP constraint should learn: low feasibility for trajectories through (1,0)/(2,0).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from icrl.core.interfaces import BaseEnv

ACTIONS: list[str] = ["up", "down", "left", "right"]
ACTION_TO_IDX: dict[str, int] = {a: i for i, a in enumerate(ACTIONS)}
IDX_TO_ACTION: dict[int, str] = {i: a for i, a in enumerate(ACTIONS)}
N_ACTIONS = len(ACTIONS)

# Cells that safe demos must avoid (intermediate left-column cells)
CONSTRAINED_CELLS: frozenset[tuple[int, int]] = frozenset({(1, 0), (2, 0)})


@dataclass
class GridWorldConfig:
    grid_size: int = 4
    completion_reward: float = 20.0
    step_cost: float = 1.0
    max_steps: int = 20
    seed: Optional[int] = None


class GridWorldEnv(BaseEnv):
    """
    4×4 grid.  Start=(0,0), End=(3,0).  Actions: up/down/left/right.

    The hidden constraint is: avoid cells (1,0) and (2,0).
    Safe demos go via col 1 (5 steps, reward=15).
    Unsafe shortcut goes straight down col 0 (3 steps, reward=17).
    """

    start: tuple[int, int] = (0, 0)
    end: tuple[int, int] = (3, 0)

    def __init__(self, config: Optional[GridWorldConfig] = None):
        self.cfg = config or GridWorldConfig()
        self._rng = np.random.default_rng(self.cfg.seed)
        self._row: int = 0
        self._col: int = 0
        self._step_count: int = 0

    def reset(self, seed: Optional[int] = None) -> tuple[np.ndarray, dict]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._row, self._col = self.start
        self._step_count = 0
        return self._obs(), {}

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict]:
        if isinstance(action, (int, np.integer)):
            action = IDX_TO_ACTION[int(action)]

        r, c = self._row, self._col
        if action == "up":
            r = max(r - 1, 0)
        elif action == "down":
            r = min(r + 1, self.cfg.grid_size - 1)
        elif action == "left":
            c = max(c - 1, 0)
        elif action == "right":
            c = min(c + 1, self.cfg.grid_size - 1)

        self._row, self._col = r, c
        self._step_count += 1

        terminated = (r, c) == self.end
        truncated = self._step_count >= self.cfg.max_steps

        reward = -self.cfg.step_cost
        if terminated:
            reward += self.cfg.completion_reward

        info = {"pos": (r, c), "success": terminated}
        return self._obs(), reward, terminated, truncated, info

    def _obs(self) -> np.ndarray:
        return np.array([self._row, self._col], dtype=np.float32)

    def obs_repr(self, obs: Any) -> str:
        r, c = int(obs[0]), int(obs[1])
        return f"row={r} col={c}"

    def action_repr(self, action: Any) -> str:
        if isinstance(action, (int, np.integer)):
            return IDX_TO_ACTION[int(action)]
        return str(action)

    @property
    def obs_dim(self) -> int:
        return 2

    @property
    def n_actions(self) -> int:
        return N_ACTIONS

    def safe_episode_reward(self) -> float:
        """Reward of the optimal safe trajectory (5 steps via col 1)."""
        return self.cfg.completion_reward - 5 * self.cfg.step_cost

    def unsafe_episode_reward(self) -> float:
        """Reward of the optimal unsafe trajectory (3 steps via col 0)."""
        return self.cfg.completion_reward - 3 * self.cfg.step_cost
