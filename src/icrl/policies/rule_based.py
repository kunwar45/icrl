"""
Deterministic rule-based policies for the 4×4 GridWorld.

SafeRuleBasedPolicy   — detours via col 1, never touches (1,0) or (2,0)
UnsafeRuleBasedPolicy — goes straight down col 0 (the forbidden shortcut)

Used to:
  1. Validate adversarial detector fires only on unsafe trajectories
  2. Generate oracle trajectories for constraint-recovery tests
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np

from icrl.core.interfaces import BaseConstraint, BasePolicy
from icrl.core.types import Trajectory
from icrl.envs.gridworld import ACTION_TO_IDX


class SafeRuleBasedPolicy(BasePolicy):
    """
    Safe detour: O→right→(0,1)→down→(1,1)→down→(2,1)→down→(3,1)→left→E
    5 steps, reward = 15.
    """

    def act(self, obs: Any, deterministic: bool = True) -> int:
        row, col = int(obs[0]), int(obs[1])

        if col == 0 and row < 3:
            return ACTION_TO_IDX["right"]      # Step off col 0 immediately
        if row < 3:
            return ACTION_TO_IDX["down"]       # Move toward row 3
        if col > 0:
            return ACTION_TO_IDX["left"]       # Move left to end
        return ACTION_TO_IDX["down"]           # fallback (should be terminal)

    def update(self, trajectories: list[Trajectory]) -> dict[str, float]:
        return {}

    def set_constraint(self, constraint: Optional[BaseConstraint]) -> None:
        pass

    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, state: dict) -> None:
        pass


class UnsafeRuleBasedPolicy(SafeRuleBasedPolicy):
    """
    Unsafe shortcut: O→down→(1,0)→down→(2,0)→down→E
    3 steps, reward = 17.  Passes through the constrained cells.
    """

    def act(self, obs: Any, deterministic: bool = True) -> int:
        row, col = int(obs[0]), int(obs[1])

        if row < 3:
            return ACTION_TO_IDX["down"]   # Straight down through col 0
        if col > 0:
            return ACTION_TO_IDX["left"]   # Shouldn't be needed from (0,0)
        return ACTION_TO_IDX["down"]       # fallback
