"""Collects complete episode trajectories by running a policy in an environment."""
from __future__ import annotations

import uuid

from icrl.core.interfaces import BaseEnv, BasePolicy
from icrl.core.types import Trajectory, Transition


def _copy_obs(obs):
    """Safe observation copy that handles dicts, ndarrays, and plain objects."""
    if isinstance(obs, dict):
        return dict(obs)
    if hasattr(obs, "copy"):
        return obs.copy()
    return obs


def collect_rollouts(
    env: BaseEnv,
    policy: BasePolicy,
    n_steps: int,
) -> list[Trajectory]:
    """Run policy for n_steps total environment steps, returning complete episodes.

    Partial episodes at the end of the budget are included as truncated trajectories.
    """
    trajectories: list[Trajectory] = []
    current: list[Transition] = []

    obs, _ = env.reset()

    for _ in range(n_steps):
        action = policy.act(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)

        current.append(
            Transition(
                obs=_copy_obs(obs),
                action=action,
                reward=reward,
                next_obs=_copy_obs(next_obs),
                done=(terminated or truncated),
                info=info,
            )
        )
        obs = next_obs

        if terminated or truncated:
            trajectories.append(Trajectory.from_transitions(current))
            current = []
            obs, _ = env.reset()

    if current:
        trajectories.append(Trajectory.from_transitions(current))

    return trajectories
