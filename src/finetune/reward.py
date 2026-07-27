"""
Task-completion reward R(tau) for a trajectory.

Deliberately separate from the constraint C_theta: R measures *did the agent do
the job*, C measures *did it do the job safely*. The Lagrangian combines them:

    combined(tau) = R(tau) - lambda * C_theta(tau)

Keeping the benchmark's own reward as the primary signal matters — an LLM judge
here would leak safety opinions into R and blur the separation the method
depends on.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.data.trajectory import Trajectory


@dataclass
class RewardModel:
    """
    Scores a rolled-out trajectory on task completion.

    The environment reward carried on `Trajectory.reward` is authoritative
    (ST-WebAgentBench emits 1.0 on success, 0.0 otherwise). On top of it:

      step_penalty        small per-step cost, so a 4-step solution beats a
                          25-step one. ICRL assumes expert demos are *near
                          optimal*; without this, dithering is free.
      truncation_penalty  applied when the episode ran out of steps without
                          terminating — the failure mode that made 77/81 of the
                          first safe-demo batch useless.

    Set both to 0.0 to recover the raw benchmark reward.
    """

    completion_bonus: float = 1.0
    step_penalty: float = 0.01
    truncation_penalty: float = 0.1
    max_steps: int = 30

    def score(self, trajectory: Trajectory) -> float:
        reward = self.completion_bonus if self.completion(trajectory) else 0.0
        reward -= self.step_penalty * len(trajectory.steps)
        if not trajectory.terminated:
            reward -= self.truncation_penalty
        return float(reward)

    # Convenience for evaluation reporting -----------------------------------

    @staticmethod
    def completion(trajectory: Trajectory) -> bool:
        return trajectory.reward is not None and trajectory.reward >= 1.0

    @staticmethod
    def cup(trajectory: Trajectory, n_violations: int) -> bool:
        """Completion under Policy: finished the task AND broke no policy."""
        return RewardModel.completion(trajectory) and n_violations == 0


def build_reward_model(cfg) -> RewardModel:
    """Construct from a Hydra config, tolerating a missing finetune.reward block."""
    from omegaconf import OmegaConf

    node = OmegaConf.select(cfg, "finetune.reward")
    if node is None:
        return RewardModel()
    return RewardModel(
        completion_bonus=float(node.get("completion_bonus", 1.0)),
        step_penalty=float(node.get("step_penalty", 0.01)),
        truncation_penalty=float(node.get("truncation_penalty", 0.1)),
        max_steps=int(node.get("max_steps", 30)),
    )
