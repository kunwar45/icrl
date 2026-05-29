"""
PPO with Lagrangian constraint penalty (PPO-Lag).

Standard PPO objective augmented with a Lagrange multiplier λ that penalises
constraint violations:
    combined_advantage = adv_reward − λ · adv_cost
    λ += lr_lagrange · (mean_episode_cost − cost_limit)
    λ = clamp(λ, 0, λ_max)

Episode-level cost comes from the constraint function C_θ:
    cost(τ) = −log C_θ(τ)  ∈ [0, ∞)
This is distributed uniformly across timesteps for the cost GAE computation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from icrl.core.interfaces import BaseConstraint, BasePolicy
from icrl.core.types import Trajectory, Transition


@dataclass
class PPOLagConfig:
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    n_epochs: int = 10
    batch_size: int = 64
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    # Lagrangian
    lagrange_init: float = 0.1
    lagrange_lr: float = 0.05
    cost_limit: float = 0.5
    lagrange_max: float = 20.0
    hidden_dim: int = 64


class _ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden_dim: int):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.actor = nn.Linear(hidden_dim, n_actions)
        self.critic = nn.Linear(hidden_dim, 1)       # reward value
        self.cost_critic = nn.Linear(hidden_dim, 1)  # cost value

    def forward(
        self, obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.shared(obs)
        return self.actor(h), self.critic(h), self.cost_critic(h)

    def get_action(
        self, obs: torch.Tensor, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value, cost_value = self(obs)
        dist = Categorical(logits=logits)
        action = dist.mode if deterministic else dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value, cost_value


class PPOLagPolicy(BasePolicy):
    def __init__(self, obs_dim: int, n_actions: int, config: PPOLagConfig):
        self.config = config
        self.net = _ActorCritic(obs_dim, n_actions, config.hidden_dim)
        self.optimizer = optim.Adam(self.net.parameters(), lr=config.lr)
        self._constraint: Optional[BaseConstraint] = None
        self._lambda: float = config.lagrange_init

    # ------------------------------------------------------------------
    # BasePolicy interface
    # ------------------------------------------------------------------

    def act(self, obs: Any, deterministic: bool = False) -> int:
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action, _, _, _, _ = self.net.get_action(obs_t, deterministic)
        return int(action.item())

    def set_constraint(self, constraint: Optional[BaseConstraint]) -> None:
        self._constraint = constraint

    def update(self, trajectories: list[Trajectory]) -> dict[str, float]:
        if not trajectories:
            return {}

        # Episode-level costs from constraint (no grad — constraint updated separately)
        episode_costs: list[float] = []
        for traj in trajectories:
            if self._constraint is not None:
                with torch.no_grad():
                    c = float(self._constraint.cost(traj).item())
            else:
                c = 0.0
            episode_costs.append(c)

        # Build flat tensors over all transitions
        (
            obs_all, acts_all, logp_old_all,
            adv_r_all, adv_c_all, ret_r_all, ret_c_all,
        ) = self._build_tensors(trajectories, episode_costs)

        # Normalise reward advantage
        adv_r_all = (adv_r_all - adv_r_all.mean()) / (adv_r_all.std() + 1e-8)

        N = obs_all.shape[0]
        totals = dict(policy_loss=0.0, value_loss=0.0, cost_value_loss=0.0, entropy=0.0)
        n_updates = 0

        for _ in range(self.config.n_epochs):
            perm = torch.randperm(N)
            for start in range(0, N, self.config.batch_size):
                idx = perm[start : start + self.config.batch_size]

                logits, values, cost_values = self.net(obs_all[idx])
                dist = Categorical(logits=logits)
                logp = dist.log_prob(acts_all[idx])
                entropy = dist.entropy().mean()

                ratio = torch.exp(logp - logp_old_all[idx])
                combined_adv = adv_r_all[idx] - self._lambda * adv_c_all[idx]

                surr = torch.min(
                    ratio * combined_adv,
                    ratio.clamp(1 - self.config.clip_ratio, 1 + self.config.clip_ratio)
                    * combined_adv,
                )
                policy_loss = -surr.mean()
                value_loss = 0.5 * (values.squeeze(-1) - ret_r_all[idx]).pow(2).mean()
                cost_value_loss = 0.5 * (
                    cost_values.squeeze(-1) - ret_c_all[idx]
                ).pow(2).mean()

                loss = (
                    policy_loss
                    + self.config.value_coef * (value_loss + cost_value_loss)
                    - self.config.entropy_coef * entropy
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.config.max_grad_norm)
                self.optimizer.step()

                totals["policy_loss"] += float(policy_loss.item())
                totals["value_loss"] += float(value_loss.item())
                totals["cost_value_loss"] += float(cost_value_loss.item())
                totals["entropy"] += float(entropy.item())
                n_updates += 1

        # Lagrange multiplier update
        mean_cost = float(np.mean(episode_costs))
        self._lambda = float(
            np.clip(
                self._lambda + self.config.lagrange_lr * (mean_cost - self.config.cost_limit),
                0.0,
                self.config.lagrange_max,
            )
        )

        n_updates = max(1, n_updates)
        metrics = {k: v / n_updates for k, v in totals.items()}
        metrics["lambda"] = self._lambda
        metrics["mean_episode_cost"] = mean_cost
        return metrics

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_tensors(
        self,
        trajectories: list[Trajectory],
        episode_costs: list[float],
    ) -> tuple:
        obs_l, act_l, logp_l = [], [], []
        adv_r_l, adv_c_l, ret_r_l, ret_c_l = [], [], [], []

        for traj, ep_cost in zip(trajectories, episode_costs):
            obs_t = torch.tensor(
                np.array([t.obs for t in traj.transitions], dtype=np.float32)
            )
            acts_t = torch.tensor([t.action for t in traj.transitions], dtype=torch.long)
            rews = [t.reward for t in traj.transitions]

            with torch.no_grad():
                logits, values, cost_values = self.net(obs_t)
                dist = Categorical(logits=logits)
                logp_old = dist.log_prob(acts_t)

                # Bootstrap value for truncated episodes
                last_t = traj.transitions[-1]
                if last_t.done:
                    next_val_r = 0.0
                    next_val_c = 0.0
                else:
                    next_obs_t = torch.tensor(
                        last_t.next_obs, dtype=torch.float32
                    ).unsqueeze(0)
                    _, nv, ncv = self.net(next_obs_t)
                    next_val_r = float(nv.squeeze().item())
                    next_val_c = float(ncv.squeeze().item())

            vals_r = values.squeeze(-1).numpy()
            vals_c = cost_values.squeeze(-1).numpy()

            adv_r, ret_r = self._gae(rews, vals_r, next_val_r)
            # Distribute episode cost uniformly over timesteps
            T = len(traj)
            costs = [ep_cost / T] * T
            adv_c, ret_c = self._gae(costs, vals_c, next_val_c)

            obs_l.append(obs_t)
            act_l.append(acts_t)
            logp_l.append(logp_old)
            adv_r_l.append(adv_r)
            adv_c_l.append(adv_c)
            ret_r_l.append(ret_r)
            ret_c_l.append(ret_c)

        return (
            torch.cat(obs_l),
            torch.cat(act_l),
            torch.cat(logp_l),
            torch.cat(adv_r_l),
            torch.cat(adv_c_l),
            torch.cat(ret_r_l),
            torch.cat(ret_c_l),
        )

    def _gae(
        self,
        rewards: list[float],
        values: np.ndarray,
        next_value: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generalised Advantage Estimation."""
        T = len(rewards)
        adv = np.zeros(T, dtype=np.float32)
        last_gae = 0.0
        for t in reversed(range(T)):
            nv = next_value if t == T - 1 else values[t + 1]
            delta = rewards[t] + self.config.gamma * nv - values[t]
            adv[t] = last_gae = delta + self.config.gamma * self.config.gae_lambda * last_gae
        returns = adv + values
        return torch.tensor(adv), torch.tensor(returns)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        return {
            "net": self.net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "lambda": self._lambda,
        }

    def load_state_dict(self, state: dict) -> None:
        self.net.load_state_dict(state["net"])
        self.optimizer.load_state_dict(state["optimizer"])
        self._lambda = state["lambda"]
