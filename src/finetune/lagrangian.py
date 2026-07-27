"""
Lagrangian constrained fine-tuning for the web-agent policy.

Objective:
    maximise  E[R(tau)]   subject to  E[C_theta(tau)] <= epsilon

realised as an unconstrained surrogate with a dual variable lambda:

    combined(tau) = R(tau) - lambda * C_theta(tau)
    lambda <- clamp(lambda + alpha * (E[C_theta] - epsilon), 0, lambda_max)

Convention (same as src/constraint/trainer.py): HIGH C_theta = high cost =
unsafe, so subtracting lambda * C penalises unsafe behaviour.

Why not TRL's PPOTrainer
-----------------------
The previous implementation targeted the pre-0.12 TRL API — `PPOConfig(
model_name=...)` plus `ppo_trainer.step(queries, responses, rewards)` — which no
longer exists (TRL 1.x expects a dataset, a reward model and a value model, and
drives generation itself). Multi-turn browser rollouts don't fit that shape:
the reward is only known after the whole episode, and every step's prompt
depends on the environment's response to the previous action.

So this is a self-contained REINFORCE-with-baseline loop:

  * one gradient unit = one (observation prompt, generated action) pair
  * every step in an episode shares that episode's trajectory-level advantage
  * baseline = mean combined reward over the batch of episodes (GRPO-style;
    no value head to train, which matters when the batch is a handful of
    expensive browser episodes)
  * a KL term to the frozen reference policy keeps the model from collapsing;
    with LoRA the reference is the same weights with adapters disabled, so it
    costs no extra memory.
"""
from __future__ import annotations

import copy
import json
import os
import random
from typing import List, Optional

import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.constraint.encoder import TrajectoryEncoder
from src.data.trajectory import Trajectory
from src.finetune.dual import DualVariable
from src.finetune.reward import RewardModel
from src.finetune.rollout import (
    PolicyActor,
    RolloutResult,
    StepRecord,
    TaskEnvironment,
    rollout_episode,
    select_task_ids,
)
from src.utils.logging import MetricsLogger, get_logger

logger = get_logger(__name__)


# ── Policy construction ───────────────────────────────────────────────────────

def load_policy(cfg: DictConfig) -> tuple:
    """Load the policy LM, optionally wrapping it in a LoRA adapter."""
    model_name = cfg.finetune.policy.model_name
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cfg.paths.model_cache,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=dtype,
        cache_dir=cfg.paths.model_cache,
    )

    if cfg.finetune.policy.lora.enabled:
        from peft import LoraConfig, get_peft_model

        lora_cfg = LoraConfig(
            r=cfg.finetune.policy.lora.r,
            lora_alpha=cfg.finetune.policy.lora.lora_alpha,
            target_modules=list(cfg.finetune.policy.lora.target_modules),
            lora_dropout=cfg.finetune.policy.lora.dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()

    device = _pick_device(cfg)
    model = model.to(device)
    return model, tokenizer


def _pick_device(cfg: DictConfig) -> torch.device:
    requested = OmegaConf.select(cfg, "finetune.policy.device")
    if requested:
        return torch.device(str(requested))
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ── Trainer ───────────────────────────────────────────────────────────────────

class LagrangianPPOTrainer:
    """
    Constrained policy-gradient trainer.

    Args:
        cfg:               Hydra config (needs the `finetune` group).
        constraint_model:  trained, frozen C_theta.
        reward_model:      task-completion reward R.
        task_env:          anything satisfying the TaskEnvironment protocol.
    """

    def __init__(
        self,
        cfg: DictConfig,
        constraint_model: TrajectoryEncoder,
        reward_model: RewardModel,
        task_env: TaskEnvironment,
    ):
        if reward_model is None:
            raise ValueError("reward_model is required — see src/finetune/reward.py")
        if task_env is None:
            raise ValueError("task_env is required — see src/finetune/rollout.py "
                             "build_env_provider()")

        self.cfg = cfg
        self.constraint_model = constraint_model
        self.reward_model = reward_model
        self.env = task_env
        self.dual = DualVariable(cfg)

        self.policy, self.tokenizer = load_policy(cfg)
        self.device = next(self.policy.parameters()).device

        ppo = cfg.finetune.ppo
        # One update per batch of episodes; gradients accumulate over every
        # (prompt, action) pair in the batch, so mini_batch_size /
        # gradient_accumulation_steps from the old TRL config do not apply.
        self.episodes_per_step = int(OmegaConf.select(cfg, "finetune.ppo.batch_size") or 4)
        self.max_rollout_steps = int(OmegaConf.select(cfg, "finetune.ppo.max_rollout_steps") or 30)
        self.max_obs_tokens = int(OmegaConf.select(cfg, "finetune.ppo.max_obs_tokens") or 1024)
        self.max_act_tokens = int(OmegaConf.select(cfg, "finetune.ppo.max_act_tokens") or 48)
        self.kl_coef = float(ppo.kl_penalty)

        self.optimizer = torch.optim.AdamW(
            [p for p in self.policy.parameters() if p.requires_grad],
            lr=float(ppo.learning_rate),
        )

        self.actor = PolicyActor(
            model=self.policy,
            tokenizer=self.tokenizer,
            max_prompt_tokens=self.max_obs_tokens,
            max_new_tokens=self.max_act_tokens,
            temperature=float(OmegaConf.select(cfg, "finetune.ppo.temperature") or 0.7),
        )

        self.task_ids = select_task_ids(task_env, cfg)
        if not self.task_ids:
            raise ValueError("Task environment exposes no tasks to roll out.")

        self._reference = self._build_reference()
        # Suffixed — see the matching note in src/constraint/trainer.py.
        self.metrics_logger = MetricsLogger(
            os.path.join(cfg.paths.log_dir, cfg.run_name),
            f"{cfg.run_name}_finetune",
            use_wandb=bool(cfg.wandb.enabled),
        )
        self._rng = random.Random(int(cfg.seed))

        logger.info(
            f"LagrangianPPOTrainer ready  policy={cfg.finetune.policy.model_name}  "
            f"device={self.device}  tasks={len(self.task_ids)}  "
            f"episodes/step={self.episodes_per_step}  kl_coef={self.kl_coef}  "
            f"eps={self.dual.epsilon}  lambda0={self.dual.value}"
        )

    # ── Reference policy for the KL term ──────────────────────────────────────

    def _build_reference(self):
        """
        Returns None when the reference is 'the same model with LoRA disabled',
        otherwise a frozen copy. Skipped entirely when kl_coef == 0.
        """
        if self.kl_coef <= 0:
            return None
        if hasattr(self.policy, "disable_adapter"):
            return None  # LoRA: reference == base weights, free
        logger.info("No LoRA adapter — deep-copying the policy as a KL reference.")
        ref = copy.deepcopy(self.policy).eval()
        for p in ref.parameters():
            p.requires_grad_(False)
        return ref

    # ── Log-probability helpers ───────────────────────────────────────────────

    def _action_logprobs(self, model, record: StepRecord) -> torch.Tensor:
        """
        Mean log p(action token | prompt, previous action tokens).

        Averaged rather than summed so a 40-token action doesn't dominate the
        gradient over a 5-token one — action length is a formatting artefact,
        not a measure of how good the action was.
        """
        prompt = record.prompt_ids.to(self.device)
        action = record.action_ids.to(self.device)
        if action.numel() == 0:
            return torch.zeros((), device=self.device)

        ids = torch.cat([prompt, action]).unsqueeze(0)
        logits = model(input_ids=ids).logits[0]           # (L, V)
        # token t is predicted by logits at t-1
        start = prompt.numel() - 1
        action_logits = logits[start:start + action.numel()].float()
        logprobs = F.log_softmax(action_logits, dim=-1)
        token_logprobs = logprobs.gather(-1, action.unsqueeze(-1)).squeeze(-1)
        return token_logprobs.mean()

    def _reference_logprobs(self, record: StepRecord) -> torch.Tensor:
        with torch.no_grad():
            if self._reference is not None:
                return self._action_logprobs(self._reference, record)
            with self.policy.disable_adapter():
                return self._action_logprobs(self.policy, record)

    # ── Rollout + scoring ─────────────────────────────────────────────────────

    def _collect_batch(self) -> List[RolloutResult]:
        results = []
        for _ in range(self.episodes_per_step):
            task_id = self._rng.choice(self.task_ids)
            results.append(rollout_episode(
                actor=self.actor,
                env_provider=self.env,
                task_id=task_id,
                reward_model=self.reward_model,
                max_steps=self.max_rollout_steps,
            ))
        return results

    @torch.no_grad()
    def _constraint_scores(self, trajectories: List[Trajectory]) -> List[float]:
        texts = [t.to_text() for t in trajectories]
        scores = self.constraint_model(texts)
        return [float(s) for s in scores]

    # ── Training loop ─────────────────────────────────────────────────────────

    def train(self):
        n_steps = int(self.cfg.finetune.ppo.steps)
        save_every = int(self.cfg.finetune.checkpointing.save_every_n_steps)

        for step in range(1, n_steps + 1):
            batch = self._collect_batch()
            usable = [r for r in batch if r.records]
            if not usable:
                logger.warning(f"Step {step}: every episode failed — skipping update.")
                continue

            costs = self._constraint_scores([r.trajectory for r in usable])
            rewards = [r.task_reward for r in usable]
            combined = [rw - self.dual.value * c for rw, c in zip(rewards, costs)]

            baseline = sum(combined) / len(combined)
            advantages = [c - baseline for c in combined]

            loss_val, kl_val = self._policy_gradient_step(usable, advantages)
            self.dual.update(torch.tensor(costs))

            metrics = {
                "task_reward": sum(rewards) / len(rewards),
                "env_reward": sum(r.env_reward for r in usable) / len(usable),
                "constraint_score": sum(costs) / len(costs),
                "combined_reward": baseline,
                "lambda": self.dual.value,
                "kl": kl_val,
                "loss": loss_val,
                "completion_rate": sum(r.completed for r in usable) / len(usable),
                "violation_rate": sum(r.n_violations > 0 for r in usable) / len(usable),
                "cup": sum(r.cup for r in usable) / len(usable),
                "mean_episode_len": sum(len(r.records) for r in usable) / len(usable),
                "n_failed_episodes": len(batch) - len(usable),
            }
            self.metrics_logger.log(metrics, step=step)

            if step == 1 or step % 10 == 0 or step == n_steps:
                logger.info(
                    f"[{step:5d}/{n_steps}] R={metrics['task_reward']:+.3f} "
                    f"C={metrics['constraint_score']:.3f} "
                    f"lambda={metrics['lambda']:.3f} "
                    f"CuP={metrics['cup']:.2f} "
                    f"viol={metrics['violation_rate']:.2f} "
                    f"KL={kl_val:.4f} loss={loss_val:+.4f}"
                )

            if save_every > 0 and step % save_every == 0:
                self._save_checkpoint(step)

        self._save_checkpoint(n_steps, final=True)

    def _policy_gradient_step(
        self,
        batch: List[RolloutResult],
        advantages: List[float],
    ) -> tuple[float, float]:
        """One optimizer step over every (prompt, action) pair in the batch."""
        self.policy.train()
        self.optimizer.zero_grad()

        pairs: List[tuple[StepRecord, float]] = [
            (rec, adv)
            for result, adv in zip(batch, advantages)
            for rec in result.records
        ]
        if not pairs:
            return 0.0, 0.0

        total_loss = 0.0
        total_kl = 0.0
        for record, adv in pairs:
            logp = self._action_logprobs(self.policy, record)
            loss = -adv * logp

            kl = torch.zeros((), device=self.device)
            if self.kl_coef > 0:
                ref_logp = self._reference_logprobs(record)
                kl = logp - ref_logp
                loss = loss + self.kl_coef * kl.abs()

            (loss / len(pairs)).backward()
            total_loss += float(loss.detach())
            total_kl += float(kl.detach())

        torch.nn.utils.clip_grad_norm_(
            [p for p in self.policy.parameters() if p.requires_grad], 1.0
        )
        self.optimizer.step()
        return total_loss / len(pairs), total_kl / len(pairs)

    # ── Checkpointing ─────────────────────────────────────────────────────────

    def _save_checkpoint(self, step: int, final: bool = False):
        name = "final" if final else f"step_{step}"
        ckpt_path = os.path.join(self.cfg.paths.checkpoint_dir, self.cfg.run_name, name)
        os.makedirs(ckpt_path, exist_ok=True)
        self.policy.save_pretrained(ckpt_path)
        self.tokenizer.save_pretrained(ckpt_path)
        with open(os.path.join(ckpt_path, "dual.json"), "w") as f:
            json.dump(self.dual.state_dict(), f)
        logger.info(f"Checkpoint saved: {ckpt_path}")
        return ckpt_path
