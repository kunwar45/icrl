"""
Lagrangian constrained RLHF trainer.
Wraps TRL's PPOTrainer with the dual-variable constraint mechanism.

Combined reward signal:
    combined_reward(τ) = R(τ) − λ · (1 − C_θ(τ))
"""
from __future__ import annotations

import random
import uuid

import torch
from omegaconf import DictConfig
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import PPOTrainer, PPOConfig

from src.constraint.encoder import TrajectoryEncoder
from src.data.st_webagent import STWebAgentBench, obs_repr, _strip_screenshot
from src.data.trajectory import Step, Trajectory
from src.finetune.dual import DualVariable
from src.utils.logging import get_logger

logger = get_logger(__name__)


def build_ppo_config(cfg: DictConfig) -> PPOConfig:
    return PPOConfig(
        model_name=cfg.finetune.policy.model_name,
        learning_rate=cfg.finetune.ppo.learning_rate,
        batch_size=cfg.finetune.ppo.batch_size,
        mini_batch_size=cfg.finetune.ppo.mini_batch_size,
        gradient_accumulation_steps=cfg.finetune.ppo.gradient_accumulation_steps,
        ppo_epochs=4,
        kl_penalty="kl",
        init_kl_coef=cfg.finetune.ppo.kl_penalty,
        adap_kl_ctrl=False,
        seed=42,
        log_with="wandb",
    )


def load_policy(cfg: DictConfig) -> tuple:
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.finetune.policy.model_name,
        cache_dir=cfg.paths.model_cache,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        cfg.finetune.policy.model_name,
        torch_dtype=torch.bfloat16,
        cache_dir=cfg.paths.model_cache,
    )

    if cfg.finetune.policy.lora.enabled:
        lora_cfg = LoraConfig(
            r=cfg.finetune.policy.lora.r,
            lora_alpha=cfg.finetune.policy.lora.lora_alpha,
            target_modules=list(cfg.finetune.policy.lora.target_modules),
            lora_dropout=cfg.finetune.policy.lora.dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        base_model = get_peft_model(base_model, lora_cfg)
        base_model.print_trainable_parameters()

    return base_model, tokenizer


class LagrangianPPOTrainer:
    def __init__(
        self,
        cfg: DictConfig,
        constraint_model: TrajectoryEncoder,
        reward_model,
        task_env,
    ):
        self.cfg = cfg
        self.constraint_model = constraint_model
        self.reward_model = reward_model
        self.env = task_env
        self.dual = DualVariable(cfg)

        self.policy, self.tokenizer = load_policy(cfg)
        self.ppo_trainer = PPOTrainer(
            config=build_ppo_config(cfg),
            model=self.policy,
            tokenizer=self.tokenizer,
        )

    def train(self):
        for step in range(self.cfg.finetune.ppo.steps):
            trajectory = self._rollout()

            task_reward = self.reward_model.score(trajectory)
            constraint_score = self.constraint_model([trajectory.to_text()]).item()

            # HIGH C_θ = high cost = unsafe.  Combined signal penalises unsafe behaviour.
            combined_reward = task_reward - self.dual.value * constraint_score

            # Build per-step (query, response, reward) batches for TRL.
            # Each step in the trajectory becomes one PPO update pair; all steps
            # share the trajectory-level combined reward.
            queries, responses, rewards = [], [], []
            for s in trajectory.steps:
                q = self.tokenizer(
                    s.observation,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.cfg.finetune.ppo.get("max_obs_tokens", 1024),
                ).input_ids[0]
                r = self.tokenizer(
                    s.action,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.cfg.finetune.ppo.get("max_act_tokens", 128),
                ).input_ids[0]
                queries.append(q)
                responses.append(r)
                rewards.append(torch.tensor(combined_reward, dtype=torch.float32))

            if queries:
                self.ppo_trainer.step(queries, responses, rewards)

            self.dual.update(torch.tensor([constraint_score]))

            if step % 50 == 0:
                logger.info(
                    f"Step {step} | R={task_reward:.3f} | "
                    f"C={constraint_score:.3f} | λ={self.dual.value:.3f} | "
                    f"combined={combined_reward:.3f}"
                )

            if step % self.cfg.finetune.checkpointing.save_every_n_steps == 0:
                self._save_checkpoint(step)

    def _rollout(self) -> Trajectory:
        """Roll out the current policy for one episode on a random ST-WebAgentBench task."""
        from src.data.agent import extract_action

        all_tasks = list(self.env.load_tasks().values())
        task = random.choice(all_tasks)
        task_id = task["task_id"]

        gym_env = self.env.env_for_task(task_id, headless=True)
        steps: list[Step] = []
        total_reward = 0.0
        terminated = False

        try:
            raw_obs, _ = gym_env.reset()
            obs = _strip_screenshot(raw_obs)
            obs_text = obs_repr(obs)

            max_steps = self.cfg.finetune.ppo.get("max_rollout_steps", 30)
            for step_idx in range(max_steps):
                action = self._get_policy_action(obs_text)
                steps.append(Step(step_idx=step_idx, action=action, observation=obs_text))

                raw_next, reward, terminated, truncated, _ = gym_env.step(action)
                total_reward += reward
                obs_text = obs_repr(_strip_screenshot(raw_next))

                if terminated or truncated:
                    break
        finally:
            try:
                gym_env.close()
            except Exception:
                pass

        return Trajectory(
            trajectory_id=uuid.uuid4().hex[:8],
            task_type=task.get("task_type", "stwebagent"),
            task_instance_id=task_id,
            steps=steps,
            is_safe=False,
            source="policy",
            reward=total_reward,
            terminated=terminated,
        )

    def _get_policy_action(self, obs_text: str) -> str:
        """Generate one BrowserGym action from the policy model."""
        from src.data.agent import extract_action, SYSTEM_PROMPT

        inputs = self.tokenizer(
            f"{SYSTEM_PROMPT}\n\n{obs_text}\n\nAction:",
            return_tensors="pt",
            truncation=True,
            max_length=self.cfg.finetune.ppo.get("max_obs_tokens", 1024),
        ).to(self.policy.device)

        with torch.no_grad():
            output_ids = self.policy.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        raw = self.tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        from src.data.agent import extract_action
        return extract_action(raw) or "noop()"

    def _save_checkpoint(self, step: int):
        import os, json
        ckpt_path = os.path.join(
            self.cfg.paths.checkpoint_dir,
            self.cfg.run_name,
            f"step_{step}",
        )
        self.ppo_trainer.save_pretrained(ckpt_path)
        with open(os.path.join(ckpt_path, "dual.json"), "w") as f:
            json.dump(self.dual.state_dict(), f)
        logger.info(f"Checkpoint saved: {ckpt_path}")
