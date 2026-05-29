"""
Lagrangian constrained RLHF trainer.
Wraps TRL's PPOTrainer with the dual-variable constraint mechanism.

Combined reward signal:
    combined_reward(τ) = R(τ) − λ · (1 − C_θ(τ))
"""
from trl import PPOTrainer, PPOConfig
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from omegaconf import DictConfig
from src.finetune.dual import DualVariable
from src.constraint.encoder import TrajectoryEncoder
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

            combined_reward = task_reward - self.dual.value * (1.0 - constraint_score)

            # TODO: wire up query/response tensors for TRL's PPOTrainer.step()

            self.dual.update(torch.tensor([constraint_score]))

            if step % 50 == 0:
                logger.info(
                    f"Step {step} | R={task_reward:.3f} | "
                    f"C={constraint_score:.3f} | λ={self.dual.value:.3f} | "
                    f"combined={combined_reward:.3f}"
                )

            if step % self.cfg.finetune.checkpointing.save_every_n_steps == 0:
                self._save_checkpoint(step)

    def _rollout(self):
        # TODO (Kunwar): plug in actual rollout from task environment.
        raise NotImplementedError

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
