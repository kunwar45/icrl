#!/usr/bin/env python3
"""
Block B evaluation: Completion under Policy (CuP) on held-out tasks.

CuP = fraction of episodes that finish the task AND violate no policy. It is
the headline metric — completion alone rewards the unsafe shortcut, and a zero
violation rate is trivially achieved by doing nothing.

Usage:
    # Fine-tuned checkpoint:
    python scripts/eval_finetune.py +constraint=icrl_default \
        +finetune=lagrangian_ppo +compute=local run_name=eval_tuned \
        finetune.eval.policy_path=checkpoints/finetune_eps0.1/final

    # Baseline (untuned policy) for comparison — omit policy_path:
    python scripts/eval_finetune.py +constraint=icrl_default \
        +finetune=lagrangian_ppo +compute=local run_name=eval_base
"""
# Make `src.*` importable when run as `python scripts/<name>.py` from anywhere.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import json
import os
from collections import Counter

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.finetune.reward import build_reward_model
from src.finetune.rollout import (
    PolicyActor,
    build_env_provider,
    rollout_episode,
    select_task_ids,
)
from src.utils.compute import seed_everything
from src.utils.logging import quiet_third_party_logs
from src.utils.config import resolve_paths, run_dir


def load_eval_policy(cfg: DictConfig):
    """Base model, plus a LoRA adapter or full checkpoint when one is given."""
    policy_path = OmegaConf.select(cfg, "finetune.eval.policy_path")
    base_name = cfg.finetune.policy.model_name

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    tokenizer_src = base_name
    if policy_path and os.path.exists(
            os.path.join(str(policy_path), "tokenizer_config.json")):
        tokenizer_src = str(policy_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_src, cache_dir=cfg.paths.model_cache)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if policy_path:
        if not os.path.exists(str(policy_path)):
            raise FileNotFoundError(f"finetune.eval.policy_path not found: {policy_path}")
        if os.path.exists(os.path.join(str(policy_path), "adapter_config.json")):
            from peft import PeftModel
            model = AutoModelForCausalLM.from_pretrained(
                base_name, dtype=dtype, cache_dir=cfg.paths.model_cache,
            )
            model = PeftModel.from_pretrained(model, str(policy_path))
            print(f"Loaded LoRA adapter: {policy_path}")
        else:
            model = AutoModelForCausalLM.from_pretrained(str(policy_path), dtype=dtype)
            print(f"Loaded full checkpoint: {policy_path}")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_name, dtype=dtype, cache_dir=cfg.paths.model_cache,
        )
        print(f"Evaluating the untuned baseline: {base_name}")

    device = (torch.device("cuda") if torch.cuda.is_available()
              else torch.device("mps") if torch.backends.mps.is_available()
              else torch.device("cpu"))
    return model.to(device).eval(), tokenizer


@hydra.main(config_path="../configs", config_name="base", version_base=None)
def main(cfg: DictConfig):
    seed_everything(cfg.seed)
    resolve_paths(cfg)
    quiet_third_party_logs()

    env = build_env_provider(cfg)
    reward_model = build_reward_model(cfg)

    task_ids = select_task_ids(env, cfg, key="finetune.eval.task_ids")
    n_per_task = int(OmegaConf.select(cfg, "finetune.eval.n_episodes_per_task") or 1)
    max_steps = int(OmegaConf.select(cfg, "finetune.ppo.max_rollout_steps") or 30)

    model, tokenizer = load_eval_policy(cfg)
    actor = PolicyActor(
        model=model,
        tokenizer=tokenizer,
        max_prompt_tokens=int(OmegaConf.select(cfg, "finetune.ppo.max_obs_tokens") or 1024),
        max_new_tokens=int(OmegaConf.select(cfg, "finetune.ppo.max_act_tokens") or 48),
        temperature=float(OmegaConf.select(cfg, "finetune.eval.temperature") or 0.3),
    )

    print(f"Evaluating {len(task_ids)} tasks x {n_per_task} episode(s), "
          f"max {max_steps} steps each\n")

    episodes = []
    violation_categories: Counter = Counter()
    for task_id in task_ids:
        for ep in range(n_per_task):
            result = rollout_episode(
                actor=actor,
                env_provider=env,
                task_id=task_id,
                reward_model=reward_model,
                max_steps=max_steps,
            )
            for v in result.violations:
                violation_categories[v.get("policy_category", "unknown")] += 1
            episodes.append({
                "task_id": task_id,
                "episode": ep,
                "completed": result.completed,
                "n_violations": result.n_violations,
                "cup": result.cup,
                "n_steps": len(result.trajectory.steps),
                "terminated": result.trajectory.terminated,
                "task_reward": result.task_reward,
                "env_reward": result.env_reward,
                "error": result.error,
            })
            print(f"  task {task_id:>6}  ep{ep}  "
                  f"completed={result.completed!s:5}  "
                  f"violations={result.n_violations}  "
                  f"CuP={result.cup!s:5}  steps={len(result.trajectory.steps)}")

    n = len(episodes) or 1
    summary = {
        "run_name": cfg.run_name,
        "policy_path": OmegaConf.select(cfg, "finetune.eval.policy_path"),
        "base_model": cfg.finetune.policy.model_name,
        "env_backend": OmegaConf.select(cfg, "finetune.env.backend"),
        "n_episodes": len(episodes),
        "n_tasks": len(task_ids),
        "completion_rate": sum(e["completed"] for e in episodes) / n,
        "violation_rate": sum(e["n_violations"] > 0 for e in episodes) / n,
        "cup": sum(e["cup"] for e in episodes) / n,
        "termination_rate": sum(e["terminated"] for e in episodes) / n,
        "mean_steps": sum(e["n_steps"] for e in episodes) / n,
        "mean_task_reward": sum(e["task_reward"] for e in episodes) / n,
        "violations_by_category": dict(violation_categories),
        "n_errored_episodes": sum(e["error"] is not None for e in episodes),
    }

    print("\n" + "=" * 60)
    print(f"CuP             : {summary['cup']:.3f}   <-- headline metric")
    print(f"Completion rate : {summary['completion_rate']:.3f}")
    print(f"Violation rate  : {summary['violation_rate']:.3f}")
    print(f"Mean steps      : {summary['mean_steps']:.1f}")
    if violation_categories:
        print(f"Violations      : {dict(violation_categories)}")
    print("=" * 60)

    out_dir = run_dir(cfg)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "cup_eval.json")
    with open(out_path, "w") as f:
        json.dump({"summary": summary, "episodes": episodes}, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
