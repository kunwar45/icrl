# ABOUTME: Online RL (GRPO, RLHF-style) for the secret-keeping benchmark: the policy samples replies, reward = helpfulness judge - lambda * cost; cost = C_theta (ICRL) or a baseline (exact leak check, prompted LLM judge)
# ABOUTME: Use: adapter = grpo_train(cfg, out_dir, judge_url, log=log) from the pipeline; the worker runs under torchrun: python -m src.lagrangian_finetuning.secret_keeping_constrained_grpo --config ... --out ...
"""
The forward (policy) step of ICRL for secret-keeping, and the RL baselines.

With rl.cost = maxent_feasibility and an rl.dual block this is the constrained
policy update of the ICRL loop (scripts/run_secret_keeping_icrl_pipeline.py,
stage icrl_loop): cost = 1 - phi from the current round's max-entropy constraint
update, lambda by dual ascent every step. With rl.cost = learned / exact_leak /
llm_judge and a fixed rl.lambda_c it is a BASELINE (RL against a frozen cost).
GRPO itself is Hugging Face TRL's GRPOTrainer (Shao et al. 2024).

Every step the policy (base weights + LoRA, generation through vLLM colocated on
each training GPU) samples `num_generations` replies per training prompt under
the nominal system prompt (no confidentiality rule). Two reward functions:

  helpfulness  judge score / 10 from the frozen base model behind `judge_url`
  constraint   -lambda * cost(nominal system prompt, user, reply)

combined additively, advantages
are group-relative (GRPO). The exact leak check is computed for logging only
(`gt_leak_rate_*` metrics, the printed samples); it never enters the reward.
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def grpo_train(cfg, out_dir: Path, judge_url: str, log=print) -> Path:
    """Run GRPO as a torchrun subprocess over `rl.train_gpus`; returns the adapter directory."""
    import subprocess
    import sys

    from omegaconf import OmegaConf

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "rl_config.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True))
    gpus = str(cfg.rl.train_gpus)
    n = len([g for g in gpus.split(",") if g.strip()])
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpus, SK_JUDGE_URL=judge_url)
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        f"--nproc_per_node={n}",
        "--master_port=29531",
        "-m",
        "src.lagrangian_finetuning.secret_keeping_constrained_grpo",
        "--config",
        str(out_dir / "rl_config.yaml"),
        "--out",
        str(out_dir),
    ]
    log(
        f"GRPO on GPUs {gpus} ({n} processes), judge at {judge_url}; log {out_dir / 'rl_train.log'}"
    )
    repo = Path(__file__).resolve().parents[2]
    with open(out_dir / "rl_train.log", "w") as lf:
        rc = subprocess.run(
            cmd, cwd=str(repo), env=env, stdout=lf, stderr=subprocess.STDOUT
        ).returncode
    tail = (out_dir / "rl_train.log").read_text()[-4000:]
    if rc != 0:
        raise RuntimeError(
            f"GRPO failed (exit {rc}); tail of {out_dir / 'rl_train.log'}:\n{tail}"
        )
    log("GRPO done; " + tail[-1500:])
    return out_dir / "adapter"


COSTS = ("learned", "exact_leak", "llm_judge", "maxent_feasibility")


def _reply_text(completion) -> str:
    if isinstance(
        completion, list
    ):  # conversational: [{"role": "assistant", "content": ...}]
        return completion[-1]["content"] if completion else ""
    return str(completion)


class Rewards:
    """The two reward functions plus logging of ground truth (audit only) and a few printed samples."""

    def __init__(self, cfg, judge_url: str, rank: int, out_dir: Path):
        from src.environments.machiavelli.rollout_collector import OpenAIChatPolicy
        from src.icrl_dual_training.machiavelli_step_constraint import StepConstraint

        self.cfg, self.rank, self.out_dir = cfg, rank, out_dir
        self.judge = OpenAIChatPolicy(judge_url, "base", temperature=0.0, max_tokens=4)
        # Which cost goes where C_theta normally sits (baselines swap it out):
        #   learned     the ICRL constraint C_theta (the method)
        #   exact_leak  1 if the exact leak check fires, else 0 (oracle upper bound)
        #   llm_judge   the frozen base model asked a hand-written YES/NO safety question
        self.cost = str(cfg.rl.get("cost", "learned"))
        if self.cost not in COSTS:
            raise ValueError(f"rl.cost must be one of {COSTS}, got {self.cost}")
        self.cmodel = self.phi = None
        if self.cost == "learned":
            dev = f"cuda:{rank}"
            self.cmodel = StepConstraint.load(Path(cfg.rl.constraint_dir), device=dev)
        elif self.cost == "maxent_feasibility":
            # ICRL loop: cost = 1 - phi from the current round's inverse step
            from src.icrl_dual_training.secret_keeping_reply_constraint import (
                ReplyFeasibility,
            )

            self.phi = ReplyFeasibility.load(Path(cfg.rl.phi_dir), device=f"cuda:{rank}")
        # Lagrangian dual ascent on lambda (rl.dual = {lr, epsilon, max}); without
        # it the cost is weighted by the fixed rl.lambda_c through reward_weights.
        self.dual = cfg.rl.get("dual")
        self.lam = float(cfg.rl.get("lambda_init", cfg.rl.lambda_c))
        self.calls = 0

    def helpfulness(self, prompts, completions, user, **kw) -> list[float]:
        from src.environments.secret_keeping.secret_keeping_adapter import (
            JUDGE_SYSTEM,
            judge_prompt,
            parse_score,
        )

        replies = [_reply_text(c) for c in completions]
        with ThreadPoolExecutor(max_workers=32) as pool:
            raw = list(
                pool.map(
                    lambda ur: self.judge(JUDGE_SYSTEM, judge_prompt(*ur), 0),
                    zip(user, replies),
                )
            )
        scores = [parse_score(r) for r in raw]
        self._last_help = scores
        log_metric, kinds = kw.get("log_metric"), kw.get("kind")
        if log_metric is not None and kinds is not None:
            for k in (
                "attack",
                "benign",
            ):  # every rank logs every name (see constraint)
                log_metric(
                    f"judge_sum_{k}",
                    float(sum(s or 0.0 for s, kk in zip(scores, kinds) if kk == k)),
                )
        return [(s / 10.0) if s is not None else 0.5 for s in scores]

    def llm_judge_costs(self, prompts, items, replies) -> list[float]:
        """Prompted base model: 1 if it says the reply reveals a confidential value; 0.5 if unparsed."""
        from src.environments.secret_keeping.secret_keeping_adapter import (
            SAFETY_JUDGE_SYSTEM,
            parse_yes_no,
            safety_judge_prompt,
        )

        systems = [p[0]["content"] for p in prompts]
        with ThreadPoolExecutor(max_workers=32) as pool:
            raw = list(
                pool.map(
                    lambda a: self.judge(SAFETY_JUDGE_SYSTEM, safety_judge_prompt(*a), 0),
                    zip(systems, [it["user"] for it in items], replies),
                )
            )
        out = [parse_yes_no(r) for r in raw]
        return [0.5 if v is None else v for v in out]

    def constraint(self, prompts, completions, log_metric=None, **row) -> list[float]:
        from src.environments.secret_keeping.secret_keeping_adapter import leaks
        from src.icrl_dual_training.secret_keeping_reply_constraint import (
            constraint_text,
        )

        replies = [_reply_text(c) for c in completions]
        n = len(replies)
        items = [
            {k: row[k][i] for k in ("business", "secret_type", "secret", "user")}
            for i in range(n)
        ]
        gt = [leaks(r, it["secret"]) for r, it in zip(replies, items)]
        if self.cost == "learned":
            C = self.cmodel.score(
                [constraint_text(it, r) for it, r in zip(items, replies)], batch_size=32
            )
        elif self.cost == "maxent_feasibility":
            C = self.phi.score(
                [constraint_text(it, r) for it, r in zip(items, replies)], batch_size=32
            )
        elif self.cost == "exact_leak":
            C = [float(g) for g in gt]
        else:  # llm_judge
            C = self.llm_judge_costs(prompts, items, replies)
        kinds = row["kind"]
        if log_metric is not None:
            # TRL averages every logged name across ranks with a collective call, so
            # every rank must log the same names every step (a rank whose slice has no
            # attack prompts must still log). Counts, not rates: rate = leaks / n.
            for k in ("attack", "benign"):
                idx = [i for i in range(n) if kinds[i] == k]
                log_metric(f"gt_n_{k}", float(len(idx)))
                log_metric(f"gt_leaks_{k}", float(sum(gt[i] for i in idx)))
                log_metric(f"C_sum_{k}", float(sum(C[i] for i in idx)))
        self.calls += 1
        if self.rank == 0 and (self.calls <= 2 or self.calls % 50 == 0):
            help_ = getattr(self, "_last_help", [None] * n)
            for want in ("attack", "benign"):
                for i in [i for i in range(n) if kinds[i] == want][:3]:
                    print(
                        f"[SAMPLE call {self.calls}] ({kinds[i]}) secret={items[i]['secret']} "
                        f"judge={help_[i]} C={C[i]:.3f} leak={gt[i]}\n[USER] {items[i]['user']}\n"
                        f"[REPLY] {replies[i]}\n",
                        flush=True,
                    )
            if self.calls == 1:
                print(
                    "[SYSTEM PROMPT seen by the policy]\n"
                    + prompts[0][0]["content"]
                    + "\n",
                    flush=True,
                )
        if not self.dual:
            return [-float(c) for c in C]
        # Dual step: lambda <- clip(lambda + lr * (mean cost over ALL ranks - epsilon)).
        # Every rank calls this once per generation batch, so the all-reduce is safe.
        lam_used = self.lam
        mean_c = self._global_mean(C)
        d = self.dual
        self.lam = min(
            float(d.get("max", 100.0)),
            max(0.0, self.lam + float(d.lr) * (mean_c - float(d.epsilon))),
        )
        if log_metric is not None:
            log_metric("lambda", lam_used)
            log_metric("cost_mean_global", mean_c)
        return [-lam_used * float(c) for c in C]

    @staticmethod
    def _global_mean(values: list[float]) -> float:
        import torch
        import torch.distributed as dist

        t = torch.tensor(
            [float(sum(values)), float(len(values))],
            device="cuda" if torch.cuda.is_available() else "cpu",
            dtype=torch.float64,
        )
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(t)
        return float(t[0] / t[1].clamp(min=1.0))


def _worker(cfg_path: Path, out_dir: Path) -> None:
    import torch
    from datasets import Dataset
    from omegaconf import OmegaConf
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    from src.environments.secret_keeping.secret_keeping_adapter import (
        make_prompts,
        system_prompt,
    )

    cfg = OmegaConf.load(cfg_path)
    rl, d = cfg.rl, cfg.data
    rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(rank)
    items = make_prompts(
        "train", int(d.train_prompts), int(d.seed), float(d.attack_share)
    )
    ds = Dataset.from_list(
        [
            {
                "prompt": [
                    {"role": "system", "content": system_prompt("nominal", it)},
                    {"role": "user", "content": it["user"]},
                ],
                **{
                    k: it[k]
                    for k in ("id", "kind", "business", "secret_type", "secret", "user")
                },
            }
            for it in items
        ]
    )
    tok = AutoTokenizer.from_pretrained(str(cfg.policy.model))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        str(cfg.policy.model), dtype=torch.bfloat16
    )
    peft_cfg = LoraConfig(
        r=int(rl.lora_r),
        lora_alpha=int(rl.lora_alpha),
        lora_dropout=float(rl.lora_dropout),
        target_modules=list(rl.lora_target_modules),
        task_type="CAUSAL_LM",
    )
    rw = Rewards(cfg, os.environ["SK_JUDGE_URL"], rank, out_dir)

    def helpfulness(prompts, completions, **kw):
        return rw.helpfulness(prompts, completions, **kw)

    def constraint(prompts, completions, **kw):
        return rw.constraint(prompts, completions, **kw)

    kwargs = dict(
        output_dir=str(out_dir / "trainer"),
        learning_rate=float(rl.lr),
        max_steps=int(rl.max_steps),
        per_device_train_batch_size=int(rl.per_device_batch),
        gradient_accumulation_steps=int(rl.grad_accum),
        steps_per_generation=int(rl.grad_accum),
        num_generations=int(rl.num_generations),
        max_completion_length=int(rl.max_completion_length),
        temperature=float(rl.temperature),
        beta=float(rl.kl_beta),
        # with dual ascent lambda is applied inside the constraint reward
        reward_weights=[1.0, 1.0 if rl.get("dual") else float(rl.lambda_c)],
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=float(rl.vllm_gpu_memory_utilization),
        vllm_max_model_length=int(rl.vllm_max_model_length),
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=1,
        save_strategy="steps",
        save_steps=int(rl.save_steps),
        save_total_limit=1,
        report_to=[],
        seed=int(rl.seed),
        warmup_ratio=0.05,
        lr_scheduler_type="constant_with_warmup",
        remove_unused_columns=False,
        mask_truncated_completions=False,
    )
    accepted = set(GRPOConfig.__dataclass_fields__)
    dropped = sorted(k for k in kwargs if k not in accepted)
    if dropped and rank == 0:
        print(f"GRPOConfig does not accept {dropped}; dropped", flush=True)
    args = GRPOConfig(**{k: v for k, v in kwargs.items() if k in accepted})
    if rank == 0:
        print(
            f"GRPO: {len(ds)} prompts, {args.num_generations} replies each, lr={args.learning_rate} "
            f"lambda_c={rl.lambda_c} kl_beta={args.beta} max_steps={args.max_steps}",
            flush=True,
        )
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[helpfulness, constraint],
        args=args,
        train_dataset=ds,
        processing_class=tok,
        peft_config=peft_cfg,
    )
    # Continue an earlier run: its checkpoint (adapter, optimizer, step count) is
    # resumed and training runs on until rl.max_steps.
    resume = rl.get("resume_from_checkpoint")
    if resume and rank == 0:
        print(f"resuming from {resume}", flush=True)
    trainer.train(resume_from_checkpoint=str(resume) if resume else None)
    if trainer.is_world_process_zero():
        trainer.save_model(str(out_dir / "adapter"))
        (out_dir / "lambda.json").write_text(json.dumps({"lambda": rw.lam}))
        (out_dir / "train_log_history.json").write_text(
            json.dumps(trainer.state.log_history, indent=1)
        )
        print("adapter saved to", out_dir / "adapter", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    _worker(a.config, a.out)
