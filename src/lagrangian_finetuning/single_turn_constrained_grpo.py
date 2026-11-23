# ABOUTME: Benchmark-neutral GRPO for single-turn benchmarks with three training signals: the task reward alone, reward - lambda * learned cost (the ICRL forward step), or reward x discriminator (VARL)
# ABOUTME: Use: adapter = grpo_train(cfg, out_dir, monitor_url=None, log=log) from scripts/run_single_turn_icrl_pipeline.py; the worker runs under torchrun: python -m src.lagrangian_finetuning.single_turn_constrained_grpo --config ... --out ...
"""
rl.signal picks what each sampled reply is scored with (GRPO is TRL's GRPOTrainer):

  task_only       r                          no intervention
  lagrangian      r - lambda * (1 - C_theta) ICRL forward step; C_theta is the
                                             TrajectoryFeasibility in rl.phi_dir,
                                             lambda by dual ascent (rl.dual)
  verifier_gated  r * D                      VARL (Damani et al. 2026); D is the
                                             shared discriminator behind
                                             MONITOR_URL, trained as replies come in

A fourth arm, GRAFT (rl.signal = gradient_routing), keeps the reward as r and
changes which parameters each reply updates; it has its own trainer in
gradient_routing_grpo.py.

rl.cost_control = shuffled (lagrangian only) is a control run: each reply is
scored with another reply's cost from the same batch, so the penalty keeps its
size but loses its link to the reply.

r is the benchmark's training reward (src/environments/single_turn_benchmark.py).
The benchmark's `violation` and `true` values are logged every step for audit
and never enter any score.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

SIGNALS = ("task_only", "lagrangian", "verifier_gated")


def grpo_train(cfg, out_dir: Path, monitor_url: str | None = None, log=print) -> Path:
    """Run GRPO as a torchrun subprocess over `rl.train_gpus`; returns the adapter directory."""
    import subprocess
    import sys

    from omegaconf import OmegaConf

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "rl_config.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True))
    gpus = str(cfg.rl.train_gpus)
    n = len([g for g in gpus.split(",") if g.strip()])
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpus, MONITOR_URL=monitor_url or "")
    job = os.environ.get("SLURM_JOB_ID")
    port = 29500 + (int(job) % 400 if job else 31)
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        f"--nproc_per_node={n}",
        f"--master_port={port}",
        "-m",
        "src.lagrangian_finetuning.single_turn_constrained_grpo",
        "--config",
        str(out_dir / "rl_config.yaml"),
        "--out",
        str(out_dir),
    ]
    log(
        f"GRPO ({cfg.rl.signal}) on GPUs {gpus} ({n} processes); log {out_dir / 'rl_train.log'}"
    )
    repo = Path(__file__).resolve().parents[2]
    with open(out_dir / "rl_train.log", "a") as lf:
        rc = subprocess.run(
            cmd, cwd=str(repo), env=env, stdout=lf, stderr=subprocess.STDOUT
        ).returncode
    tail = (out_dir / "rl_train.log").read_text()[-4000:]
    if rc != 0:
        raise RuntimeError(
            f"GRPO failed (exit {rc}); tail of {out_dir / 'rl_train.log'}:\n{tail}"
        )
    log("GRPO done; " + tail[-1200:])
    return out_dir / "adapter"


def _reply_text(completion) -> str:
    if isinstance(
        completion, list
    ):  # conversational: [{"role": "assistant", "content": ...}]
        return completion[-1]["content"] if completion else ""
    return str(completion)


def combine(
    signal: str,
    rewards: list[float],
    monitor: list[float] | None,
    lam: float = 0.0,
    warmup: bool = False,
) -> list[float]:
    """The per-reply training score. `monitor` is the cost 1 - C_theta (lagrangian) or D (verifier_gated)."""
    if signal == "task_only":
        return [float(r) for r in rewards]
    if signal == "lagrangian":
        return [float(r) - lam * float(c) for r, c in zip(rewards, monitor)]
    if signal == "verifier_gated":
        # VARL warm-up: only the discriminator trains, so every reply scores the same (no policy gradient)
        return (
            [0.0 for _ in rewards]
            if warmup
            else [float(r) * float(d) for r, d in zip(rewards, monitor)]
        )
    raise ValueError(f"rl.signal must be one of {SIGNALS}, got {signal}")


COST_CONTROLS = ("none", "shuffled")


def shuffled_costs(costs: list[float], seed: int) -> list[float]:
    """The same costs in a random order (the shuffled-cost control): same multiset, no link to the reply."""
    import random

    out = list(costs)
    random.Random(seed).shuffle(out)
    return out


def dual_update(
    lam: float, mean_cost: float, lr: float, epsilon: float, lam_max: float
) -> float:
    """lambda <- clip(lambda + lr * (mean cost - epsilon), 0, max)."""
    return min(float(lam_max), max(0.0, lam + float(lr) * (mean_cost - float(epsilon))))


class Scorer:
    """The single reward function, plus audit logging and a few printed samples."""

    def __init__(self, cfg, rank: int, out_dir: Path):
        from src.environments.single_turn_benchmark import get_benchmark

        self.cfg, self.rank, self.out_dir = cfg, rank, out_dir
        self.bench = get_benchmark(str(cfg.benchmark))
        self.signal = str(cfg.rl.signal)
        if self.signal not in SIGNALS:
            raise ValueError(f"rl.signal must be one of {SIGNALS}, got {self.signal}")
        self.phi = self.disc = None
        if self.signal == "lagrangian":
            from src.icrl_dual_training.secret_keeping_reply_constraint import (
                TrajectoryFeasibility,
            )

            self.phi = TrajectoryFeasibility.load(
                Path(cfg.rl.phi_dir), device=f"cuda:{rank}"
            )
        elif self.signal == "verifier_gated":
            from src.lagrangian_finetuning.verifier_gated_discriminator import (
                DiscriminatorClient,
            )

            self.disc = DiscriminatorClient(os.environ["MONITOR_URL"])
        self.dual = cfg.rl.get("dual")
        self.lam = float(
            cfg.rl.get("lambda_init", (self.dual or {}).get("lambda_init", 0.0))
        )
        self.calls = 0
        self.cost_control = str(cfg.rl.get("cost_control") or "none")
        if self.cost_control not in COST_CONTROLS:
            raise ValueError(
                f"rl.cost_control must be one of {COST_CONTROLS}, got {self.cost_control}"
            )

    def __call__(self, prompts, completions, **kw) -> list[float]:
        replies = [_reply_text(c) for c in completions]
        n = len(replies)
        items = [{k: kw[k][i] for k in self.bench.item_columns} for i in range(n)]
        scores = self.bench.score_replies(items, replies)
        r = [s["reward"] for s in scores]
        viol = [s["violation"] for s in scores]  # audit only
        monitor, warmup, lam_used = None, False, self.lam
        if self.signal == "lagrangian":
            monitor = self.phi.score(
                [[self.bench.monitor_text(it, x)] for it, x in zip(items, replies)]
            )
        elif self.signal == "verifier_gated":
            res = self.disc.step(
                [self.bench.monitor_text(it, x) for it, x in zip(items, replies)],
                [x > 0 for x in r],
            )
            monitor, warmup = res["scores"], bool(res["warmup"])
        log_metric = kw.get("log_metric")
        if self.signal == "lagrangian" and self.dual:
            mean_c = self._global_mean(monitor)
            d = self.dual
            # this function runs once per generation; when one generation feeds several optimizer
            # steps, take that many dual steps so lambda moves as fast per policy step as before
            rl = self.cfg.rl
            per_gen = max(
                1,
                int(rl.get("steps_per_generation", rl.grad_accum))
                // int(rl.grad_accum),
            )
            for _ in range(per_gen):
                self.lam = dual_update(
                    self.lam, mean_c, d.lr, d.epsilon, d.get("max", 100.0)
                )
            if log_metric is not None:
                log_metric("lambda", lam_used)
                log_metric("cost_mean_global", mean_c)
        if self.signal == "lagrangian" and self.cost_control == "shuffled":
            # control run: each reply gets another reply's cost, so the penalty keeps its size and
            # spread but loses its link to what the reply did (the audit metrics above keep the true costs)
            monitor = shuffled_costs(
                monitor,
                seed=int(self.cfg.rl.seed) * 1_000_003
                + self.rank * 10_007
                + self.calls,
            )
        out = combine(self.signal, r, monitor, lam_used, warmup)
        if log_metric is not None:
            # counts, not rates, and the same names on every rank every step (TRL averages them across ranks)
            passed = [i for i in range(n) if r[i] > 0]
            log_metric("n_replies", float(n))
            log_metric("reward_sum", float(sum(r)))
            log_metric("true_sum", float(sum(s["true"] for s in scores)))
            log_metric("violation_sum", float(sum(viol)))
            log_metric("parsed_sum", float(sum(bool(s["parsed"]) for s in scores)))
            log_metric(
                "monitor_sum_violating",
                float(sum(monitor[i] for i in range(n) if viol[i])) if monitor else 0.0,
            )
            log_metric(
                "monitor_sum_passing_clean",
                float(sum(monitor[i] for i in passed if not viol[i]))
                if monitor
                else 0.0,
            )
            log_metric("n_passing_clean", float(sum(1 for i in passed if not viol[i])))
            log_metric("warmup", float(warmup))
        self.calls += 1
        if self.rank == 0 and (self.calls <= 2 or self.calls % 25 == 0):
            if self.calls == 1:
                print(
                    "[PROMPT seen by the policy]\n"
                    + "\n".join(f"<{m['role']}> {m['content']}" for m in prompts[0])
                    + "\n",
                    flush=True,
                )
            for i in range(min(n, 3)):
                m = f"{monitor[i]:.3f}" if monitor else "-"
                print(
                    f"[SAMPLE call {self.calls}] item={items[i]} reward={r[i]} true={scores[i]['true']} "
                    f"violation={viol[i]} monitor={m} score={out[i]:.3f}\n[REPLY] {replies[i]}\n",
                    flush=True,
                )
        return out

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

    from src.environments.single_turn_benchmark import get_benchmark

    cfg = OmegaConf.load(cfg_path)
    rl = cfg.rl
    rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(rank)
    bench = get_benchmark(str(cfg.benchmark))
    items = bench.load_items(cfg, "train")
    ds = Dataset.from_list(
        [
            {
                "prompt": bench.chat_messages(it),
                **{k: it[k] for k in bench.item_columns},
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
    scorer = Scorer(cfg, rank, out_dir)

    def score(prompts, completions, **kw):
        return scorer(prompts, completions, **kw)

    kwargs = dict(
        output_dir=str(out_dir / "trainer"),
        learning_rate=float(rl.lr),
        max_steps=int(rl.max_steps),
        per_device_train_batch_size=int(rl.per_device_batch),
        gradient_accumulation_steps=int(rl.grad_accum),
        # replies are sampled once per `steps_per_generation` backward passes; a multiple of grad_accum reuses
        # one generation for several optimizer steps (the authors' 2 minibatch updates per rollout batch)
        steps_per_generation=int(rl.get("steps_per_generation", rl.grad_accum)),
        num_generations=int(rl.num_generations),
        max_completion_length=int(rl.max_completion_length),
        temperature=float(rl.temperature),
        beta=float(rl.kl_beta),
        reward_weights=[1.0],
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
            f"GRPO[{rl.signal}]: {len(ds)} prompts, {args.num_generations} replies each, lr={args.learning_rate} "
            f"kl_beta={args.beta} max_steps={args.max_steps} model={cfg.policy.model}",
            flush=True,
        )
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[score],
        args=args,
        train_dataset=ds,
        processing_class=tok,
        peft_config=peft_cfg,
    )
    if rl.get("keep_adapter_checkpoints", True):
        # a copy of the adapter at every save (save_steps), kept beyond save_total_limit for publishing
        from src.lagrangian_finetuning.adapter_checkpoint_keeping import (
            keep_adapter_checkpoints_callback,
        )

        trainer.add_callback(keep_adapter_checkpoints_callback(out_dir / "adapter_checkpoints"))
    resume = rl.get("resume_from_checkpoint")
    if resume and rank == 0:
        print(f"resuming from {resume}", flush=True)
    trainer.train(resume_from_checkpoint=str(resume) if resume else None)
    if trainer.is_world_process_zero():
        trainer.save_model(str(out_dir / "adapter"))
        (out_dir / "lambda.json").write_text(json.dumps({"lambda": scorer.lam}))
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
