# ABOUTME: SFT on a single-turn benchmark's demonstrations (LoRA), then merges the adapter into the base weights so every RL arm can start from the same primed model
# ABOUTME: Use: merged_dir = sft_prime(cfg, out_dir, log=log) from scripts/run_single_turn_icrl_pipeline.py; the worker runs under torchrun: python -m src.lagrangian_finetuning.demonstration_sft_priming --config ... --out ...
"""
Countdown-Code only shows reward hacking in Qwen2.5-7B-Instruct after it has
been fine-tuned on the o4-mini demonstrations (Khalifa et al. 2026, Sec. 4), so
that fine-tuning is the starting point of every arm. The demonstrations are
used as released by the benchmark's outcome filter; nothing here reads the
true reward or the violation label.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def sft_prime(cfg, out_dir: Path, log=print) -> Path:
    """Run SFT as a torchrun subprocess over `sft.gpus`, merge, and return the merged model directory."""
    import subprocess
    import sys

    from omegaconf import OmegaConf

    merged = Path(str(cfg.sft.merged_dir))
    if (merged / "config.json").exists():
        log(f"sft_prime: merged model already at {merged}")
        return merged
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "sft_config.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True))
    gpus = str(cfg.sft.gpus)
    n = len([g for g in gpus.split(",") if g.strip()])
    job = os.environ.get("SLURM_JOB_ID")
    port = 29100 + (int(job) % 400 if job else 41)
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        f"--nproc_per_node={n}",
        f"--master_port={port}",
        "-m",
        "src.lagrangian_finetuning.demonstration_sft_priming",
        "--config",
        str(out_dir / "sft_config.yaml"),
        "--out",
        str(out_dir),
    ]
    log(f"SFT priming on GPUs {gpus} ({n} processes); log {out_dir / 'sft_train.log'}")
    repo = Path(__file__).resolve().parents[2]
    with open(out_dir / "sft_train.log", "a") as lf:
        rc = subprocess.run(
            cmd,
            cwd=str(repo),
            env=dict(os.environ, CUDA_VISIBLE_DEVICES=gpus),
            stdout=lf,
            stderr=subprocess.STDOUT,
        ).returncode
    tail = (out_dir / "sft_train.log").read_text()[-3000:]
    if rc != 0:
        raise RuntimeError(
            f"SFT priming failed (exit {rc}); tail of {out_dir / 'sft_train.log'}:\n{tail}"
        )
    log("SFT priming done; " + tail[-600:])
    return merged


def _worker(cfg_path: Path, out_dir: Path) -> None:
    import torch
    from datasets import Dataset
    from omegaconf import OmegaConf
    from peft import LoraConfig, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    from src.environments.single_turn_benchmark import get_benchmark

    cfg = OmegaConf.load(cfg_path)
    f = cfg.sft
    rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(rank)
    bench = get_benchmark(str(cfg.benchmark))
    demos = bench.load_items(cfg, "demonstrations")
    ex = [
        {
            "prompt": bench.chat_messages(d),
            "completion": [{"role": "assistant", "content": d["response"]}],
        }
        for d in demos
    ]
    if rank == 0:
        print(f"SFT priming on {len(ex)} demonstrations", flush=True)
        for e in ex[:2]:
            print(
                f"[SFT EXAMPLE]\n[USER] {e['prompt'][1]['content']}\n[TARGET] {e['completion'][0]['content']}\n",
                flush=True,
            )
    tok = AutoTokenizer.from_pretrained(str(cfg.policy.model))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        str(cfg.policy.model), dtype=torch.bfloat16
    )
    peft_cfg = LoraConfig(
        r=int(f.lora_r),
        lora_alpha=int(f.lora_alpha),
        lora_dropout=float(f.lora_dropout),
        target_modules=list(f.lora_target_modules),
        task_type="CAUSAL_LM",
    )
    kwargs = dict(
        output_dir=str(out_dir / "trainer"),
        per_device_train_batch_size=int(f.per_device_batch),
        gradient_accumulation_steps=int(f.grad_accum),
        learning_rate=float(f.lr),
        num_train_epochs=float(f.epochs),
        max_steps=int(f.get("max_steps", -1)),
        max_length=int(f.max_length),
        completion_only_loss=True,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        ddp_find_unused_parameters=False,
        logging_steps=10,
        save_strategy="no",
        report_to=[],
        seed=int(f.seed),
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
    )
    accepted = set(SFTConfig.__dataclass_fields__)
    args = SFTConfig(**{k: v for k, v in kwargs.items() if k in accepted})
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=Dataset.from_list(ex),
        processing_class=tok,
        peft_config=peft_cfg,
    )
    trainer.train()
    if trainer.is_world_process_zero():
        trainer.save_model(str(out_dir / "adapter"))
        (out_dir / "train_log_history.json").write_text(
            json.dumps(trainer.state.log_history, indent=1)
        )
        del trainer, model
        torch.cuda.empty_cache()
        base = AutoModelForCausalLM.from_pretrained(
            str(cfg.policy.model), dtype=torch.bfloat16
        )
        merged = PeftModel.from_pretrained(
            base, str(out_dir / "adapter")
        ).merge_and_unload()
        tmp = Path(str(f.merged_dir) + ".tmp")
        merged.save_pretrained(str(tmp))
        tok.save_pretrained(str(tmp))
        tmp.rename(Path(str(f.merged_dir)))
        print("merged model saved to", f.merged_dir, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    _worker(a.config, a.out)
