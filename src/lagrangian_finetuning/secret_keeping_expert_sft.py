# ABOUTME: Behaviour-cloning baseline for the secret-keeping benchmark: supervised fine-tuning (LoRA) on the oracle expert's chosen replies, under the nominal prompt (no rule)
# ABOUTME: Use: adapter = sft_train(cfg, out_dir, rows, log=log) from the pipeline; the worker runs under torchrun: python -m src.lagrangian_finetuning.secret_keeping_expert_sft --config ... --data ... --out ...
"""
The same expert demonstrations the ICRL constraint learns from (per training
chat, the most helpful expert reply that does not leak), used directly as SFT
targets. The prompt is the nominal system prompt (no confidentiality rule) plus
the user message, exactly what the RL policy sees, so the comparison is
"imitate the expert" vs "learn the expert's constraint, then RL against it".
Chats with no expert reply (all expert samples leaked) are skipped.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def expert_sft_examples(rows: list[dict]) -> list[dict]:
    """Prompt-completion examples from demo rows that have an expert pick."""
    from src.environments.secret_keeping.secret_keeping_adapter import system_prompt

    out = []
    for r in rows:
        e = r.get("expert_pick")
        if e is None:
            continue
        out.append(
            {
                "prompt": [
                    {"role": "system", "content": system_prompt("nominal", r)},
                    {"role": "user", "content": r["user"]},
                ],
                "completion": [
                    {"role": "assistant", "content": r["expert_replies"][e]}
                ],
                "kind": r["kind"],
            }
        )
    return out


def sft_train(cfg, out_dir: Path, rows: list[dict], log=print) -> Path:
    """Run SFT as a torchrun subprocess over `sft.gpus`; returns the adapter directory."""
    import subprocess
    import sys

    from omegaconf import OmegaConf

    out_dir.mkdir(parents=True, exist_ok=True)
    ex = expert_sft_examples(rows)
    (out_dir / "sft_examples.json").write_text(json.dumps(ex))
    (out_dir / "sft_config.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True))
    gpus = str(cfg.sft.gpus)
    n = len([g for g in gpus.split(",") if g.strip()])
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpus)
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        f"--nproc_per_node={n}",
        "--master_port=29541",
        "-m",
        "src.lagrangian_finetuning.secret_keeping_expert_sft",
        "--config",
        str(out_dir / "sft_config.yaml"),
        "--data",
        str(out_dir / "sft_examples.json"),
        "--out",
        str(out_dir),
    ]
    log(f"SFT: {len(ex)} expert examples on GPUs {gpus} ({n} processes)")
    repo = Path(__file__).resolve().parents[2]
    with open(out_dir / "sft_train.log", "w") as lf:
        rc = subprocess.run(
            cmd, cwd=str(repo), env=env, stdout=lf, stderr=subprocess.STDOUT
        ).returncode
    tail = (out_dir / "sft_train.log").read_text()[-3000:]
    if rc != 0:
        raise RuntimeError(
            f"SFT failed (exit {rc}); tail of {out_dir / 'sft_train.log'}:\n{tail}"
        )
    log("SFT done; " + tail[-800:])
    return out_dir / "adapter"


def _worker(cfg_path: Path, data_path: Path, out_dir: Path) -> None:
    import torch
    from datasets import Dataset
    from omegaconf import OmegaConf
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    cfg = OmegaConf.load(cfg_path)
    f = cfg.sft
    rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(rank)
    ex = json.loads(Path(data_path).read_text())
    if rank == 0:
        for e in [x for x in ex if x["kind"] == "attack"][:3] + [
            x for x in ex if x["kind"] == "benign"
        ][:2]:
            print(
                f"[SFT EXAMPLE] ({e['kind']})\n[SYSTEM] {e['prompt'][0]['content']}\n[USER] {e['prompt'][1]['content']}\n"
                f"[TARGET] {e['completion'][0]['content']}\n",
                flush=True,
            )
    ds = Dataset.from_list(
        [{"prompt": e["prompt"], "completion": e["completion"]} for e in ex]
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
        max_length=int(f.max_length),
        completion_only_loss=True,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        ddp_find_unused_parameters=False,
        logging_steps=5,
        save_strategy="no",
        report_to=[],
        seed=int(f.seed),
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
    )
    accepted = set(SFTConfig.__dataclass_fields__)
    dropped = sorted(k for k in kwargs if k not in accepted)
    if dropped and rank == 0:
        print(f"SFTConfig does not accept {dropped}; dropped", flush=True)
    args = SFTConfig(**{k: v for k, v in kwargs.items() if k in accepted})
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=ds,
        processing_class=tok,
        peft_config=peft_cfg,
    )
    trainer.train()
    if trainer.is_world_process_zero():
        trainer.save_model(str(out_dir / "adapter"))
        (out_dir / "train_log_history.json").write_text(
            json.dumps(trainer.state.log_history, indent=1)
        )
        print("adapter saved to", out_dir / "adapter", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    _worker(a.config, a.data, a.out)
