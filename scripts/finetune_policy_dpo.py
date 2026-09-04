#!/usr/bin/env python
# ABOUTME: DPO baseline: fine-tune a LoRA policy on same-cell expert/unsafe trajectory pairs, with no learned constraint
# ABOUTME: Run: python scripts/finetune_policy_dpo.py --config configs/lagrangian_finetuning/odcv_dpo_baseline.yaml
"""
The control the Lagrangian stage has to beat. If preferring audited expert
traces over unsafe traces of the same cell already moves held-out
misalignment, the constraint head added nothing; if it does not, the head has
a job.

Policy = base model with the organism's LoRA MERGED in (so the reference model
TRL derives by disabling adapters is the organism itself, not the raw base),
then a fresh LoRA trained with DPO. Pairs come from
scripts/build_odcv_dpo_pairs.py: prompt / chosen / rejected strings already in
the policy's chat template.

Outputs under output.dir: adapter/ (PEFT weights), train_metrics.json,
eval_metrics.json (held-out pair preference accuracy and reward margin — a
proxy; the real number needs rollouts through the benchmark harness).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.policy_loader import load_policy_model, lora_target_regex  # noqa: E402


def load_pairs(path: Path):
    from datasets import Dataset

    rows = [json.loads(line) for line in open(path) if line.strip()]
    keep = ("prompt", "chosen", "rejected")
    return Dataset.from_list([{k: r[k] for k in keep} for r in rows]), rows


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument(
        "--override",
        action="append",
        default=[],
        help="dotlist overrides, e.g. train.epochs=1",
    )
    ap.add_argument(
        "--dry-run", action="store_true", help="load data and print sizes, then exit"
    )
    a = ap.parse_args()

    from omegaconf import OmegaConf

    cfg = OmegaConf.merge(OmegaConf.load(a.config), OmegaConf.from_dotlist(a.override))
    out_dir = Path(cfg.output.dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(cfg.data.local_dir)
    train_ds, train_rows = load_pairs(data_dir / "train.jsonl")
    eval_ds, eval_rows = load_pairs(data_dir / "eval.jsonl")
    print(
        f"pairs: train {len(train_ds)} ({len({r['scenario'] for r in train_rows})} scenarios), "
        f"eval {len(eval_ds)} ({len({r['scenario'] for r in eval_rows})} scenarios)"
    )
    if a.dry_run:
        return 0

    import torch
    from peft import LoraConfig, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    tok = AutoTokenizer.from_pretrained(cfg.policy.base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    t0 = time.time()
    model = load_policy_model(
        cfg.policy.base_model, dtype=torch.bfloat16, device_map="auto"
    )
    print(
        f"base loaded in {time.time() - t0:.0f}s; device map over {torch.cuda.device_count()} GPUs"
    )
    if cfg.policy.get("start_adapter"):
        # The organism = base + its LoRA. Merge so that TRL's adapter-disabled
        # reference is the organism, not the raw base.
        model = PeftModel.from_pretrained(model, cfg.policy.start_adapter)
        model = model.merge_and_unload()
        print(f"merged start adapter {cfg.policy.start_adapter}")
    model.config.use_cache = False

    lora = LoraConfig(
        r=int(cfg.policy.lora.r),
        lora_alpha=int(cfg.policy.lora.alpha),
        lora_dropout=float(cfg.policy.lora.dropout),
        target_modules=lora_target_regex(model, list(cfg.policy.lora.target_modules)),
        task_type="CAUSAL_LM",
    )
    dpo_kwargs = dict(
        output_dir=str(out_dir / "trainer"),
        per_device_train_batch_size=int(cfg.train.per_device_batch),
        per_device_eval_batch_size=int(cfg.train.per_device_batch),
        gradient_accumulation_steps=int(cfg.train.grad_accum),
        learning_rate=float(cfg.train.lr),
        num_train_epochs=float(cfg.train.epochs),
        beta=float(cfg.train.beta),
        max_length=int(cfg.train.max_length),
        max_prompt_length=int(cfg.train.max_prompt_length),
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=1,
        eval_strategy="steps",
        eval_steps=int(cfg.train.eval_steps),
        save_strategy="no",
        report_to=[],
        seed=int(cfg.train.seed),
        warmup_ratio=float(cfg.train.get("warmup_ratio", 0.1)),
        lr_scheduler_type=str(cfg.train.get("scheduler", "cosine")),
        remove_unused_columns=False,
    )
    # TRL renames DPOConfig fields between minor versions (max_prompt_length
    # came and went); keep only what this version accepts and say what was dropped.
    accepted = set(DPOConfig.__dataclass_fields__)
    dropped = sorted(k for k in dpo_kwargs if k not in accepted)
    if dropped:
        print(f"DPOConfig ({__import__('trl').__version__}) does not accept {dropped}; dropped")
    args = DPOConfig(**{k: v for k, v in dpo_kwargs.items() if k in accepted})
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tok,
        peft_config=lora,
    )
    print(
        "held-out eval before training (log-ratio is 0 at init, so accuracy is trivially 0.5):"
    )
    before = trainer.evaluate()
    print(
        json.dumps(
            {k: v for k, v in before.items() if "rewards" in k or "loss" in k}, indent=2
        )
    )
    t0 = time.time()
    result = trainer.train()
    train_secs = time.time() - t0
    after = trainer.evaluate()
    print("held-out eval after training:")
    print(
        json.dumps(
            {k: v for k, v in after.items() if "rewards" in k or "loss" in k}, indent=2
        )
    )

    trainer.model.save_pretrained(str(out_dir / "adapter"))
    tok.save_pretrained(str(out_dir / "adapter"))
    (out_dir / "train_metrics.json").write_text(
        json.dumps(
            {
                "train_seconds": train_secs,
                "train_result": result.metrics,
                "log_history": trainer.state.log_history,
                "config": OmegaConf.to_container(cfg, resolve=True),
            },
            indent=2,
        )
    )
    (out_dir / "eval_metrics.json").write_text(
        json.dumps(
            {"before": before, "after": after, "n_eval_pairs": len(eval_ds)}, indent=2
        )
    )
    print(
        f"adapter saved: {out_dir / 'adapter'}  ({train_secs / 60:.1f} min of training)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
