#!/usr/bin/env python
# ABOUTME: One round of Lagrangian rejection-sampling fine-tuning on ODCV: serve, sample, score R and C_theta, keep by R - lambda*C, SFT the LoRA, update lambda
# ABOUTME: Run: python scripts/finetune_policy_lagrangian_rejection_sampling.py --config configs/lagrangian_finetuning/odcv_lagrangian_rejection_sampling.yaml [--round N]
"""
The ICRL fine-tuning stage for a benchmark whose episodes cost seconds of
sandbox and dozens of model turns: policy improvement against R - lambda*C_theta
with the gradient step replaced by supervised learning on the samples the
objective selects. Same lambda dynamics as Lagrangian PPO (dual ascent on the
mean constraint score against a budget epsilon); no per-token log-probs.

State lives in <output.dir>/state.json so each round is one SLURM job:

  round 0   merge the organism's LoRA into the base (once), serve it, sample the
            training cells: the baseline mean C_theta, submission and integrity
            rates; lambda_1 from the dual update. No training.
  round k   serve merged + LoRA_{k-1}, sample, score, keep the top keep_per_cell
            rollouts per cell by R - lambda_k*C among those with R=1 and
            C < keep_max_score, fine-tune LoRA_{k-1} -> LoRA_k on them, update lambda.

Guards written every round: submission rate (inaction collapse), integrity
violation rate (a rule-based check the head never saw), kept-sample counts.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.policy_loader import load_policy_model, lora_target_regex  # noqa: E402

from src.environments.odcv.transcripts import (  # noqa: E402
    messages_of,
    render_prompt_completion,
    steps_to_actions_text,
    transcript_to_steps,
)
from src.environments.odcv.vllm_server import VllmServer  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def load_state(out: Path, cfg) -> dict:
    p = out / "state.json"
    if p.exists():
        return json.loads(p.read_text())
    return {
        "round": 0,
        "lambda": float(cfg.constraint.lambda_init),
        "adapter": None,
        "history": [],
    }


def save_state(out: Path, state: dict) -> None:
    (out / "state.json").write_text(json.dumps(state, indent=2))


def save_processor_files(base_model: str, merged: Path) -> None:
    """vLLM loads Qwen3.6 as a conditional-generation model and wants its image/video
    processor configs beside the weights; save_pretrained on the model writes none of
    them (round 0 of job 5229271 died on exactly that). Save the processor, and copy
    every non-weight file of the source snapshot as a belt-and-braces fallback."""
    try:
        from transformers import AutoProcessor

        AutoProcessor.from_pretrained(base_model).save_pretrained(str(merged))
    except Exception as e:
        print(
            f"AutoProcessor save failed ({str(e)[:120]}); copying snapshot files instead"
        )
    try:
        from huggingface_hub import snapshot_download

        snap = Path(snapshot_download(base_model, local_files_only=True))
        for f in snap.iterdir():
            if (
                f.is_file()
                and not f.name.endswith(".safetensors")
                and f.name != "model.safetensors.index.json"
                and not (merged / f.name).exists()
            ):
                shutil.copy(f, merged / f.name)
    except Exception as e:
        print(f"snapshot copy skipped ({str(e)[:120]})")


def merge_organism(cfg) -> Path:
    merged = Path(cfg.policy.merged_dir)
    if (merged / "config.json").exists():
        print(f"merged organism present: {merged}", flush=True)
        return merged
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.time()
    model = load_policy_model(
        cfg.policy.base_model, dtype=torch.bfloat16, device_map="auto"
    )
    model = PeftModel.from_pretrained(
        model, cfg.policy.start_adapter
    ).merge_and_unload()
    merged.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(merged), safe_serialization=True, max_shard_size="5GB")
    AutoTokenizer.from_pretrained(cfg.policy.base_model).save_pretrained(str(merged))
    save_processor_files(cfg.policy.base_model, merged)
    print(
        f"merged and saved organism to {merged} in {time.time() - t0:.0f}s", flush=True
    )
    del model
    torch.cuda.empty_cache()
    return merged


def train_scenarios(cfg) -> list[str]:
    audit = json.loads(Path(cfg.rollouts.audit_file).read_text())
    held = set(
        json.loads(Path(cfg.rollouts.splits_file).read_text())["held_out_task_ids"]
    )
    return sorted(
        r["scenario"] for r in audit["scenarios"] if r["scenario"] not in held
    )


def collect(
    cfg, round_dir: Path, base_url: str, model_name: str, scenarios: list[str]
) -> Path:
    """Sample the training cells via the collector script (its own process, its own summary)."""
    out = round_dir / "rollouts"
    cmd = [
        sys.executable,
        str(REPO / "scripts/collect_odcv_rollouts.py"),
        "--config",
        str(REPO / "configs/trajectory_collection/odcv_rollouts.yaml"),
        "--override",
        f"benchmark.bench_dir={cfg.rollouts.bench_dir}",
        "--override",
        f"benchmark.audit_file={cfg.rollouts.audit_file}",
        "--override",
        f"benchmark.base_sif={cfg.rollouts.base_sif}",
        "--override",
        "benchmark.scenarios=[" + ",".join(scenarios) + "]",
        "--override",
        f"model.name={model_name}",
        "--override",
        f"model.base_url={base_url}",
        "--override",
        f"rollouts.passes={int(cfg.rollouts.samples_per_cell)}",
        "--override",
        f"rollouts.temperature={float(cfg.rollouts.temperature)}",
        "--override",
        f"rollouts.concurrency={int(cfg.rollouts.concurrency)}",
        "--override",
        f"rollouts.timeout_s={int(cfg.rollouts.timeout_s)}",
        "--override",
        f"output.dir={out}",
    ]
    subprocess.run(cmd, check=True)
    return out


def score_rollouts(cfg, rollouts_dir: Path, audit_rows: dict) -> list[dict]:
    """Attach R and C_theta to every rollout found under rollouts_dir."""
    import torch
    from src.models.model_loader import load_model_and_tokenizer
    from src.trajectory_embedding.trajectory_encoder import (
        TrajectoryEncoder,
        load_constraint_head,
    )
    from omegaconf import OmegaConf

    rows = []
    for res in sorted(rollouts_dir.glob("rollouts/*/*/pass*/result.json")):
        r = json.loads(res.read_text())
        rec = res.parent / "messages_record.txt"
        if not rec.exists() or rec.stat().st_size == 0:
            continue
        steps, meta = transcript_to_steps(
            rec.read_text(errors="replace"), obs_chars=int(cfg.train.obs_chars)
        )
        if not steps:
            continue
        rows.append(
            {
                "path": str(res.parent),
                "scenario": r["scenario"],
                "framing": r["framing"],
                "steps": steps,
                "scenario_prompt": meta["scenario_prompt"],
                "R": float(
                    bool(r.get("submitted")) and bool(r.get("prescribed_tool_ran"))
                ),
                "submitted": bool(r.get("submitted")),
                "integrity_violated": bool(r.get("integrity_violated")),
                "n_steps": len(steps),
            }
        )
    if not rows:
        return rows
    dummy = OmegaConf.create({"paths": {"model_cache": None}})
    backbone, tok = load_model_and_tokenizer(
        cfg.constraint.encoder_model, dummy, causal_lm=False
    )
    enc = TrajectoryEncoder(
        model=backbone, tokenizer=tok, max_length=int(cfg.constraint.max_length)
    )
    if torch.cuda.is_available():
        enc.to("cuda")
    load_constraint_head(
        enc, str(cfg.constraint.head_path), model_name=cfg.constraint.encoder_model
    )
    assert enc.text_mode == "actions_only", (
        f"head text_mode is {enc.text_mode}; this loop scores actions-only text"
    )
    enc.eval()
    texts = [steps_to_actions_text(r["steps"]) for r in rows]
    with torch.no_grad():
        pooled = enc.embed_texts(texts, batch_size=8)
        scores = enc.head(pooled).squeeze(-1).float().cpu().tolist()
    for r, c in zip(rows, scores):
        r["C"] = float(c)
    del enc, backbone
    torch.cuda.empty_cache()
    return rows


def select(
    rows: list[dict], lam: float, keep_per_cell: int, keep_max_score: float
) -> list[dict]:
    by_cell = defaultdict(list)
    for r in rows:
        r["objective"] = r["R"] - lam * r["C"]
        by_cell[(r["framing"], r["scenario"])].append(r)
    kept = []
    for cell, rs in by_cell.items():
        ok = [r for r in rs if r["R"] >= 1.0 and r["C"] < keep_max_score]
        ok.sort(key=lambda r: -r["objective"])
        kept.extend(ok[:keep_per_cell])
    return kept


def sft(
    cfg, merged: Path, prev_adapter: Path | None, kept: list[dict], round_dir: Path
) -> Path:
    import torch
    from datasets import Dataset
    from peft import LoraConfig, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    # TRL logs the mean entropy of the policy logits every step. On this model the
    # [2, L, 248320] fp32 logits (vocab 248k) make that reshape die with "unspecified
    # launch failure" once gradients are attached (jobs 5229579, 5230779), while the
    # loss itself is fine. Entropy is a log line; skip it.
    import trl.trainer.sft_trainer as _trl_mod

    _trl_mod.entropy_from_logits = lambda logits, chunk_size=128: torch.zeros(
        logits.shape[:-1], device=logits.device, dtype=torch.float32
    )

    tok = AutoTokenizer.from_pretrained(str(merged))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    examples = []
    for r in kept:
        prompt, completion = render_prompt_completion(
            tok, messages_of(r["scenario_prompt"], r["steps"])
        )
        n = len(tok(prompt + completion)["input_ids"])
        if n <= int(cfg.train.max_length):
            examples.append({"prompt": prompt, "completion": completion})
    print(
        f"SFT examples: {len(examples)} of {len(kept)} kept rollouts fit in {cfg.train.max_length} tokens",
        flush=True,
    )
    quant = bool(cfg.policy.get("quantize_4bit", False))
    model = load_policy_model(str(merged), quantize_4bit=quant, dtype=torch.bfloat16, device_map="auto")
    model.config.use_cache = False
    if quant:
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                                                gradient_checkpointing_kwargs={"use_reentrant": False})
    if prev_adapter is not None:
        model = PeftModel.from_pretrained(model, str(prev_adapter), is_trainable=True)
        peft_cfg = None
    else:
        peft_cfg = LoraConfig(
            r=int(cfg.policy.lora.r),
            lora_alpha=int(cfg.policy.lora.alpha),
            lora_dropout=float(cfg.policy.lora.dropout),
            target_modules=lora_target_regex(
                model, list(cfg.policy.lora.target_modules)
            ),
            task_type="CAUSAL_LM",
        )
    kwargs = dict(
        output_dir=str(round_dir / "trainer"),
        per_device_train_batch_size=int(cfg.train.per_device_batch),
        gradient_accumulation_steps=int(cfg.train.grad_accum),
        learning_rate=float(cfg.train.lr),
        num_train_epochs=float(cfg.train.epochs),
        max_length=int(cfg.train.max_length),
        completion_only_loss=True,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        seed=int(cfg.train.seed),
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        remove_unused_columns=False,
        use_logits_to_keep=True,
    )
    accepted = set(SFTConfig.__dataclass_fields__)
    args = SFTConfig(**{k: v for k, v in kwargs.items() if k in accepted})
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=Dataset.from_list(examples),
        processing_class=tok,
        peft_config=peft_cfg,
    )
    t0 = time.time()
    result = trainer.train()
    adapter_dir = round_dir / "adapter"
    trainer.model.save_pretrained(str(adapter_dir))
    tok.save_pretrained(str(adapter_dir))
    (round_dir / "train_metrics.json").write_text(
        json.dumps(
            {
                "n_examples": len(examples),
                "train_seconds": time.time() - t0,
                "train_result": result.metrics,
                "log_history": trainer.state.log_history,
            },
            indent=2,
        )
    )
    del trainer, model
    torch.cuda.empty_cache()
    return adapter_dir


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--override", action="append", default=[])
    ap.add_argument(
        "--round", type=int, default=None, help="default: next round from state.json"
    )
    a = ap.parse_args()
    from omegaconf import OmegaConf

    cfg = OmegaConf.merge(OmegaConf.load(a.config), OmegaConf.from_dotlist(a.override))
    out = Path(cfg.output.dir)
    out.mkdir(parents=True, exist_ok=True)
    state = load_state(out, cfg)
    rnd = state["round"] if a.round is None else a.round
    round_dir = out / f"round_{rnd:02d}"
    round_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"=== {cfg.run_name} round {rnd} lambda={state['lambda']:.3f} adapter={state['adapter']} ===",
        flush=True,
    )

    merged = merge_organism(cfg)
    scenarios = train_scenarios(cfg)
    audit_rows = {
        r["scenario"]: r
        for r in json.loads(Path(cfg.rollouts.audit_file).read_text())["scenarios"]
    }
    adapter = Path(state["adapter"]) if state["adapter"] else None

    serve = cfg.rollouts.serve
    with VllmServer(
        str(merged),
        lora_path=str(adapter) if adapter else None,
        lora_name="policy",
        tensor_parallel=int(serve.tensor_parallel),
        gpus=str(serve.gpus),
        max_model_len=int(serve.max_model_len),
        max_num_seqs=int(serve.max_num_seqs),
        gpu_fraction=float(serve.gpu_fraction),
        log_path=round_dir / "vllm.log",
    ) as server:
        rollouts_dir = collect(
            cfg, round_dir, server.base_url, server.served_name, scenarios
        )

    rows = score_rollouts(cfg, rollouts_dir, audit_rows)
    lam = float(state["lambda"])
    mean_c = sum(r["C"] for r in rows) / max(1, len(rows))
    stats = {
        "round": rnd,
        "lambda_used": lam,
        "n_rollouts": len(rows),
        "mean_C": mean_c,
        "mean_R": sum(r["R"] for r in rows) / max(1, len(rows)),
        "submitted_pct": 100 * sum(r["submitted"] for r in rows) / max(1, len(rows)),
        "integrity_violated_pct": 100
        * sum(r["integrity_violated"] for r in rows)
        / max(1, len(rows)),
        "C_over_0.5_pct": 100 * sum(r["C"] > 0.5 for r in rows) / max(1, len(rows)),
    }
    kept = select(
        rows, lam, int(cfg.rollouts.keep_per_cell), float(cfg.constraint.keep_max_score)
    )
    stats["n_kept"] = len(kept)
    stats["kept_cells"] = len({(r["framing"], r["scenario"]) for r in kept})
    (round_dir / "scored_rollouts.json").write_text(
        json.dumps(
            [
                {k: v for k, v in r.items() if k not in ("steps", "scenario_prompt")}
                for r in rows
            ],
            indent=2,
        )
    )
    print(json.dumps(stats, indent=2), flush=True)

    # dual ascent on the mean constraint score of THIS round's policy
    new_lam = max(
        0.0,
        min(
            float(cfg.constraint.lambda_max),
            lam
            + float(cfg.constraint.lambda_lr)
            * (mean_c - float(cfg.constraint.epsilon)),
        ),
    )
    stats["lambda_next"] = new_lam

    if rnd >= 1 and kept:
        new_adapter = sft(cfg, merged, adapter, kept, round_dir)
        state["adapter"] = str(new_adapter)
    elif rnd >= 1:
        print("no rollouts qualified for SFT this round; adapter unchanged", flush=True)
    state["lambda"] = new_lam
    state["round"] = rnd + 1
    state["history"].append(stats)
    save_state(out, state)
    (round_dir / "round_stats.json").write_text(json.dumps(stats, indent=2))
    print(
        f"=== round {rnd} done; next lambda {new_lam:.3f}; adapter {state['adapter']} ===",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
