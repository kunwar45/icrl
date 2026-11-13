# ABOUTME: Lagrangian fine-tuning step for MACHIAVELLI: score every option at the policy's visited states by R - lambda*C_theta, build DPO pairs, train a LoRA, update lambda by dual ascent
# ABOUTME: Use from scripts/run_machiavelli_icrl_pipeline.py: pairs, stats = build_pairs(episodes, scores, lam, cfg); adapter = dpo_train(cfg, pairs, out_dir); lam = dual_ascent(lam, mean_c, cfg)
"""
The objective is the repo's: maximise R - lambda * C_theta, lambda adapted so the
policy's expected constraint cost meets a budget epsilon.

Per visited state s with options a_1..a_n:

  R(s, a_i)  points the option unlocks plus the mean points of short random
             continuations after it (a Monte-Carlo estimate from the game
             engine, the benchmark's own reward), centred across the options
             and divided by `reward_scale`
  C(s, a_i)  the learned constraint's score of (scene, option); ground-truth
             labels are NOT used here, only for evaluation
  S_i        R_i - lambda * C_i

The best and worst option by S form one DPO pair when their gap exceeds
`pair_margin`. lambda therefore changes which pairs exist and which side wins,
so it acts on the gradient: the ODCV rejection-sampling loop failed because its
multiplier could not (see scripts/finetune_policy_constrained_kto.py).

Each round trains a fresh LoRA from the base model on the cumulative pair pool
(the ODCV loop's choice, for robustness against over-optimisation), with an NLL
term on the chosen completion (`rpo_alpha`) against the likelihood collapse the
KTO run hit.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from src.environments.machiavelli.machiavelli_adapter import constraint_text


def option_texts(
    episodes: list[dict], scene_chars: int
) -> list[tuple[int, int, int, str]]:
    """(episode index, step index, option index, constraint text) for every option of every valid step."""
    out = []
    for ei, e in enumerate(episodes):
        for si, s in enumerate(e["steps"]):
            for oi, ch in enumerate(s["choices"]):
                out.append((ei, si, oi, constraint_text(s["scene"], ch, scene_chars)))
    return out


def attach_scores(
    episodes: list[dict], keys: list[tuple[int, int, int, str]], scores: list[float]
) -> None:
    for (ei, si, oi, _), sc in zip(keys, scores):
        st = episodes[ei]["steps"][si]
        st.setdefault("option_C", [None] * len(st["choices"]))
        st["option_C"][oi] = float(sc)


def build_pairs(episodes: list[dict], lam: float, cfg) -> tuple[list[dict], dict]:
    """DPO pairs from scored episodes. Requires option_C on each step (attach_scores)."""
    f = cfg.finetune
    margin, scale = float(f.pair_margin), float(f.reward_scale)
    pairs, taken_c, gaps = [], [], []
    n_states = n_kept = n_chosen_safer = 0
    for e in episodes:
        mp = float(e["max_points"]) or 1.0
        for s in e["steps"]:
            if not s["valid_response"] or len(s["choices"]) < 2:
                continue
            n_states += 1
            C = np.array(s["option_C"], dtype=float)
            taken_c.append(float(C[s["action"]]))
            R_raw = np.array(
                s["option_returns"] if s.get("option_returns") else s["option_rewards"],
                dtype=float,
            )
            R = 100.0 * R_raw / mp
            R = (R - R.mean()) / scale
            S = R - lam * C
            best, worst = int(np.argmax(S)), int(np.argmin(S))
            gap = float(S[best] - S[worst])
            if best == worst or gap < margin:
                continue
            n_kept += 1
            gaps.append(gap)
            n_chosen_safer += int(C[best] < C[worst])
            prompt = [
                {"role": "system", "content": s["system"]},
                {"role": "user", "content": s["user"]},
            ]
            pairs.append(
                {
                    "prompt": prompt,
                    "chosen": [{"role": "assistant", "content": str(best)}],
                    "rejected": [{"role": "assistant", "content": str(worst)}],
                    "game": e["game"],
                    "seed": e["seed"],
                    "t": s["t"],
                    "C_chosen": float(C[best]),
                    "C_rejected": float(C[worst]),
                    "R_chosen": float(R[best]),
                    "R_rejected": float(R[worst]),
                    # ground truth, recorded for audit only; never used to build the pair
                    "gt_chosen_violates": bool(s["option_harms"][best]),
                    "gt_rejected_violates": bool(s["option_harms"][worst]),
                }
            )
    stats = {
        "lambda": lam,
        "states": n_states,
        "pairs": n_kept,
        "mean_C_taken": float(np.mean(taken_c)) if taken_c else None,
        "pairs_where_chosen_has_lower_C": n_chosen_safer,
        "mean_pair_gap": float(np.mean(gaps)) if gaps else None,
        "pairs_gt_chosen_clean_rejected_violating": sum(
            (not p["gt_chosen_violates"]) and p["gt_rejected_violates"] for p in pairs
        ),
        "pairs_gt_chosen_violating": sum(p["gt_chosen_violates"] for p in pairs),
    }
    return pairs, stats


def dual_ascent(lam: float, mean_c: float, cfg) -> float:
    """lambda <- clip(lambda + lr * (mean C of taken actions - epsilon), 0, lambda_max)."""
    f = cfg.finetune
    return float(
        max(
            0.0,
            min(
                float(f.lambda_max),
                lam + float(f.lambda_lr) * (mean_c - float(f.epsilon)),
            ),
        )
    )


def dpo_train(cfg, pairs: list[dict], out_dir: Path, log=print) -> Path:
    """Fresh LoRA from the base policy on `pairs`, data-parallel over `finetune.dpo_gpus`.

    Runs as a torchrun subprocess (one process per GPU) so the trainer gets real
    DDP; with `quantize_4bit` each process holds a 4-bit copy of the policy, which
    is what lets a 32B model train on every GPU at once instead of being split
    layer-by-layer across them. Returns the adapter directory.
    """
    import subprocess
    import sys

    from omegaconf import OmegaConf

    f = cfg.finetune
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "dpo_pairs.json").write_text(json.dumps(pairs))
    (out_dir / "dpo_config.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True))
    gpus = str(f.get("dpo_gpus") or "0")
    n = len([g for g in gpus.split(",") if g.strip()])
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpus)
    cmd = [
        sys.executable, "-m", "torch.distributed.run", f"--nproc_per_node={n}", "--master_port=29517",
        "-m", "src.lagrangian_finetuning.machiavelli_constrained_dpo",
        "--config", str(out_dir / "dpo_config.yaml"),
        "--pairs", str(out_dir / "dpo_pairs.json"),
        "--out", str(out_dir),
    ]
    log(f"DPO: {len(pairs)} pairs on GPUs {gpus} ({n} processes)")
    repo = Path(__file__).resolve().parents[2]
    with open(out_dir / "dpo_train.log", "w") as lf:
        rc = subprocess.run(cmd, cwd=str(repo), env=env, stdout=lf, stderr=subprocess.STDOUT).returncode
    tail = (out_dir / "dpo_train.log").read_text()[-3000:]
    if rc != 0:
        raise RuntimeError(f"DPO training failed (exit {rc}); tail of {out_dir / 'dpo_train.log'}:\n{tail}")
    log("DPO done; " + " | ".join(l for l in tail.splitlines() if "train_loss" in l or "rewards/margins" in l)[-600:])
    return out_dir / "adapter"


def _dpo_worker(cfg_path: Path, pairs_path: Path, out_dir: Path) -> None:
    """One DDP process: load the policy (bf16 or 4-bit) on this rank's GPU and train."""
    import torch
    from datasets import Dataset
    from omegaconf import OmegaConf
    from peft import LoraConfig, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import DPOConfig, DPOTrainer

    cfg = OmegaConf.load(cfg_path)
    pairs = json.loads(Path(pairs_path).read_text())
    f, pol = cfg.finetune, cfg.policy
    rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(rank)
    tok = AutoTokenizer.from_pretrained(str(pol.model))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    quant = bool(f.get("quantize_4bit", False))
    load = dict(dtype=torch.bfloat16, device_map={"": rank})
    if quant:
        load["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(str(pol.model), **load)
    model.config.use_cache = False
    if quant:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    peft_cfg = LoraConfig(
        r=int(f.lora_r), lora_alpha=int(f.lora_alpha), lora_dropout=float(f.lora_dropout),
        target_modules=list(f.lora_target_modules), task_type="CAUSAL_LM",
    )
    # TRL truncates prompt+answer from the end; an over-long prompt would lose the
    # one-token answer the pair is about, so those pairs are dropped instead.
    budget = int(f.max_length) - 16
    rows, too_long = [], 0
    for p in pairs:
        text = tok.apply_chat_template(p["prompt"], tokenize=False, add_generation_prompt=True)
        if len(tok(text)["input_ids"]) > budget:
            too_long += 1
            continue
        rows.append({"prompt": p["prompt"], "chosen": p["chosen"], "rejected": p["rejected"]})
    if rank == 0:
        print(f"DPO pairs: {len(rows)} kept, {too_long} dropped for prompts over {budget} tokens", flush=True)
    sft_w = float(f.rpo_alpha) if f.get("rpo_alpha") else 0.0
    kwargs = dict(
        output_dir=str(out_dir / "trainer"),
        per_device_train_batch_size=int(f.per_device_batch),
        gradient_accumulation_steps=int(f.grad_accum),
        learning_rate=float(f.lr),
        num_train_epochs=float(f.epochs),
        beta=float(f.beta),
        max_length=int(f.max_length),
        truncation_mode="keep_start",
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        ddp_find_unused_parameters=False,
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        seed=int(f.seed),
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        remove_unused_columns=False,
    )
    accepted = set(DPOConfig.__dataclass_fields__)
    # NLL on the chosen answer against likelihood collapse: TRL >= 1.0 spells it as
    # an extra "sft" loss term, older TRL as rpo_alpha.
    if sft_w > 0:
        if "loss_weights" in accepted:
            kwargs.update(loss_type=["sigmoid", "sft"], loss_weights=[1.0, sft_w])
        else:
            kwargs["rpo_alpha"] = sft_w
    dropped = sorted(k for k in kwargs if k not in accepted)
    if dropped and rank == 0:
        print(f"DPOConfig does not accept {dropped}; dropped", flush=True)
    args = DPOConfig(**{k: v for k, v in kwargs.items() if k in accepted})
    if rank == 0:
        print(
            f"DPO on {len(rows)} pairs: 4bit={quant} beta={args.beta} lr={args.learning_rate} "
            f"epochs={args.num_train_epochs} loss={getattr(args, 'loss_type', None)} "
            f"weights={getattr(args, 'loss_weights', None)}",
            flush=True,
        )
    trainer = DPOTrainer(
        model=model, ref_model=None, args=args, train_dataset=Dataset.from_list(rows),
        processing_class=tok, peft_config=peft_cfg,
    )
    result = trainer.train()
    if trainer.is_world_process_zero():
        adapter = out_dir / "adapter"
        trainer.model.save_pretrained(str(adapter))
        tok.save_pretrained(str(adapter))
        (out_dir / "train_metrics.json").write_text(
            json.dumps(
                {"metrics": result.metrics, "log_history": trainer.state.log_history, "n_pairs": len(rows),
                 "dropped_too_long": too_long, "quantize_4bit": quant},
                indent=2, default=str,
            )
        )
        print(f"train_loss {result.metrics.get('train_loss')}", flush=True)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="DDP worker for dpo_train; launched by torchrun")
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--pairs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    _dpo_worker(a.config, a.pairs, a.out)
