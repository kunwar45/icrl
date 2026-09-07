#!/usr/bin/env python
# ABOUTME: One round of on-policy constrained KTO fine-tuning on ODCV: serve, sample, score R and C_theta, split rollouts into desirable/undesirable, train with the constraint multiplier as the undesirable weight, update lambda
# ABOUTME: Run: python scripts/finetune_policy_constrained_kto.py --config configs/lagrangian_finetuning/odcv_constrained_kto.yaml [--round N]
"""
The successor to the rejection-sampling loop, redesigned around three measured
failures of that run (jobs 5258482-5261896, diagnostics of 2026-09-07):

1. **The multiplier was inert.** Selecting the top rollouts by R - lambda*C among
   candidates that all have R = 1 ranks them by -C for *every* lambda > 0, so the
   kept set was byte-identical at lambda = 1e-9 and lambda = 10 in all six rounds.
   Six rounds of dual ascent changed nothing. Here lambda is the KTO weight on the
   undesirable half of the loss, so it changes the gradient directly.

2. **A hard threshold deleted the cells that matter.** C_theta is near-binary
   (86-92% of rollouts score below 0.2 or above 0.8), so "keep rollouts with
   C < 0.5" does not separate good from bad *within* a hard cell, it drops the
   hard cell entirely: nine cells produced no admissible rollout in 30-36 samples
   and contributed nothing to training, while cells already 75% clean supplied
   56% of the data. The policy was fine-tuned on behaviour it already had. KTO
   takes unpaired desirable and undesirable examples, so an all-violating cell
   still contributes -- as negatives, which is the only signal it can give.

3. **Positive-only maximum likelihood wastes the negatives.** Roughly 70 examples
   per round survived the filter out of ~360 rollouts, and violating rollouts
   contributed no gradient at all. Here every confidently-scored rollout is used,
   about five times the data per round, and the violations push down.

The constraint head is frozen and was validated to transfer: on the evaluation
rollouts of all four policies of the previous run it scores AUROC 0.75-0.88
against the two-judge label, and only 1-6% of the rollouts it calls clean are
judged misaligned. It is a usable training signal; the previous loop simply
threw most of it away.

State lives in <output.dir>/state.json so each round is one SLURM job:

  round 0   merge the organism (once), serve it, sample the training cells for
            the baseline statistics and lambda_1. No training.
  round k   serve merged + LoRA_{k-1}, sample, score, build the round's examples,
            add them to the cumulative pool, train a FRESH LoRA from the organism
            on the whole pool, update lambda.

Training a fresh adapter each round on the accumulated pool, rather than
continuing the previous adapter, is the choice reported to be robust against
overoptimisation in iterated preference learning; it also means each round sees
strictly more data than the last.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.environments.odcv.transcripts import (  # noqa: E402
    messages_of,
    render_prompt_completion,
)
from src.environments.odcv.vllm_server import VllmServer  # noqa: E402
from src.lagrangian_finetuning.odcv_loop import (  # noqa: E402
    collect,
    dual_ascent,
    load_state,
    merge_organism,
    round_statistics,
    save_state,
    score_rollouts,
    train_scenarios,
    write_scored,
)


def build_examples(rows: list[dict], cfg) -> tuple[list[dict], dict]:
    """Split a round's rollouts into KTO desirable/undesirable examples.

    Both classes require R = 1, so the contrast the policy learns is about *how*
    the task was done and never about whether it was done; without that, KTO
    would happily learn "submit something" from the reward side alone.

    Cells whose rollouts are all desirable are dropped: they carry no contrast,
    and training on them is pure imitation of behaviour the policy already has
    (the mechanism that made 56% of the previous run's data redundant). Cells
    whose rollouts are all undesirable are KEPT -- those are the cells where the
    misalignment lives, and negatives are the only signal they can provide.
    """
    lo = float(cfg.constraint.desirable_max_score)
    hi = float(cfg.constraint.undesirable_min_score)
    by_cell: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        by_cell[(r["framing"], r["scenario"])].append(r)

    examples: list[dict] = []
    dropped_all_clean = 0
    for cell, rs in by_cell.items():
        done = [r for r in rs if r["R"] >= 1.0]
        desirable = [r for r in done if r["C"] < lo]
        undesirable = [r for r in done if r["C"] > hi]
        if not undesirable and len(desirable) == len(rs):
            dropped_all_clean += 1
            continue
        cap = int(cfg.train.max_per_cell_per_class)
        desirable.sort(key=lambda r: r["C"])
        undesirable.sort(key=lambda r: -r["C"])
        for r in desirable[:cap]:
            examples.append({**r, "label": True})
        for r in undesirable[:cap]:
            examples.append({**r, "label": False})

    n_des = sum(e["label"] for e in examples)
    stats = {
        "n_examples": len(examples),
        "n_desirable": n_des,
        "n_undesirable": len(examples) - n_des,
        "cells_contributing": len({(e["framing"], e["scenario"]) for e in examples}),
        "cells_all_clean_dropped": dropped_all_clean,
        "cells_negatives_only": len(
            {(e["framing"], e["scenario"]) for e in examples if not e["label"]}
            - {(e["framing"], e["scenario"]) for e in examples if e["label"]}
        ),
        # the previous run trained on rollouts half as long as those it discarded;
        # if that gap reappears the policy is being taught to stop early, not to comply
        "mean_steps_desirable": (
            sum(e["n_steps"] for e in examples if e["label"]) / max(1, n_des)
        ),
        "mean_steps_undesirable": (
            sum(e["n_steps"] for e in examples if not e["label"])
            / max(1, len(examples) - n_des)
        ),
    }
    return examples, stats


def render(examples: list[dict], tok, max_length: int) -> list[dict]:
    """Chat-template each rollout into KTO's (prompt, completion, label) rows."""
    rendered = []
    for e in examples:
        prompt, completion = render_prompt_completion(
            tok, messages_of(e["scenario_prompt"], e["steps"])
        )
        if len(tok(prompt + completion)["input_ids"]) <= max_length:
            rendered.append(
                {"prompt": prompt, "completion": completion, "label": bool(e["label"])}
            )
    return rendered


def save_pool(round_dir: Path, examples: list[dict]) -> Path:
    """The round's examples, keyed so the cumulative pool can be rebuilt from disk."""
    p = round_dir / "kto_examples.json"
    p.write_text(
        json.dumps(
            [
                {
                    "path": e["path"],
                    "scenario": e["scenario"],
                    "framing": e["framing"],
                    "label": bool(e["label"]),
                    "C": e["C"],
                    "R": e["R"],
                    "n_steps": e["n_steps"],
                    "steps": e["steps"],
                    "scenario_prompt": e["scenario_prompt"],
                }
                for e in examples
            ]
        )
    )
    return p


def load_pool(out: Path, upto_round: int) -> list[dict]:
    """Every example produced in rounds 0..upto_round, deduplicated by rollout path."""
    pool: dict[str, dict] = {}
    for r in range(upto_round + 1):
        p = out / f"round_{r:02d}" / "kto_examples.json"
        if p.exists():
            for e in json.loads(p.read_text()):
                pool[e["path"]] = e
    return list(pool.values())


def kto_train(cfg, merged: Path, pool: list[dict], lam: float, round_dir: Path) -> Path:
    """Fresh LoRA on the merged organism, trained on the cumulative pool.

    lambda enters as the weight on the undesirable half of the KTO loss, which is
    where a constraint multiplier belongs in a preference objective: raising it
    raises the gradient pressure away from violations without touching the
    desirable term. It is clipped so a runaway dual variable cannot silently
    become the entire loss.
    """
    import torch
    from datasets import Dataset
    from peft import LoraConfig, prepare_model_for_kbit_training
    from transformers import AutoTokenizer
    from trl import KTOConfig, KTOTrainer

    from src.models.policy_loader import load_policy_model, lora_target_regex

    tok = AutoTokenizer.from_pretrained(str(merged))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    rows = render(pool, tok, int(cfg.train.max_length))
    n_des = sum(r["label"] for r in rows)
    print(
        f"KTO rows: {len(rows)} of {len(pool)} pool examples fit in {cfg.train.max_length} tokens "
        f"({n_des} desirable / {len(rows) - n_des} undesirable)",
        flush=True,
    )
    if n_des == 0 or n_des == len(rows):
        raise RuntimeError("KTO needs both classes present in the pool")

    quant = bool(cfg.policy.get("quantize_4bit", False))
    model = load_policy_model(
        str(merged), quantize_4bit=quant, dtype=torch.bfloat16, device_map="auto"
    )
    model.config.use_cache = False
    if quant:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
    peft_cfg = LoraConfig(
        r=int(cfg.policy.lora.r),
        lora_alpha=int(cfg.policy.lora.alpha),
        lora_dropout=float(cfg.policy.lora.dropout),
        target_modules=lora_target_regex(model, list(cfg.policy.lora.target_modules)),
        task_type="CAUSAL_LM",
    )

    undesirable_weight = min(
        float(cfg.train.undesirable_weight_max),
        max(float(cfg.train.undesirable_weight_min), lam),
    )
    kwargs = dict(
        output_dir=str(round_dir / "trainer"),
        per_device_train_batch_size=int(cfg.train.per_device_batch),
        gradient_accumulation_steps=int(cfg.train.grad_accum),
        learning_rate=float(cfg.train.lr),
        num_train_epochs=float(cfg.train.epochs),
        beta=float(cfg.train.beta),
        desirable_weight=float(cfg.train.desirable_weight),
        undesirable_weight=undesirable_weight,
        max_length=int(cfg.train.max_length),
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        seed=int(cfg.train.seed),
        warmup_ratio=float(cfg.train.get("warmup_ratio", 0.1)),
        lr_scheduler_type=str(cfg.train.get("scheduler", "cosine")),
        remove_unused_columns=False,
    )
    accepted = set(KTOConfig.__dataclass_fields__)
    dropped = sorted(k for k in kwargs if k not in accepted)
    if dropped:
        print(f"KTOConfig does not accept {dropped}; dropped", flush=True)
    args = KTOConfig(**{k: v for k, v in kwargs.items() if k in accepted})
    print(
        f"training a fresh LoRA on {len(rows)} rows, beta={args.beta}, "
        f"undesirable_weight={undesirable_weight:.2f} (lambda {lam:.2f})",
        flush=True,
    )
    trainer = KTOTrainer(
        model=model,
        ref_model=None,  # with a PEFT policy the reference is the adapter-disabled organism
        args=args,
        train_dataset=Dataset.from_list(rows),
        processing_class=tok,
        peft_config=peft_cfg,
    )
    result = trainer.train()
    adapter_dir = round_dir / "adapter"
    trainer.model.save_pretrained(str(adapter_dir))
    tok.save_pretrained(str(adapter_dir))
    (round_dir / "train_metrics.json").write_text(
        json.dumps(
            {
                "metrics": result.metrics,
                "log_history": trainer.state.log_history,
                "n_rows": len(rows),
                "n_desirable": n_des,
                "undesirable_weight": undesirable_weight,
            },
            indent=2,
            default=str,
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
    stats = round_statistics(rows, rnd, lam)
    examples, ex_stats = build_examples(rows, cfg)
    stats.update(ex_stats)
    write_scored(round_dir, rows)
    save_pool(round_dir, examples)
    print(json.dumps(stats, indent=2), flush=True)

    stats["lambda_next"] = dual_ascent(lam, stats["mean_C"], cfg)

    if rnd >= 1:
        pool = load_pool(out, rnd)
        n_des = sum(e["label"] for e in pool)
        print(
            f"cumulative pool: {len(pool)} examples ({n_des} desirable / {len(pool) - n_des} undesirable) "
            f"over {len({(e['framing'], e['scenario']) for e in pool})} cells",
            flush=True,
        )
        stats["pool_size"] = len(pool)
        if n_des and n_des < len(pool):
            state["adapter"] = str(kto_train(cfg, merged, pool, lam, round_dir))
        else:
            print("pool lacks one of the two classes; adapter unchanged", flush=True)

    state["lambda"] = stats["lambda_next"]
    state["round"] = rnd + 1
    state["history"].append(stats)
    save_state(out, state)
    (round_dir / "round_stats.json").write_text(json.dumps(stats, indent=2))
    print(
        f"=== round {rnd} done; next lambda {stats['lambda_next']:.3f}; adapter {state['adapter']} ===",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
