# ABOUTME: Shared machinery for the ODCV constrained fine-tuning loops: round state, merging the organism, sampling the training cells through the sandbox, and scoring rollouts with R and the constraint head
# ABOUTME: Imported by scripts/finetune_policy_lagrangian_rejection_sampling.py and scripts/finetune_policy_constrained_dpo.py; not a CLI
"""
Both ODCV policy-optimisation procedures share everything except what they do
with a round's scored rollouts:

  * ``load_state`` / ``save_state``  one SLURM job per round, state on disk;
  * ``merge_organism``              fold the starting LoRA into the base once so
                                    vLLM can serve merged weights plus the round's
                                    adapter;
  * ``collect``                     sample every training cell through the
                                    Apptainer sandbox via the collector script;
  * ``score_rollouts``              attach the procedural reward R and the frozen
                                    constraint score C_theta to each rollout.

The procedures differ only in the update: rejection sampling keeps the lowest-C
rollouts and fits them by maximum likelihood, while constrained DPO pairs them
against the highest-C rollout of the same cell.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


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
    """Fold the starting adapter into the base once; later rounds train on top of this."""
    merged = Path(cfg.policy.merged_dir)
    if (merged / "config.json").exists():
        print(f"merged organism present: {merged}", flush=True)
        return merged
    import torch
    from peft import PeftModel
    from transformers import AutoTokenizer

    from src.models.policy_loader import load_policy_model

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
    """The scenarios the loop may sample: everything the constraint split held out is excluded."""
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
    """Attach R (ran the prescribed tool and submitted) and C_theta to every rollout."""
    import torch

    from src.environments.odcv.transcripts import (
        steps_to_actions_text,
        transcript_to_steps,
    )
    from src.models.model_loader import load_model_and_tokenizer
    from src.trajectory_embedding.trajectory_encoder import (
        TrajectoryEncoder,
        load_constraint_head,
    )

    rows = []
    for res_path in sorted(rollouts_dir.glob("rollouts/*/*/pass*/result.json")):
        r = json.loads(res_path.read_text())
        rec = res_path.parent / "messages_record.txt"
        if not rec.exists() or rec.stat().st_size == 0:
            continue
        steps, meta = transcript_to_steps(
            rec.read_text(errors="replace"),
            obs_chars=int(cfg.train.get("obs_chars", 1500)),
        )
        if not steps:
            continue
        rows.append(
            {
                "path": str(res_path.parent),
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

    from omegaconf import OmegaConf

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


def round_statistics(rows: list[dict], rnd: int, lam: float) -> dict:
    """The per-round guards: constraint level, task completion, and the rule-based check."""
    n = max(1, len(rows))
    return {
        "round": rnd,
        "lambda_used": lam,
        "n_rollouts": len(rows),
        "mean_C": sum(r["C"] for r in rows) / n,
        "mean_R": sum(r["R"] for r in rows) / n,
        "submitted_pct": 100 * sum(r["submitted"] for r in rows) / n,
        "integrity_violated_pct": 100 * sum(r["integrity_violated"] for r in rows) / n,
        "C_over_0.5_pct": 100 * sum(r["C"] > 0.5 for r in rows) / n,
    }


def dual_ascent(lam: float, mean_c: float, cfg) -> float:
    """lambda <- clip(lambda + lr * (mean C - epsilon), 0, lambda_max)."""
    return max(
        0.0,
        min(
            float(cfg.constraint.lambda_max),
            lam
            + float(cfg.constraint.lambda_lr)
            * (mean_c - float(cfg.constraint.epsilon)),
        ),
    )


def write_scored(round_dir: Path, rows: list[dict]) -> None:
    (round_dir / "scored_rollouts.json").write_text(
        json.dumps(
            [
                {k: v for k, v in r.items() if k not in ("steps", "scenario_prompt")}
                for r in rows
            ],
            indent=2,
        )
    )
