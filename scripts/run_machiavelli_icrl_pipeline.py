#!/usr/bin/env python
# ABOUTME: End-to-end ICRL on MACHIAVELLI in one resumable job: expert + unsafe demos, constraint C_theta, Lagrangian DPO rounds, held-out evaluation of ground-truth violations
# ABOUTME: Run: python scripts/run_machiavelli_icrl_pipeline.py --config configs/lagrangian_finetuning/machiavelli_constrained_dpo.yaml [--smoke] [--stages demos,constraint,finetune,evaluate]
"""
Stages, each skipped when its outputs already exist (so a killed job resumes):

  demos       serve the base policy; collect expert-prompt and nominal-prompt
              episodes on the training games (the two demo sets), and evaluate the
              base policy (nominal prompt, and expert prompt as a reference) on
              held-out and training games
  constraint  train C_theta on expert vs unsafe steps; test it on every option
              of every state in the base policy's held-out evaluation episodes
  finetune    rounds of: serve base + current adapter, roll out on training
              games, score every option with C_theta, build DPO pairs by
              R - lambda*C, train a fresh LoRA on the cumulative pairs, update lambda
  evaluate    serve base + final adapter; evaluate exactly as the base policy was;
              write results.json comparing ground-truth violations and points

--smoke shrinks everything (2 games, 5 steps, 1 round, tiny training) and prints
the exact model input, output and ground-truth verdict for 5 steps per arm.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_cfg(path: Path, overrides: list[str], smoke: bool):
    from omegaconf import OmegaConf

    cfg = OmegaConf.merge(OmegaConf.load(path), OmegaConf.from_dotlist(overrides))
    if smoke:
        cfg = OmegaConf.merge(
            cfg,
            OmegaConf.create(
                {
                    "run_name": cfg.run_name + "_smoke",
                    "games": {
                        "train": list(cfg.games.train)[:2],
                        "heldout": list(cfg.games.heldout)[:1],
                    },
                    "rollouts": {
                        "max_steps": 5,
                        "demo_seeds": [0, 1],
                        "round_seeds": [10],
                        "workers": 8,
                        "option_returns": {"horizon": 3, "samples": 1},
                    },
                    "constraint": {
                        "n_iterations": 2,
                        "n_constraint_steps": 5,
                        "max_demos_per_class": 64,
                    },
                    "finetune": {
                        "rounds": 1,
                        "max_pairs_per_round": 16,
                        "per_device_batch": 2,
                        "grad_accum": 1,
                    },
                    "evaluation": {"seeds": [100]},
                }
            ),
        )
        cfg.output.dir = str(Path(cfg.output.dir).parent / cfg.run_name)
    cfg.output.dir = str(Path(cfg.output.dir).resolve())
    return cfg


def show_examples(title: str, episodes: list[dict], n: int = 5) -> None:
    """Print the exact input, output and ground-truth verdict for n steps."""
    steps = [(e, s) for e in episodes for s in e["steps"]][:n]
    print(f"\n{'=' * 30} {title}: {len(steps)} example steps {'=' * 30}", flush=True)
    for e, s in steps:
        print(f"--- {e['game']} seed {e['seed']} step {s['t']} ---")
        print("[SYSTEM PROMPT]\n" + s["system"])
        print("[USER PROMPT]\n" + s["user"])
        print(
            f"[MODEL OUTPUT] {s['response']!r} -> action {s['action']} (valid={s['valid_response']})"
        )
        print(
            f"[GROUND TRUTH] chosen harms={s['harms']} violated={s['violated']} avoidable={s['avoidable']} "
            f"forced={s['forced']} | per-option harms={s['option_harms']} | reward={s['reward']}"
        )
    print(flush=True)


def served(cfg, out_dir: Path, adapter: Path | None):
    from src.environments.odcv.vllm_server import VllmServer

    s = cfg.policy.serve
    return VllmServer(
        str(cfg.policy.model),
        lora_path=str(adapter) if adapter else None,
        lora_name="policy",
        tensor_parallel=int(s.tensor_parallel),
        gpus=str(s.gpus),
        max_model_len=int(s.max_model_len),
        max_num_seqs=int(s.max_num_seqs),
        gpu_fraction=float(s.gpu_fraction),
        log_path=out_dir / "vllm.log",
        max_lora_rank=max(64, int(cfg.finetune.lora_r)),
        reasoning_parser=s.reasoning_parser,
        tool_call_parser=s.tool_call_parser,
    )


def collect(cfg, policy, games, seeds, kind, *, option_returns=None):
    from src.environments.machiavelli.rollout_collector import collect_episodes

    return collect_episodes(
        list(games),
        list(seeds),
        kind,
        policy,
        max_steps=int(cfg.rollouts.max_steps),
        workers=int(cfg.rollouts.workers),
        option_returns=option_returns,
        log=log,
    )


def evaluate_policy(
    cfg, base_url, model_name, tag: str, out: Path, kinds=("nominal",)
) -> dict:
    from src.environments.machiavelli.rollout_collector import (
        OpenAIChatPolicy,
        episode_metrics,
        save_episodes,
    )

    pol = OpenAIChatPolicy(
        base_url, model_name, temperature=float(cfg.evaluation.temperature)
    )
    res = {}
    for kind in kinds:
        for split in ("heldout", "train"):
            p = out / "eval" / f"{tag}_{kind}_{split}.jsonl"
            eps = collect(cfg, pol, cfg.games[split], cfg.evaluation.seeds, kind)
            save_episodes(p, eps)
            res[f"{kind}_{split}"] = episode_metrics(eps)
            log(f"eval {tag} {kind} {split}: {json.dumps(res[f'{kind}_{split}'])}")
    (out / "eval" / f"{tag}_metrics.json").write_text(json.dumps(res, indent=2))
    return res


# ---- stages -----------------------------------------------------------------


def stage_demos(cfg, out: Path, smoke: bool) -> None:
    from src.environments.machiavelli.rollout_collector import (
        OpenAIChatPolicy,
        episode_metrics,
        save_episodes,
    )

    if (out / "eval" / "base_metrics.json").exists():
        log("demos: done already")
        return
    with served(cfg, out / "demos", None) as server:
        pol = OpenAIChatPolicy(
            server.base_url,
            server.served_name,
            temperature=float(cfg.rollouts.temperature),
        )
        summary = {}
        for kind in ("expert", "nominal"):
            eps = collect(cfg, pol, cfg.games.train, cfg.rollouts.demo_seeds, kind)
            save_episodes(out / "demos" / f"{kind}.jsonl", eps)
            summary[kind] = episode_metrics(eps)
            log(f"demos {kind}: {json.dumps(summary[kind])}")
            if smoke:
                show_examples(f"demo arm '{kind}' prompt", eps)
        (out / "demos" / "summary.json").write_text(json.dumps(summary, indent=2))
        evaluate_policy(
            cfg,
            server.base_url,
            server.served_name,
            "base",
            out,
            kinds=("nominal", "expert"),
        )


def stage_constraint(cfg, out: Path) -> None:
    from src.environments.machiavelli.rollout_collector import load_episodes
    from src.icrl_dual_training.machiavelli_step_constraint import (
        demo_steps,
        evaluate_step_constraint,
        option_rows,
        train_step_constraint,
    )

    cdir = out / "constraint"
    if (cdir / "eval.json").exists():
        log("constraint: done already")
        return
    sc = int(cfg.constraint.scene_chars)
    expert, unsafe = demo_steps(
        load_episodes(out / "demos" / "expert.jsonl"),
        load_episodes(out / "demos" / "nominal.jsonl"),
        scene_chars=sc,
    )
    model = train_step_constraint(cfg, expert, unsafe, cdir, log=log)
    res = {}
    for split in ("heldout", "train"):
        rows = option_rows(
            load_episodes(out / "eval" / f"base_nominal_{split}.jsonl"), scene_chars=sc
        )
        res[split] = evaluate_step_constraint(model, rows)
        log(
            f"constraint on base-policy {split} states (every option, ground truth): {json.dumps(res[split])}"
        )
    (cdir / "eval.json").write_text(json.dumps(res, indent=2))


def stage_finetune(cfg, out: Path, smoke: bool) -> None:
    from src.environments.machiavelli.rollout_collector import (
        OpenAIChatPolicy,
        episode_metrics,
        save_episodes,
    )
    from src.icrl_dual_training.machiavelli_step_constraint import StepConstraint
    from src.lagrangian_finetuning.machiavelli_constrained_dpo import (
        attach_scores,
        build_pairs,
        dpo_train,
        dual_ascent,
        option_texts,
    )

    state_p = out / "state.json"
    state = (
        json.loads(state_p.read_text())
        if state_p.exists()
        else {
            "round": 0,
            "lambda": float(cfg.finetune.lambda_init),
            "adapter": None,
            "history": [],
        }
    )
    rounds = int(cfg.finetune.rounds)
    if state["round"] >= rounds:
        log("finetune: done already")
        return
    cmodel = StepConstraint.load(
        out / "constraint", device=cfg.constraint.get("device")
    )
    sc = int(cfg.constraint.scene_chars)
    while state["round"] < rounds:
        r, lam = state["round"], float(state["lambda"])
        rdir = out / f"round_{r:02d}"
        rdir.mkdir(parents=True, exist_ok=True)
        log(f"=== round {r}: lambda={lam:.3f} adapter={state['adapter']} ===")
        adapter = Path(state["adapter"]) if state["adapter"] else None
        with served(cfg, rdir, adapter) as server:
            pol = OpenAIChatPolicy(
                server.base_url,
                server.served_name,
                temperature=float(cfg.rollouts.temperature),
            )
            eps = collect(
                cfg,
                pol,
                cfg.games.train,
                cfg.rollouts.round_seeds,
                "nominal",
                option_returns=dict(cfg.rollouts.option_returns),
            )
        keys = option_texts(eps, sc)
        attach_scores(eps, keys, cmodel.score([k[3] for k in keys]))
        save_episodes(rdir / "rollouts.jsonl", eps)
        pairs, stats = build_pairs(eps, lam, cfg)
        rng = random.Random(r)
        if len(pairs) > int(cfg.finetune.max_pairs_per_round):
            pairs = rng.sample(pairs, int(cfg.finetune.max_pairs_per_round))
        (rdir / "pairs.json").write_text(json.dumps(pairs))
        stats["rollout_ground_truth"] = episode_metrics(eps)
        log(f"round {r} stats: {json.dumps(stats)}")
        if smoke:
            for p in pairs[:5]:
                print(
                    f"[PAIR] {p['game']} t={p['t']} chosen={p['chosen'][0]['content']} (C={p['C_chosen']:.2f}, "
                    f"R={p['R_chosen']:.2f}, gt_violates={p['gt_chosen_violates']}) rejected={p['rejected'][0]['content']} "
                    f"(C={p['C_rejected']:.2f}, R={p['R_rejected']:.2f}, gt_violates={p['gt_rejected_violates']})",
                    flush=True,
                )
        pool = []
        for k in range(r + 1):
            pp = out / f"round_{k:02d}" / "pairs.json"
            if pp.exists():
                pool += json.loads(pp.read_text())
        log(f"round {r}: training a fresh LoRA on {len(pool)} cumulative pairs")
        if pool:
            new_adapter = dpo_train(cfg, pool, rdir, log=log)
            state["adapter"] = str(new_adapter)
        stats["lambda_next"] = dual_ascent(lam, stats["mean_C_taken"] or 0.0, cfg)
        (rdir / "round_stats.json").write_text(json.dumps(stats, indent=2))
        state["history"].append(stats)
        state["lambda"] = stats["lambda_next"]
        state["round"] = r + 1
        state_p.write_text(json.dumps(state, indent=2))


def stage_evaluate(cfg, out: Path) -> None:
    state = json.loads((out / "state.json").read_text())
    adapter = Path(state["adapter"]) if state.get("adapter") else None
    if adapter is None:
        raise RuntimeError("no fine-tuned adapter to evaluate")
    if not (out / "eval" / "tuned_metrics.json").exists():
        with served(cfg, out / "eval", adapter) as server:
            evaluate_policy(
                cfg,
                server.base_url,
                server.served_name,
                "tuned",
                out,
                kinds=("nominal",),
            )
    base = json.loads((out / "eval" / "base_metrics.json").read_text())
    tuned = json.loads((out / "eval" / "tuned_metrics.json").read_text())
    cons = json.loads((out / "constraint" / "eval.json").read_text())
    keys = ["violation_rate", "avoidable_violation_rate", "points_pct", "invalid_rate"]
    table = {}
    for split in ("heldout", "train"):
        table[split] = {
            k: {
                "base_nominal": [
                    base[f"nominal_{split}"][k],
                    base[f"nominal_{split}"][k + "_ci95"],
                ],
                "base_expert_prompt": [
                    base[f"expert_{split}"][k],
                    base[f"expert_{split}"][k + "_ci95"],
                ],
                "tuned_nominal": [
                    tuned[f"nominal_{split}"][k],
                    tuned[f"nominal_{split}"][k + "_ci95"],
                ],
            }
            for k in keys
        }
    results = {
        "run": cfg.run_name,
        "adapter": str(adapter),
        "constraint_eval": cons,
        "rounds": state["history"],
        "comparison": table,
    }
    (out / "results.json").write_text(json.dumps(results, indent=2))
    log("RESULTS " + json.dumps(table, indent=2))


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--override", action="append", default=[])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--stages", default="demos,constraint,finetune,evaluate")
    a = ap.parse_args()
    cfg = load_cfg(a.config.resolve(), a.override, a.smoke)
    os.environ["MACHIAVELLI_ROOT"] = str(Path(cfg.machiavelli_root).resolve())
    out = Path(cfg.output.dir)
    out.mkdir(parents=True, exist_ok=True)
    from omegaconf import OmegaConf

    (out / "config_resolved.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True))
    log(f"run {cfg.run_name} -> {out}")
    stages = a.stages.split(",")
    if "demos" in stages:
        stage_demos(cfg, out, a.smoke)
    if "constraint" in stages:
        stage_constraint(cfg, out)
    if "finetune" in stages:
        stage_finetune(cfg, out, a.smoke)
    if "evaluate" in stages:
        stage_evaluate(cfg, out)
    log("pipeline finished")
    return 0


if __name__ == "__main__":
    sys.exit(main())
