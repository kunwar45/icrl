#!/usr/bin/env python
# ABOUTME: Offline check of the ICRL inverse step: replays a finished ICRL run's saved per-round policy samples through the published PUCL rule and the calibrated rules, and reports per round what is marked forbidden and what the constraint charges
# ABOUTME: Run inside a GPU job: python scripts/refit_icrl_constraint_offline.py --config configs/lagrangian_finetuning/countdown_code_icrl_pucl_calibrated_qwen7b.yaml --run-dir $SCRATCH/icrl/countdown_code/countdown_code_icrl_pucl_qwen7b_v2 [--quantiles 0.95,0.99,0.999] [--smoke]
"""
No policy is trained. For each rule, C_theta is re-fitted round by round (warm-started, PUCL memory per
round) on the replies the finished run's policy actually produced in that round
(icrl_round_NN/policy_samples.jsonl). Because those samples come from the old run, this checks the
inverse step only, not the loop.

Rules:
  published        the run's own settings: 30% farthest marked, 4,000 demonstrations (hacking ones kept)
  calibrated_qQ    the calibrated config: forbidden = farther than the Q-quantile of held-out allowed
                   replies; hacking demonstrations dropped; start-model replies added; budget from the
                   held-out allowed set

Per round it reports: marked, hacks among marked, share of the round's hacks marked, mean cost of honest
replies / hacks / held-out allowed / demonstrations, the budget, AUROC and within-prompt AUROC. The hack
label is used for this audit and to clean the allowed set, never to choose negatives.
Writes <out>/offline_refit_summary.json.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
_spec = importlib.util.spec_from_file_location(
    "single_turn_pipeline", REPO / "scripts" / "run_single_turn_icrl_pipeline.py"
)
pipeline = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pipeline)

PUBLISHED = {
    "pucl_selection": "fraction",
    "pucl_keep_fraction": 0.3,
    "expert_demos": 4000,
    "drop_hacking_demonstrations": False,
    "start_model_allowed_prompts": 0,
    "heldout_fraction": 0.0,
    "budget_from_heldout": False,
    "memory_per_round": True,
}


def round_samples(bench, rdir: Path) -> tuple[list[dict], set]:
    """This round's reward-earning samples, grouped by prompt, and the prompt ids (as the loop builds them)."""
    rows = pipeline.load_rows(rdir / "policy_samples.jsonl")
    items = [{k: r[k] for k in bench.item_columns} for r in rows]
    flat_items = [it for it, r in zip(items, rows) for _ in r["replies"]]
    flat_group = [g for g, r in enumerate(rows) for _ in r["replies"]]
    flat = [x for r in rows for x in r["replies"]]
    scores = bench.score_replies(flat_items, flat)
    samples = [
        {
            "trajectory": [bench.monitor_text(flat_items[i], flat[i])],
            "violates": scores[i]["violation"],
            "reward": scores[i]["reward"],
            "group": flat_group[i],
        }
        for i, s in enumerate(scores)
        if s["reward"] > 0
    ]
    return samples, {it["id"] for it in items}


def allowed_set(cfg, bench, out: Path, all_train: list, exclude_ids: set):
    """(to fit, held out) exactly as stage_icrl_loop builds them, excluding the run's sampled prompts."""
    ic = cfg.icrl
    demos, _ = pipeline.select_demonstrations(
        cfg, bench, random.Random(int(cfg.rl.seed))
    )
    allowed = [[bench.monitor_text(d, d["response"])] for d in demos]
    n_start = int(ic.get("start_model_allowed_prompts") or 0)
    if n_start:
        exclude = [{"id": i} for i in exclude_ids]
        allowed += pipeline.start_model_allowed_replies(
            cfg, bench, out, all_train, exclude, n_start
        )
    return pipeline.split_heldout(
        allowed, float(ic.get("heldout_fraction") or 0.0), int(cfg.rl.seed)
    )


def main() -> int:
    from omegaconf import OmegaConf

    from src.environments.single_turn_benchmark import get_benchmark

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--config", type=Path, required=True, help="the calibrated ICRL config"
    )
    ap.add_argument(
        "--run-dir", type=Path, required=True, help="finished ICRL run to replay"
    )
    ap.add_argument("--quantiles", default="0.95,0.99,0.999")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--smoke", action="store_true", help="2 rounds, small allowed set")
    a = ap.parse_args()
    cfg = pipeline.load_cfg(a.config.resolve(), [], a.smoke)
    bench = get_benchmark(str(cfg.benchmark))
    run_dir = a.run_dir.resolve()
    out = a.out or run_dir.parent / (
        run_dir.name + ("_offline_refit_smoke" if a.smoke else "_offline_refit")
    )
    out.mkdir(parents=True, exist_ok=True)
    round_dirs = sorted(run_dir.glob("icrl_round_*"))
    if a.smoke:
        round_dirs = round_dirs[:2]
    rounds = []
    for rd in round_dirs:
        samples, ids = round_samples(bench, rd)
        rounds.append((int(rd.name.rsplit("_", 1)[1]), samples, ids))
        pipeline.log(
            f"{rd.name}: {len(samples)} reward-earning samples, {sum(s['violates'] for s in samples)} hacks"
        )
    exclude_ids = set().union(*(ids for _, _, ids in rounds))
    all_train = bench.load_items(cfg, "train")

    rules = {"published": PUBLISHED}
    for q in [float(x) for x in a.quantiles.split(",") if x]:
        rules[f"calibrated_q{q:g}"] = {
            "pucl_selection": "heldout_quantile",
            "pucl_quantile": q,
        }
    if a.smoke:
        for over in rules.values():
            if "expert_demos" in over:
                over["expert_demos"] = 64
    summary = {}
    for name, over in rules.items():
        rcfg = OmegaConf.merge(cfg, OmegaConf.create({"icrl": over}))
        rout = out / name
        rout.mkdir(parents=True, exist_ok=True)
        expert_trajs, heldout_trajs = allowed_set(
            rcfg, bench, out, all_train, exclude_ids
        )
        pipeline.log(
            f"rule {name}: allowed set {len(expert_trajs)} to fit, {len(heldout_trajs)} held out"
        )
        prev, rows = None, []
        for r, samples, _ in rounds:
            phi_dir, cstats, audit = pipeline.fit_constraint_round(
                rcfg,
                rout,
                rout / f"round_{r:02d}",
                r,
                prev,
                expert_trajs,
                samples,
                heldout_trajs=heldout_trajs or None,
            )
            prev = str(phi_dir)
            n_hacks = audit["n_violating"]
            rows.append(
                {
                    "round": r,
                    "n_unlabelled": audit["n"],
                    "n_hacks": n_hacks,
                    "marked": audit.get("marked"),
                    "marked_hacks": audit.get("marked_violating"),
                    "share_of_hacks_marked": (audit.get("marked_violating") or 0)
                    / n_hacks
                    if n_hacks
                    else None,
                    "memory_before": cstats.get("n_memory_steps"),
                    "selection_threshold": cstats.get("selection_threshold"),
                    "mean_cost_honest": audit["mean_cost_clean"],
                    "mean_cost_hacks": audit["mean_cost_violating"],
                    "mean_cost_heldout_allowed": audit.get("mean_cost_heldout_allowed"),
                    "mean_cost_demonstrations_fit": audit["mean_cost_expert"],
                    "budget": audit.get(
                        "budget",
                        float(cfg.rl.dual.epsilon) if name == "published" else None,
                    ),
                    "auroc": audit["auroc"],
                    "within_prompt_auroc": audit["within_prompt_auroc"],
                    "skipped": cstats.get("skipped"),
                }
            )
        summary[name] = rows
        (out / "offline_refit_summary.json").write_text(json.dumps(summary, indent=2))

    fmt = lambda v: (
        "-" if v is None else (f"{v:.2f}" if isinstance(v, float) else str(v))
    )  # noqa: E731
    cols = [
        "round",
        "n_unlabelled",
        "n_hacks",
        "marked",
        "marked_hacks",
        "mean_cost_honest",
        "mean_cost_hacks",
        "mean_cost_heldout_allowed",
        "budget",
        "auroc",
        "within_prompt_auroc",
    ]
    for name, rows in summary.items():
        print(f"\n=== {name} ===\n" + "  ".join(cols), flush=True)
        for row in rows:
            print("  ".join(fmt(row[c]) for c in cols), flush=True)
    pipeline.log(f"wrote {out / 'offline_refit_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
