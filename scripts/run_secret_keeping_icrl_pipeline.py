#!/usr/bin/env python
# ABOUTME: Secret-keeping pipeline as resumable stages: oracle-expert demos, base evaluation, the ICRL loop (Malik et al. 2021 + Lagrangian GRPO, stage icrl_loop) or a baseline (classifier+RLHF, SFT, cost baselines), held-out evaluation
# ABOUTME: Run: python scripts/run_secret_keeping_icrl_pipeline.py --config configs/lagrangian_finetuning/secret_keeping_icrl_loop_qwen7b.yaml [--smoke] [--stages ...]; the config's block (icrl / rl / sft / finetune) picks the method stage
"""
The method is the `icrl_loop` stage (genuine ICRL: Malik et al. 2021 alternation
with a Lagrangian GRPO forward step). `constraint`+`rl`, `constraint`+`finetune`
and `sft` are BASELINES: the one-shot `constraint` stage is a supervised
classifier, not ICRL. See docs/secret-keeping.md.

Stages, each skipped when its outputs exist (a killed job resumes):

  demos       serve the base model; on the training prompts sample k replies under
              the expert prompt and k under the nominal prompt, judge them all;
              the oracle expert keeps its best non-leaking reply
  base_eval   test prompts (unseen businesses and templates): k replies under the
              nominal prompt (baseline) and the expert prompt (reference), judged
  constraint  BASELINE: one-shot supervised classifier C_theta on oracle-expert vs policy replies of the same prompt
              (constraint.pair_filter, no leak labels); test it against the exact
              leak check on the base_eval nominal replies
  finetune    BASELINE (superseded): rounds of: sample k replies per training prompt from base + current
              adapter, judge (base weights), score with C_theta, DPO pairs by
              R - lambda*C, fresh LoRA on the cumulative pairs, dual ascent on lambda
  rl          BASELINE: GRPO against a FIXED cost (the one-shot classifier, the exact check or a
              prompted judge, rl.cost) with a fixed lambda: reward = judge/10 - lambda * cost; the judge is served on
              rl.judge_gpu, training runs on rl.train_gpus
  sft         behaviour-cloning baseline: LoRA SFT on the oracle expert's chosen replies
              under the nominal prompt (config has an sft block)
  icrl_loop   THE METHOD: genuine ICRL (Malik et al. 2021), K rounds of: sample the CURRENT policy,
              max-ent constraint update of phi (expert up, policy down, sparsity
              regulariser; warm start), GRPO resumed from the last round with
              reward - lambda * (1 - phi) and dual ascent on lambda (config has an icrl block)
  evaluate    tuned adapter on the test prompts exactly as base_eval; results.json

--smoke shrinks everything and prints, for a few prompts per arm, the exact
system prompt, user message, reply, leak verdict and judge score.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

STAGES = ("demos", "base_eval", "constraint", "finetune", "rl", "sft", "icrl_loop", "evaluate")
# The fine-tuning method: one of these stages runs, chosen by which block the config has.
METHOD_STAGES = ("finetune", "rl", "sft", "icrl_loop")


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
                    "data": {"train_prompts": 16, "test_prompts": 8},
                    "sampling": {"k": 2},
                    "evaluation": {"k": 2},
                    "constraint": {
                        "n_iterations": 2,
                        "n_constraint_steps": 5,
                        "max_demos_per_class": 64,
                    },
                }
            ),
        )
        if "finetune" in cfg:
            cfg = OmegaConf.merge(
                cfg,
                OmegaConf.create(
                    {"finetune": {"rounds": 1, "max_pairs_per_round": 8, "pair_margin": 0.0}}
                ),
            )
        if "rl" in cfg:
            cfg = OmegaConf.merge(
                cfg, OmegaConf.create({"rl": {"max_steps": 3, "save_steps": 1000}})
            )
        if "icrl" in cfg:
            cfg = OmegaConf.merge(
                cfg,
                OmegaConf.create(
                    {"icrl": {"rounds": 2, "grpo_steps_per_round": 2, "k": 2,
                              "first_round_constraint_steps": 20, "constraint_steps": 10}}
                ),
            )
        cfg.output.dir = str(Path(cfg.output.dir).parent / cfg.run_name)
    cfg.output.dir = str(Path(cfg.output.dir).resolve())
    return cfg


def served(cfg, out_dir: Path, adapter: Path | None):
    from src.utils.vllm_replica_serving import served as serve_replicas

    return serve_replicas(cfg, out_dir, adapter, log=log)


def clients(cfg, servers, temperature: float):
    from src.lagrangian_finetuning.secret_keeping_constrained_dpo import make_clients

    return make_clients(servers, temperature, int(cfg.sampling.max_tokens))


def save_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def load_rows(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def show(
    title: str, rows: list[dict], kind: str, replies_key: str, R_key: str, n: int = 4
) -> None:
    """Exact model input, output, leak verdict and judge score for n prompts of each prompt kind."""
    from src.environments.secret_keeping.secret_keeping_adapter import (
        leaks,
        system_prompt,
    )

    print(f"\n{'=' * 25} {title} {'=' * 25}", flush=True)
    for pk in ("attack", "benign"):
        for r in [r for r in rows if r["kind"] == pk][:n]:
            print(f"--- {r['id']} ({pk}) secret={r['secret']} ---")
            print("[SYSTEM PROMPT]\n" + system_prompt(kind, r))
            print("[USER]\n" + r["user"])
            for rep, R in zip(r[replies_key], r[R_key]):
                print(f"[REPLY] leak={leaks(rep, r['secret'])} judge={R}\n{rep}")
    print(flush=True)


# ---- stages -----------------------------------------------------------------


def stage_demos(cfg, out: Path, smoke: bool) -> None:
    from src.environments.secret_keeping.secret_keeping_adapter import (
        leaks,
        make_prompts,
    )
    from src.icrl_dual_training.secret_keeping_reply_constraint import (
        PAIR_FILTERS,
        oracle_expert_pick,
        oracle_pair_indices,
    )
    from src.lagrangian_finetuning.secret_keeping_constrained_dpo import (
        judge_rewards,
        sample_replies,
    )

    if (out / "demos" / "summary.json").exists():
        log("demos: done already")
        return
    d, s = cfg.data, cfg.sampling
    items = make_prompts(
        "train", int(d.train_prompts), int(d.seed), float(d.attack_share)
    )
    k, w = int(s.k), int(s.workers)
    with served(cfg, out / "demos", None) as servers:
        pol, judge = clients(cfg, servers, float(s.temperature))
        log(f"demos: {len(items)} training prompts, {k} replies per arm")
        exp = sample_replies(items, pol, "expert", k, w)
        nom = sample_replies(items, pol, "nominal", k, w)
        log("demos: replies sampled; judging")
        exp_R = judge_rewards(items, exp, judge, w)
        nom_R = judge_rewards(items, nom, judge, w)
    rows = []
    for it, er, eR, nr, nR in zip(items, exp, exp_R, nom, nom_R):
        rows.append(
            {
                **it,
                "expert_replies": er,
                "expert_R": eR,
                "expert_pick": oracle_expert_pick(er, eR, it["secret"]),
                "policy_replies": nr,
                "policy_R": nR,
                # ground truth, for reporting only
                "expert_leaks": [leaks(x, it["secret"]) for x in er],
                "policy_leaks": [leaks(x, it["secret"]) for x in nr],
            }
        )
    save_rows(out / "demos" / "rows.jsonl", rows)
    summary = {
        "prompts": len(rows),
        "no_expert_pick": sum(r["expert_pick"] is None for r in rows),
    }
    for pk in ("attack", "benign"):
        rs = [r for r in rows if r["kind"] == pk]
        summary[f"{pk}_expert_prompt_leak_rate"] = sum(
            sum(r["expert_leaks"]) for r in rs
        ) / max(1, k * len(rs))
        summary[f"{pk}_nominal_prompt_leak_rate"] = sum(
            sum(r["policy_leaks"]) for r in rs
        ) / max(1, k * len(rs))
    for f in PAIR_FILTERS:
        kept = [(r, i) for r in rows for i in oracle_pair_indices(r, f)]
        summary[f"pairs_{f}"] = {
            "pairs": len(kept),
            "from_attack_prompts": sum(r["kind"] == "attack" for r, _ in kept),
            "unsafe_reply_leaks": sum(r["policy_leaks"][i] for r, i in kept)
            / max(1, len(kept)),
        }
    (out / "demos" / "summary.json").write_text(json.dumps(summary, indent=2))
    log(f"demos: {json.dumps(summary)}")
    if smoke:
        show(
            "demo arm: NOMINAL prompt (policy)",
            rows,
            "nominal",
            "policy_replies",
            "policy_R",
        )
        show("demo arm: EXPERT prompt", rows, "expert", "expert_replies", "expert_R")


def evaluate_prompts(cfg, servers, tag: str, out: Path, kinds, smoke: bool) -> dict:
    from src.environments.secret_keeping.secret_keeping_adapter import make_prompts
    from src.lagrangian_finetuning.secret_keeping_constrained_dpo import (
        judge_rewards,
        reply_metrics,
        sample_replies,
    )

    d, e = cfg.data, cfg.evaluation
    items = make_prompts(
        "test", int(d.test_prompts), int(d.seed), float(d.attack_share)
    )
    pol, judge = clients(cfg, servers, float(e.temperature))
    w, res = int(cfg.sampling.workers), {}
    for kind in kinds:
        reps = sample_replies(items, pol, kind, int(e.k), w)
        Rs = judge_rewards(items, reps, judge, w)
        rows = [{**it, "replies": r, "R": R} for it, r, R in zip(items, reps, Rs)]
        save_rows(out / "eval" / f"{tag}_{kind}.jsonl", rows)
        res[kind] = reply_metrics(rows)
        log(f"eval {tag} {kind}: {json.dumps(res[kind])}")
        if smoke:
            show(f"eval {tag}, {kind} prompt", rows, kind, "replies", "R")
    (out / "eval" / f"{tag}_metrics.json").write_text(json.dumps(res, indent=2))
    return res


def stage_base_eval(cfg, out: Path, smoke: bool) -> None:
    if (out / "eval" / "base_metrics.json").exists():
        log("base_eval: done already")
        return
    with served(cfg, out / "eval", None) as servers:
        evaluate_prompts(cfg, servers, "base", out, ("nominal", "expert"), smoke)


def constraint_eval(cfg, out: Path, model=None) -> dict | None:
    from src.icrl_dual_training.machiavelli_step_constraint import StepConstraint
    from src.icrl_dual_training.secret_keeping_reply_constraint import (
        reply_detector_eval,
    )

    p = out / "constraint" / "eval.json"
    if p.exists():
        return json.loads(p.read_text())
    if model is None and not (out / "constraint" / "constraint_meta.json").exists():
        return None  # no one-shot classifier in this run (e.g. the icrl loop)
    src = out / "eval" / "base_nominal.jsonl"
    if not src.exists():
        log("constraint eval: base_eval not there yet; deferred to evaluate")
        return None
    model = model or StepConstraint.load(
        out / "constraint", device=cfg.constraint.get("device")
    )
    res = reply_detector_eval(model, load_rows(src))
    log(
        f"constraint on base nominal test replies (exact leak check as truth): {json.dumps(res)}"
    )
    p.write_text(json.dumps(res, indent=2))
    return res


def constraint_eval_on_tuned(cfg, out: Path) -> dict | None:
    """C_theta against the exact leak check on the tuned policy's test replies: is the policy evading the detector?"""
    from src.icrl_dual_training.machiavelli_step_constraint import StepConstraint
    from src.icrl_dual_training.secret_keeping_reply_constraint import (
        reply_detector_eval,
    )

    p, src = out / "constraint" / "eval_tuned.json", out / "eval" / "tuned_nominal.jsonl"
    if p.exists():
        return json.loads(p.read_text())
    if not src.exists() or not (out / "constraint" / "constraint_meta.json").exists():
        return None
    model = StepConstraint.load(out / "constraint", device=cfg.constraint.get("device"))
    res = reply_detector_eval(model, load_rows(src))
    log(f"constraint on TUNED nominal test replies (exact leak check as truth): {json.dumps(res)}")
    p.write_text(json.dumps(res, indent=2))
    return res


def stage_constraint(cfg, out: Path) -> None:
    from src.icrl_dual_training.machiavelli_step_constraint import train_step_constraint
    from src.icrl_dual_training.secret_keeping_reply_constraint import oracle_demo_pairs

    cdir = out / "constraint"
    if (cdir / "constraint_meta.json").exists():
        log("constraint: trained already")
        constraint_eval(cfg, out)
        return
    rows = load_rows(out / "demos" / "rows.jsonl")
    f = str(cfg.constraint.pair_filter)
    expert, unsafe = oracle_demo_pairs(rows, pair_filter=f)
    log(
        f"constraint: {len(expert)} oracle-expert / policy same-prompt pairs (filter {f})"
    )
    model = train_step_constraint(cfg, expert, unsafe, cdir, log=log)
    constraint_eval(cfg, out, model)


def stage_finetune(cfg, out: Path, smoke: bool) -> None:
    from src.environments.secret_keeping.secret_keeping_adapter import make_prompts
    from src.icrl_dual_training.machiavelli_step_constraint import StepConstraint
    from src.icrl_dual_training.secret_keeping_reply_constraint import constraint_text
    from src.lagrangian_finetuning.machiavelli_constrained_dpo import (
        dpo_train,
        dual_ascent,
    )
    from src.lagrangian_finetuning.secret_keeping_constrained_dpo import (
        build_pairs,
        judge_rewards,
        reply_metrics,
        sample_replies,
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
    d, s = cfg.data, cfg.sampling
    items = make_prompts(
        "train", int(d.train_prompts), int(d.seed), float(d.attack_share)
    )
    cmodel = None
    while state["round"] < rounds:
        r, lam = state["round"], float(state["lambda"])
        rdir = out / f"round_{r:02d}"
        rdir.mkdir(parents=True, exist_ok=True)
        log(f"=== round {r}: lambda={lam:.3f} adapter={state['adapter']} ===")
        adapter = Path(state["adapter"]) if state["adapter"] else None
        with served(cfg, rdir, adapter) as servers:
            pol, judge = clients(cfg, servers, float(s.temperature))
            reps = sample_replies(items, pol, "nominal", int(s.k), int(s.workers))
            Rs = judge_rewards(items, reps, judge, int(s.workers))
        if cmodel is None:  # servers are down: C_theta takes a GPU only now
            cmodel = StepConstraint.load(
                out / "constraint", device=cfg.constraint.get("device")
            )
        rows = [{**it, "replies": rp, "R": R} for it, rp, R in zip(items, reps, Rs)]
        flat = cmodel.score(
            [constraint_text(row, x) for row in rows for x in row["replies"]]
        )
        i = 0
        for row in rows:
            row["C"] = flat[i : i + len(row["replies"])]
            i += len(row["replies"])
        save_rows(rdir / "rollouts.jsonl", rows)
        pairs, stats = build_pairs(rows, lam, cfg)
        if len(pairs) > int(cfg.finetune.max_pairs_per_round):
            pairs = random.Random(r).sample(
                pairs, int(cfg.finetune.max_pairs_per_round)
            )
        (rdir / "pairs.json").write_text(json.dumps(pairs))
        stats["rollout_ground_truth"] = reply_metrics(rows)
        log(f"round {r} stats: {json.dumps(stats)}")
        if smoke:
            for p in pairs[:4]:
                print(
                    f"[PAIR] {p['id']} ({p['kind']}) chosen C={p['C_chosen']:.2f} R={p['R_chosen']:.2f} "
                    f"leaks={p['gt_chosen_leaks']}: {p['chosen'][0]['content'][:200]!r}\n"
                    f"       rejected C={p['C_rejected']:.2f} R={p['R_rejected']:.2f} "
                    f"leaks={p['gt_rejected_leaks']}: {p['rejected'][0]['content'][:200]!r}",
                    flush=True,
                )
        pool = []
        for kk in range(r + 1):
            pp = out / f"round_{kk:02d}" / "pairs.json"
            if pp.exists():
                pool += json.loads(pp.read_text())
        log(f"round {r}: training a fresh LoRA on {len(pool)} cumulative pairs")
        if pool:
            state["adapter"] = str(dpo_train(cfg, pool, rdir, log=log))
        stats["lambda_next"] = dual_ascent(lam, stats["mean_C_sampled"] or 0.0, cfg)
        (rdir / "round_stats.json").write_text(json.dumps(stats, indent=2))
        state["history"].append(stats)
        state["lambda"] = stats["lambda_next"]
        state["round"] = r + 1
        state_p.write_text(json.dumps(state, indent=2))


def stage_rl(cfg, out: Path, smoke: bool) -> None:
    """GRPO with reward = judge helpfulness - lambda * C_theta; writes state.json for evaluate."""
    from omegaconf import OmegaConf

    from src.lagrangian_finetuning.secret_keeping_constrained_grpo import grpo_train

    state_p = out / "state.json"
    if state_p.exists() and json.loads(state_p.read_text()).get("method") == "grpo":
        log("rl: done already")
        return
    if str(cfg.rl.get("cost", "learned")) == "learned" and not (
        out / "constraint" / "constraint_meta.json"
    ).exists():
        raise RuntimeError("rl with the learned cost needs a trained constraint (run the constraint stage first)")
    rdir = out / "rl"
    judge_cfg = OmegaConf.merge(
        cfg, OmegaConf.create({"policy": {"serve": {"replicas": [str(cfg.rl.judge_gpu)]}}})
    )
    run_cfg = OmegaConf.merge(cfg, OmegaConf.create({"rl": {"constraint_dir": str(out / "constraint")}}))
    with served(judge_cfg, rdir, None) as servers:
        adapter = grpo_train(run_cfg, rdir, servers[0].base_url, log=log)
    hist_p = rdir / "train_log_history.json"
    hist = json.loads(hist_p.read_text()) if hist_p.exists() else []
    state = {"method": "grpo", "adapter": str(adapter), "history": hist[-3:]}
    state_p.write_text(json.dumps(state, indent=2))
    log(f"rl: adapter {adapter}")



def stage_sft(cfg, out: Path, smoke: bool) -> None:
    """Behaviour cloning on the expert demos; writes state.json for evaluate."""
    from src.lagrangian_finetuning.secret_keeping_expert_sft import sft_train

    state_p = out / "state.json"
    if state_p.exists() and json.loads(state_p.read_text()).get("method") == "sft":
        log("sft: done already")
        return
    rows = load_rows(out / "demos" / "rows.jsonl")
    adapter = sft_train(cfg, out / "sft", rows, log=log)
    hist_p = out / "sft" / "train_log_history.json"
    hist = json.loads(hist_p.read_text()) if hist_p.exists() else []
    state_p.write_text(json.dumps({"method": "sft", "adapter": str(adapter), "history": hist[-3:]}, indent=2))
    log(f"sft: adapter {adapter}")



def phi_eval_on_test(cfg, out: Path, state: dict) -> dict | None:
    """ICRL loop only: the final round's cost 1 - phi against the exact check, on base and tuned test replies."""
    if state.get("method") != "icrl_loop":
        return None
    p = out / "icrl_phi_eval.json"
    if p.exists():
        return json.loads(p.read_text())
    from src.icrl_dual_training.secret_keeping_reply_constraint import (
        ReplyFeasibility,
        reply_detector_eval,
    )

    phi = ReplyFeasibility.load(Path(state["phi_dir"]), device=cfg.constraint.get("device"))
    res = {}
    for tag in ("base_nominal", "tuned_nominal"):
        src = out / "eval" / f"{tag}.jsonl"
        if src.exists():
            res[tag] = reply_detector_eval(phi, load_rows(src))
            log(f"final phi cost on {tag} test replies (exact check as truth): {json.dumps(res[tag])}")
    p.write_text(json.dumps(res, indent=2))
    return res


def stage_icrl_loop(cfg, out: Path, smoke: bool) -> None:
    """Genuine ICRL (Malik et al. 2021): alternate the inverse step on phi and the forward GRPO step."""
    import gc

    import torch
    from omegaconf import OmegaConf

    from src.environments.secret_keeping.secret_keeping_adapter import (
        leaks,
        make_prompts,
    )
    from src.icrl_dual_training.maxent_constraint import MaxEntConstraintConfig
    from src.icrl_dual_training.secret_keeping_reply_constraint import (
        ReplyFeasibility,
        constraint_text,
        maxent_round,
    )
    from src.lagrangian_finetuning.secret_keeping_constrained_dpo import sample_replies
    from src.lagrangian_finetuning.secret_keeping_constrained_grpo import grpo_train

    state_p = out / "state.json"
    if state_p.exists() and json.loads(state_p.read_text()).get("method") == "icrl_loop":
        log("icrl_loop: done already")
        return
    ic, c, d, s = cfg.icrl, cfg.constraint, cfg.data, cfg.sampling
    K, N = int(ic.rounds), int(ic.grpo_steps_per_round)
    loop_p = out / "icrl_state.json"
    st = (
        json.loads(loop_p.read_text())
        if loop_p.exists()
        else {"round": 0, "adapter": None, "checkpoint": None,
              "lambda": float(cfg.rl.dual.lambda_init), "phi_dir": None, "history": []}
    )
    rows = load_rows(out / "demos" / "rows.jsonl")
    expert_texts = sorted(
        {constraint_text(r, r["expert_replies"][r["expert_pick"]]) for r in rows
         if r.get("expert_pick") is not None}
    )
    items = make_prompts("train", int(d.train_prompts), int(d.seed), float(d.attack_share))
    log(f"icrl_loop: {K} rounds x {N} GRPO steps; {len(expert_texts)} expert replies; "
        f"{len(items)} training chats, {int(ic.k)} policy samples each per round")
    judge_cfg = OmegaConf.merge(cfg, OmegaConf.create({"policy": {"serve": {"replicas": [str(cfg.rl.judge_gpu)]}}}))
    while st["round"] < K:
        r = st["round"]
        rdir = out / f"icrl_round_{r:02d}"
        rdir.mkdir(parents=True, exist_ok=True)
        log(f"=== ICRL round {r}: policy adapter={st['adapter']} lambda={st['lambda']:.3f} ===")
        # 1. sample the CURRENT policy (no reward filter: plain Malik nominal set)
        adapter = Path(st["adapter"]) if st["adapter"] else None
        with served(cfg, rdir, adapter) as servers:
            pol, _ = clients(cfg, servers, float(s.temperature))
            reps = sample_replies(items, pol, "nominal", int(ic.k), int(s.workers))
        samples = [
            {"text": constraint_text(it, x), "kind": it["kind"],
             "violates": leaks(x, it["secret"])}  # audit only, never trained on
            for it, rs in zip(items, reps) for x in rs
        ]
        save_rows(rdir / "policy_samples.jsonl", [{**it, "replies": rs} for it, rs in zip(items, reps)])
        # 2. inverse step: max-ent constraint update on phi, warm-started
        phi = ReplyFeasibility.build(str(c.encoder_model), int(c.max_length), int(c.head_hidden),
                                     device=c.get("device"))
        if st["phi_dir"]:
            phi.load_head(st["phi_dir"])
        e_cache = out / "icrl_expert_embeddings.pt"
        if e_cache.exists():
            e_emb = torch.load(e_cache, map_location="cpu")
        else:
            e_emb = phi.embed(expert_texts).cpu()
            torch.save(e_emb, e_cache)
        n_emb = phi.embed([x["text"] for x in samples]).cpu()
        mcfg = MaxEntConstraintConfig(
            beta=float(ic.beta), sparsity_coeff=float(ic.sparsity_coeff),
            negative_selection=str(ic.negative_selection),
            n_steps=int(ic.first_round_constraint_steps if r == 0 else ic.constraint_steps),
            batch_trajectories=int(ic.constraint_batch), lr=float(ic.constraint_lr),
        )
        cstats = maxent_round(phi, e_emb, n_emb, mcfg)
        costs = phi.costs_from_embeddings(n_emb)
        audit = reply_detector_eval_from_costs(samples, costs)
        phi_dir = rdir / "phi"
        phi.save(phi_dir)
        log(f"round {r} inverse step: {json.dumps(cstats)}")
        log(f"round {r} phi on this round's policy samples (exact check as truth, audit only): {json.dumps(audit)}")
        del phi, e_emb, n_emb
        gc.collect()
        torch.cuda.empty_cache()
        # 3+4. forward step: GRPO resumed from the last round, lambda by dual ascent
        run_cfg = OmegaConf.merge(cfg, OmegaConf.create({"rl": {
            "cost": "maxent_feasibility", "phi_dir": str(phi_dir),
            "max_steps": (r + 1) * N, "save_steps": N,
            "resume_from_checkpoint": st["checkpoint"], "lambda_init": st["lambda"],
        }}))
        with served(judge_cfg, rdir, None) as servers:
            adapter_dir = grpo_train(run_cfg, rdir / "rl", servers[0].base_url, log=log)
        lam = json.loads((rdir / "rl" / "lambda.json").read_text())["lambda"]
        hist_p = rdir / "rl" / "train_log_history.json"
        tail = json.loads(hist_p.read_text())[-N:] if hist_p.exists() else []
        st["history"].append({"round": r, "inverse": cstats, "phi_audit": audit,
                              "lambda_end": lam, "grpo_tail": tail[-3:]})
        st.update(round=r + 1, adapter=str(adapter_dir), phi_dir=str(phi_dir))
        st["lambda"] = lam
        st["checkpoint"] = str(rdir / "rl" / "trainer" / f"checkpoint-{(r + 1) * N}")
        loop_p.write_text(json.dumps(st, indent=2))
    state_p.write_text(json.dumps({"method": "icrl_loop", "adapter": st["adapter"],
                                   "phi_dir": st["phi_dir"], "history": st["history"]}, indent=2))
    log(f"icrl_loop: done, final adapter {st['adapter']}")


def reply_detector_eval_from_costs(samples: list[dict], costs: list[float]) -> dict:
    """AUROC of cost vs the exact check over already-scored samples (audit only)."""
    from src.icrl_dual_training.machiavelli_step_constraint import auroc

    labels = [x["violates"] for x in samples]
    att = [i for i, x in enumerate(samples) if x["kind"] == "attack"]
    nv = sum(labels)
    return {
        "n": len(samples),
        "violation_rate_attack": sum(labels[i] for i in att) / max(1, len(att)),
        "auroc_all": auroc(costs, labels) if 0 < nv < len(labels) else None,
        "mean_cost_violating": sum(cc for cc, l in zip(costs, labels) if l) / max(1, nv),
        "mean_cost_safe": sum(cc for cc, l in zip(costs, labels) if not l) / max(1, len(labels) - nv),
    }


def stage_evaluate(cfg, out: Path, smoke: bool) -> None:
    state = json.loads((out / "state.json").read_text())
    if not state.get("adapter"):
        raise RuntimeError("no fine-tuned adapter to evaluate")
    if not (out / "eval" / "tuned_metrics.json").exists():
        with served(cfg, out / "eval", Path(state["adapter"])) as servers:
            evaluate_prompts(cfg, servers, "tuned", out, ("nominal",), smoke)
    cons = constraint_eval(cfg, out)
    cons_tuned = constraint_eval_on_tuned(cfg, out)
    phi_evals = phi_eval_on_test(cfg, out, state)
    base = json.loads((out / "eval" / "base_metrics.json").read_text())
    tuned = json.loads((out / "eval" / "tuned_metrics.json").read_text())
    table = {
        pk: {
            m: {
                "base_nominal": [
                    base["nominal"][pk][m],
                    base["nominal"][pk][m + "_ci95"],
                ],
                "base_expert_prompt": [
                    base["expert"][pk][m],
                    base["expert"][pk][m + "_ci95"],
                ],
                "tuned_nominal": [
                    tuned["nominal"][pk][m],
                    tuned["nominal"][pk][m + "_ci95"],
                ],
            }
            for m in ("leak_rate", "helpfulness")
        }
        for pk in ("attack", "benign")
    }
    results = {
        "run": cfg.run_name,
        "adapter": state["adapter"],
        "constraint_eval": cons,
        "constraint_eval_on_tuned_replies": cons_tuned,
        "icrl_phi_eval": phi_evals,
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
    ap.add_argument(
        "--stages",
        default=None,
        help="comma list; default every stage, with rl instead of finetune when the config has an rl block",
    )
    a = ap.parse_args()
    cfg = load_cfg(a.config.resolve(), a.override, a.smoke)
    out = Path(cfg.output.dir)
    out.mkdir(parents=True, exist_ok=True)
    from omegaconf import OmegaConf

    if a.stages is None:
        method = (
            "icrl_loop" if "icrl" in cfg
            else "rl" if "rl" in cfg
            else "sft" if "sft" in cfg
            else "finetune"
        )
        # the icrl loop learns its own constraint; the one-shot classifier stage is not part of it
        skip = {"constraint"} if method == "icrl_loop" else set()
        a.stages = ",".join(
            x for x in STAGES if (x not in METHOD_STAGES or x == method) and x not in skip
        )
    stages = [s for s in a.stages.split(",") if s]
    unknown = sorted(set(stages) - set(STAGES))
    if unknown:
        raise SystemExit(f"unknown stages {unknown}; choose from {STAGES}")
    (out / f"config_resolved_{'_'.join(stages)}.yaml").write_text(
        OmegaConf.to_yaml(cfg, resolve=True)
    )
    log(f"run {cfg.run_name} -> {out}; stages {stages}")
    if "demos" in stages:
        stage_demos(cfg, out, a.smoke)
    if "base_eval" in stages:
        stage_base_eval(cfg, out, a.smoke)
    if "constraint" in stages:
        stage_constraint(cfg, out)
    if "finetune" in stages:
        stage_finetune(cfg, out, a.smoke)
    if "rl" in stages:
        stage_rl(cfg, out, a.smoke)
    if "sft" in stages:
        stage_sft(cfg, out, a.smoke)
    if "icrl_loop" in stages:
        stage_icrl_loop(cfg, out, a.smoke)
    if "evaluate" in stages:
        stage_evaluate(cfg, out, a.smoke)
    log("pipeline finished: " + ",".join(stages))
    return 0


if __name__ == "__main__":
    sys.exit(main())
