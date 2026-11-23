#!/usr/bin/env python
# ABOUTME: End-to-end pipeline for single-turn benchmarks behind the benchmark seam (Countdown-Code): SFT priming, then one arm -- no intervention, VARL, or ICRL with a learned constraint -- and evaluation on held-out prompts
# ABOUTME: Run inside a GPU job: python scripts/run_single_turn_icrl_pipeline.py --config configs/lagrangian_finetuning/countdown_code_icrl_pucl_qwen7b.yaml [--smoke] [--stages base_eval,icrl_loop,evaluate]
"""
The arm is chosen by the config:

  sft block only            stage sft_prime: SFT on the demonstrations, merged
                            into the weights every arm starts from
  rl.signal task_only       stage train: GRPO on the benchmark's training reward
  rl.signal verifier_gated  stage train: VARL (reward x discriminator, the
                            discriminator trained alongside the policy)
  icrl block                stage icrl_loop: each round samples the policy,
                            re-fits the constraint C_theta from demonstrations
                            vs. those samples, then runs GRPO on
                            reward - lambda * (1 - C_theta)

The benchmark's violation label and true reward are used only for audit lines
and for evaluation. Every stage is skipped if its output already exists.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

STAGES = (
    "sft_prime",
    "base_eval",
    "train",
    "icrl_loop",
    "icrl_graft_loop",
    "evaluate",
    "evaluate_rounds",  # opt-in only: test-set eval of each saved ICRL round
)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_cfg(path: Path, overrides: list[str], smoke: bool):
    from omegaconf import OmegaConf

    cfg = OmegaConf.merge(OmegaConf.load(path), OmegaConf.from_dotlist(overrides))
    if smoke:
        small = {
            "run_name": cfg.run_name + "_smoke",
            "data": {"train_prompts": 16, "test_prompts": 5},
            "evaluation": {"k": 2},
        }
        if "rl" in cfg:
            # 3 GPUs x per_device_batch 2 x grad_accum 4 = 24 replies = 3 prompts per step
            small["rl"] = {
                "max_steps": 3,
                "save_steps": 1000,
                "per_device_batch": 2,
                "grad_accum": 4,
                "steps_per_generation": 8,
            }
        if "icrl" in cfg:
            small["icrl"] = {
                "rounds": 2,
                "grpo_steps_per_round": 2,
                "k": 2,
                "sample_prompts": 8,
                "expert_demos": 64,
                "first_round_constraint_steps": 20,
                "constraint_steps": 10,
                "min_nominal": 4,
            }
            if cfg.icrl.get("start_model_allowed_prompts"):
                small["icrl"]["start_model_allowed_prompts"] = 8
        if "varl" in cfg:
            small["varl"] = {"warmup_steps": 1}
        if "graft" in cfg:
            # no learning-rate warm-up in 3 steps, and enough anchor prompts that some are sampled
            small["graft"] = {
                "optimizer": {"warmup_steps": 0},
                "anchor": {"fraction": 0.34 if cfg.graft.anchor.fraction > 0 else 0.0},
            }
        if "sft" in cfg:
            small["sft"] = {
                "max_steps": 4,
                "merged_dir": str(cfg.sft.merged_dir) + "_smoke",
            }
            small["data"]["demonstrations"] = 64
        cfg = OmegaConf.merge(cfg, OmegaConf.create(small))
        cfg.output.dir = str(Path(cfg.output.dir).parent / cfg.run_name)
    cfg.output.dir = str(Path(cfg.output.dir).resolve())
    return cfg


def save_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def served(cfg, out_dir: Path, adapter: Path | None, replicas: list[str] | None = None):
    from omegaconf import OmegaConf

    from src.utils.vllm_replica_serving import served as serve_replicas

    if replicas is not None:
        cfg = OmegaConf.merge(
            cfg, OmegaConf.create({"policy": {"serve": {"replicas": replicas}}})
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    return serve_replicas(cfg, out_dir, adapter, log=log)


def sample_replies(
    cfg, bench, servers, items: list[dict], k: int, temperature: float
) -> list[list[str]]:
    """k replies per prompt from the served policy (one batched request per prompt)."""
    import itertools
    import threading
    from concurrent.futures import ThreadPoolExecutor

    from src.environments.machiavelli.rollout_collector import OpenAIChatPolicy

    s = cfg.sampling
    pols = [
        OpenAIChatPolicy(
            x.base_url,
            x.served_name,
            temperature=temperature,
            max_tokens=int(s.max_tokens),
            timeout_s=float(s.get("timeout_s", 1800)),
        )
        for x in servers
    ]
    cyc, lock = itertools.cycle(pols), threading.Lock()

    def one(it):
        with lock:
            pol = next(cyc)
        sys_msg, user_msg = bench.chat_messages(it)
        try:
            return pol.sample(sys_msg["content"], user_msg["content"], 0, k)
        except (
            Exception
        ) as e:  # a timed-out prompt counts as k empty replies, not a crash
            log(f"sampling failed for {it.get('id')}: {type(e).__name__}: {e}")
            return [""] * k

    with ThreadPoolExecutor(max_workers=int(s.workers)) as pool:
        return list(pool.map(one, items))


def boot_ci(per_prompt: list[float], n: int = 1000) -> list[float]:
    import numpy as np

    if len(per_prompt) < 2:
        return [float(np.mean(per_prompt)) if per_prompt else 0.0] * 2
    rng, a = np.random.default_rng(0), np.array(per_prompt)
    b = [rng.choice(a, size=len(a)).mean() for _ in range(n)]
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def auroc(scores: list[float], labels: list[int]) -> float | None:
    """P(a positive scores above a negative); None when one class is missing."""
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    if not pos or not neg:
        return None
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks, i = [0.0] * len(scores), 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        for t in range(i, j + 1):
            ranks[order[t]] = (i + j) / 2 + 1
        i = j + 1
    rank_pos = sum(r for r, y in zip(ranks, labels) if y)
    return (rank_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def within_prompt_auroc(
    scores: list[float], labels: list[int], groups: list
) -> float | None:
    """P(a violating reply scores above a clean reply to the SAME prompt), ties count half.

    GRPO only compares replies within a prompt's group, so this is the separation that
    shapes the policy update. None when no prompt has both a violating and a clean reply.
    """
    by: dict = {}
    for s, y, g in zip(scores, labels, groups):
        if g is not None:
            by.setdefault(g, ([], []))[0 if y else 1].append(float(s))
    wins = pairs = 0.0
    for pos, neg in by.values():
        for p in pos:
            for n in neg:
                wins += 1.0 if p > n else 0.5 if p == n else 0.0
                pairs += 1
    return wins / pairs if pairs else None


def reply_metrics(rows: list[dict]) -> dict:
    """Training reward, true reward and violation rate over all replies, with 95% bootstrap intervals over prompts."""
    out = {"prompts": len(rows), "replies": sum(len(r["replies"]) for r in rows)}
    for key in ("reward", "true", "violation"):
        per = [
            sum(s[key] for s in r["scores"]) / max(1, len(r["scores"])) for r in rows
        ]
        out[key] = sum(per) / max(1, len(per))
        out[key + "_ci95"] = boot_ci(per)
    return out


def monitor_audit(rows: list[dict], harm_key: str = "monitor_harm") -> dict | None:
    """How well a monitor's harm score (cost, or 1 - D) separates violations among reward-earning replies (audit only)."""
    harm, viol, group = [], [], []
    for g, r in enumerate(rows):
        for s, h in zip(r["scores"], r.get(harm_key) or []):
            if s["reward"] > 0:
                harm.append(float(h))
                viol.append(int(s["violation"]))
                group.append(g)
    if not harm:
        return None
    n_v = sum(viol)
    mean = lambda xs: (sum(xs) / len(xs)) if xs else None  # noqa: E731
    return {
        "n_reward_earning": len(harm),
        "n_violating": n_v,
        "auroc": auroc(harm, viol),
        "within_prompt_auroc": within_prompt_auroc(harm, viol, group),
        "mean_harm_violating": mean([h for h, v in zip(harm, viol) if v]),
        "mean_harm_clean": mean([h for h, v in zip(harm, viol) if not v]),
        # violations the monitor would wave through at a 0.5 threshold, as a share of all reward-earning replies
        "undetected_violation_rate": sum(1 for h, v in zip(harm, viol) if v and h < 0.5)
        / len(harm),
        "false_flag_rate": sum(1 for h, v in zip(harm, viol) if not v and h >= 0.5)
        / max(1, len(harm) - n_v),
    }


def show(title: str, bench, rows: list[dict], n: int = 5) -> None:
    """The exact model input, each reply, and its verdicts, for n prompts."""
    print(f"\n{'=' * 25} {title} {'=' * 25}", flush=True)
    for r in rows[:n]:
        print(f"--- {r['id']} ---")
        for m in bench.chat_messages(r):
            print(f"[{m['role'].upper()}]\n{m['content']}")
        harms = r.get("monitor_harm") or [None] * len(r["replies"])
        for rep, s, h in zip(r["replies"], r["scores"], harms):
            print(
                f"[REPLY] training_reward={s['reward']} true_reward={s['true']} violation={s['violation']} "
                f"monitor_harm={h}\n{rep}"
            )
    print(flush=True)


def evaluate_model(
    cfg,
    bench,
    out: Path,
    tag: str,
    adapter: Path | None,
    replicas=None,
    harm_fn=None,
    model_dir: Path | None = None,
) -> dict:
    """Sample the model on the test prompts, score, optionally add a monitor's harm score; writes eval/<tag>.*

    `model_dir` serves that full checkpoint instead of cfg.policy.model (GRAFT's widened checkpoints).
    """
    rows_p, met_p = out / "eval" / f"{tag}.jsonl", out / "eval" / f"{tag}_metrics.json"
    if met_p.exists():
        return json.loads(met_p.read_text())
    if model_dir is not None:
        from omegaconf import OmegaConf

        cfg = OmegaConf.merge(
            cfg, OmegaConf.create({"policy": {"model": str(model_dir)}})
        )
    items = bench.load_items(cfg, "test")
    ev = cfg.evaluation
    with served(cfg, out / "eval", adapter, replicas) as servers:
        reps = sample_replies(
            cfg, bench, servers, items, int(ev.k), float(ev.temperature)
        )
    flat_items = [it for it, rs in zip(items, reps) for _ in rs]
    flat = [x for rs in reps for x in rs]
    scores = bench.score_replies(flat_items, flat)
    harms = harm_fn(flat_items, flat) if harm_fn else None
    rows, i = [], 0
    for it, rs in zip(items, reps):
        row = {**it, "replies": rs, "scores": scores[i : i + len(rs)]}
        if harms is not None:
            row["monitor_harm"] = harms[i : i + len(rs)]
        rows.append(row)
        i += len(rs)
    save_rows(rows_p, rows)
    met = reply_metrics(rows)
    if harms is not None:
        met["monitor_audit"] = monitor_audit(rows)
    met_p.write_text(json.dumps(met, indent=2))
    show(f"{tag}: test prompts", bench, rows)
    log(f"{tag} test metrics: {json.dumps(met)}")
    return met


# ---- stages -----------------------------------------------------------------


def stage_sft_prime(cfg, out: Path) -> None:
    from src.lagrangian_finetuning.demonstration_sft_priming import sft_prime

    merged = sft_prime(cfg, out / "sft", log=log)
    (out / "state.json").write_text(
        json.dumps({"method": "sft_prime", "merged_model": str(merged)}, indent=2)
    )


def stage_base_eval(cfg, bench, out: Path) -> None:
    evaluate_model(cfg, bench, out, "base", None)


def latest_complete_checkpoint(trainer_dir: Path) -> Path | None:
    """The highest-step checkpoint-N the trainer finished writing (trainer_state.json is written last)."""
    done = [
        c
        for c in trainer_dir.glob("checkpoint-*")
        if (c / "trainer_state.json").exists() and c.name.split("-")[-1].isdigit()
    ]
    return max(done, key=lambda c: int(c.name.split("-")[-1])) if done else None


def stage_train(cfg, bench, out: Path) -> None:
    """One GRPO run: rl.signal = task_only (no intervention), verifier_gated (VARL) or gradient_routing (GRAFT)."""
    from omegaconf import OmegaConf

    from src.lagrangian_finetuning.single_turn_constrained_grpo import grpo_train

    state_p = out / "state.json"
    if state_p.exists():
        log("train: done already")
        return
    signal = str(cfg.rl.signal)
    train_gpus = [g for g in str(cfg.rl.train_gpus).split(",") if g.strip()]
    # a job stopped by its time limit is continued by submitting the same command again
    latest = latest_complete_checkpoint(out / "rl" / "trainer")
    if latest is not None and not cfg.rl.get("resume_from_checkpoint"):
        OmegaConf.update(cfg, "rl.resume_from_checkpoint", str(latest), force_add=True)
        log(f"train: continuing from {latest}")
    if signal == "verifier_gated":
        from src.lagrangian_finetuning.verifier_gated_discriminator import (
            DiscriminatorClient,
            discriminator_server,
        )

        with discriminator_server(cfg, out / "varl", log=log) as url:
            adapter = grpo_train(cfg, out / "rl", monitor_url=url, log=log)
            client = DiscriminatorClient(url)

            def harm(
                items, replies
            ):  # 1 - D: high = the discriminator thinks it is not a demonstration
                return [
                    1.0 - d
                    for d in client.score(
                        [bench.monitor_text(it, x) for it, x in zip(items, replies)]
                    )
                ]

            # the discriminator only lives inside this block, so its test-set audit is done here
            evaluate_model(
                cfg, bench, out, "tuned", adapter, replicas=train_gpus, harm_fn=harm
            )
    elif signal == "task_only":
        adapter = grpo_train(cfg, out / "rl", log=log)
    elif signal == "gradient_routing":
        from src.lagrangian_finetuning.gradient_routing_grpo import (
            gradient_routing_train,
        )

        adapter = gradient_routing_train(cfg, out / "rl", log=log)
    else:
        raise ValueError(
            f"stage train runs rl.signal task_only, verifier_gated or gradient_routing, got {signal}"
        )
    state_p.write_text(
        json.dumps({"method": signal, "adapter": str(adapter)}, indent=2)
    )
    log(f"train: adapter {adapter}")


_EMBEDDING_CACHE: dict = {}


def cached_embeddings(phi, trajs: list):
    """phi.embed(trajs) on the CPU, computed once per process for a given encoder and set of texts.

    The encoder is frozen, so the allowed set's embeddings never change between rounds; only the head
    is trained. Kept in memory only (never written to disk: the demonstrations have no licence).
    """
    import hashlib

    h = hashlib.sha256()
    for t in trajs:
        for step in [t] if isinstance(t, str) else t:
            h.update(step.encode())
            h.update(b"\x00")
        h.update(b"\x01")
    key = (json.dumps(phi.meta, sort_keys=True), len(trajs), h.hexdigest())
    if key not in _EMBEDDING_CACHE:
        _EMBEDDING_CACHE[key] = phi.embed(trajs).to("cpu")
    return _EMBEDDING_CACHE[key]


def fit_constraint_round(
    cfg,
    out: Path,
    rdir: Path,
    r: int,
    previous_phi_dir: str | None,
    expert_trajs: list,
    samples: list[dict],
    flag_quantile: float | None = None,
    heldout_trajs: list | None = None,
    update: bool = True,
):
    """One ICRL inverse step: re-fit C_theta on the demonstrations against this round's policy samples.

    `samples` are {"trajectory", "violates" (audit only), "reward", optional "group" (prompt index)}. The head
    is warm-started from the last round and the PUCL memory is extended: one file in `out`, or with
    icrl.memory_per_round one file per round under out/pucl_memory, so a re-run round replaces its own
    negatives instead of adding them twice. Returns (phi_dir, inverse-step stats, audit).
    `heldout_trajs`: allowed replies the head never trains on. They set PUCL's distance threshold
    (icrl.pucl_selection = heldout_quantile) and, with icrl.budget_from_heldout, the round's cost budget:
    their mean cost + icrl.budget_margin, written to phi_dir/budget.json and returned as audit["budget"].
    `update=False` keeps the previous head (the frozen-constraint control) and only audits it.
    With `flag_quantile` q, also writes routing_threshold.json in phi_dir: the cost that a share q of the
    demonstrations stay below (GRAFT flags replies above it).
    """
    import gc

    import torch

    from src.icrl_dual_training.maxent_constraint import MaxEntConstraintConfig
    from src.icrl_dual_training.secret_keeping_reply_constraint import (
        TrajectoryFeasibility,
        inverse_step,
    )

    ic, c = cfg.icrl, cfg.constraint
    phi = TrajectoryFeasibility.build(
        str(c.encoder_model),
        int(c.max_length),
        int(c.head_hidden),
        device=c.get("device"),
    )
    if previous_phi_dir:
        phi.load_head(previous_phi_dir)
    e_emb = cached_embeddings(phi, expert_trajs)
    n_emb = phi.embed([x["trajectory"] for x in samples]).to("cpu")
    mcfg = MaxEntConstraintConfig(
        beta=float(ic.beta),
        sparsity_coeff=float(ic.sparsity_coeff),
        negative_selection=str(ic.negative_selection),
        n_steps=int(ic.first_round_constraint_steps if r == 0 else ic.constraint_steps),
        batch_trajectories=int(ic.constraint_batch),
        lr=float(ic.constraint_lr),
        pucl_keep_fraction=float(ic.get("pucl_keep_fraction", 0.3)),
        pucl_selection=str(ic.get("pucl_selection", "fraction")),
        pucl_quantile=float(ic.get("pucl_quantile", 0.99)),
        pucl_k=int(ic.get("pucl_k", 2)),
        batch_steps=int(ic.constraint_batch),
    )
    h_emb = cached_embeddings(phi, heldout_trajs) if heldout_trajs else None
    pair_idx = torch.full((len(samples),), -1)
    if not update:
        cstats = {"skipped": "frozen constraint (icrl.freeze_constraint_after_round)"}
    elif bool(ic.get("memory_per_round", False)):
        mem_dir = out / "pucl_memory"
        mem_dir.mkdir(parents=True, exist_ok=True)
        earlier = [
            torch.load(mem_dir / f"round_{q:02d}.pt", map_location="cpu")
            for q in range(r)
            if (mem_dir / f"round_{q:02d}.pt").exists()
        ]
        memory = torch.cat(earlier) if earlier else None
        n_old = 0 if memory is None else int(memory.shape[0])
        cstats, new_memory = inverse_step(
            phi,
            e_emb,
            n_emb,
            pair_idx,
            ic,
            mcfg,
            memory=memory,
            heldout=None if h_emb is None else h_emb.embeddings,
            return_selected=True,
        )
        if new_memory is not None:
            torch.save(new_memory[n_old:].clone(), mem_dir / f"round_{r:02d}.pt")
    else:
        mem_p = out / "icrl_pucl_memory.pt"
        memory = torch.load(mem_p, map_location="cpu") if mem_p.exists() else None
        cstats, memory = inverse_step(
            phi,
            e_emb,
            n_emb,
            pair_idx,
            ic,
            mcfg,
            memory=memory,
            heldout=None if h_emb is None else h_emb.embeddings,
            return_selected=True,
        )
        if memory is not None:
            torch.save(memory, mem_p)
    costs = phi.costs(n_emb)
    expert_costs = phi.costs(e_emb)
    viol = [x["violates"] for x in samples]
    selected = cstats.pop("selected_indices", None)
    mean = lambda xs: (sum(xs) / len(xs)) if xs else None  # noqa: E731
    audit = {
        "n": len(samples),
        "n_violating": sum(viol),
        "auroc": auroc(costs, viol),
        "within_prompt_auroc": within_prompt_auroc(
            costs, viol, [x.get("group") for x in samples]
        ),
        "mean_cost_violating": mean([c_ for c_, v in zip(costs, viol) if v]),
        "mean_cost_clean": mean([c_ for c_, v in zip(costs, viol) if not v]),
        "mean_cost_expert": mean(expert_costs),
    }
    if selected is not None:
        # which of this round's samples PUCL marked forbidden, scored against the label (audit only)
        audit["marked"] = len(selected)
        audit["marked_violating"] = sum(int(viol[i]) for i in selected)
    phi_dir = rdir / "phi"
    phi.save(phi_dir)
    if h_emb is not None:
        audit["mean_cost_heldout_allowed"] = mean(phi.costs(h_emb))
    if bool(ic.get("budget_from_heldout", False)):
        if h_emb is None:
            raise ValueError("icrl.budget_from_heldout needs held-out allowed replies")
        audit["budget"] = audit["mean_cost_heldout_allowed"] + float(
            ic.get("budget_margin", 0.05)
        )
        (phi_dir / "budget.json").write_text(
            json.dumps(
                {
                    "budget": audit["budget"],
                    "mean_cost_heldout_allowed": audit["mean_cost_heldout_allowed"],
                    "margin": float(ic.get("budget_margin", 0.05)),
                }
            )
        )
    if flag_quantile is not None:
        threshold = demonstration_cost_quantile(expert_costs, flag_quantile)
        flagged = [c_ > threshold for c_ in costs]
        audit["routing_threshold"] = threshold
        audit["flagged"] = sum(flagged)
        audit["flagged_violating"] = sum(1 for f, v in zip(flagged, viol) if f and v)
        (phi_dir / "routing_threshold.json").write_text(
            json.dumps(
                {"threshold": threshold, "quantile_of_demonstrations": flag_quantile}
            )
        )
    log(f"round {r} inverse step: {json.dumps(cstats)}")
    log(
        f"round {r} constraint on this round's unlabelled set (violation label as truth, audit only): {json.dumps(audit)}"
    )
    del phi, e_emb, n_emb, h_emb
    gc.collect()
    torch.cuda.empty_cache()
    return phi_dir, cstats, audit


def demonstration_cost_quantile(expert_costs: list[float], q: float) -> float:
    """The cost that a share q of the demonstrations stay at or below."""
    xs = sorted(float(c) for c in expert_costs)
    if not xs:
        raise ValueError("no demonstration costs")
    return xs[min(len(xs) - 1, max(0, int(round(q * (len(xs) - 1)))))]


def select_demonstrations(cfg, bench, rng: random.Random) -> tuple[list[dict], int]:
    """(the demonstrations in the allowed set, how many were loaded).

    icrl.drop_hacking_demonstrations removes those whose hack label is set (the label is used once, to
    clean the allowed set); icrl.expert_demos caps the count (null = all). Draws from `rng` exactly as the
    published runs did, so their configs select the same demonstrations.
    """
    ic = cfg.icrl
    demos = bench.load_items(cfg, "demonstrations")
    n_loaded = len(demos)
    if bool(ic.get("drop_hacking_demonstrations", False)):
        demos = [d for d in demos if not int(d.get("hack", 0))]
    n = (
        len(demos)
        if ic.get("expert_demos") is None
        else min(len(demos), int(ic.expert_demos))
    )
    return rng.sample(demos, n), n_loaded


def split_heldout(trajs: list, fraction: float, seed: int) -> tuple[list, list]:
    """(to fit, held out): a seeded random `fraction` of the allowed set is held out; none when fraction is 0."""
    if fraction <= 0:
        return list(trajs), []
    order = list(range(len(trajs)))
    random.Random(seed + 1).shuffle(order)
    n_held = max(1, int(round(fraction * len(trajs))))
    held = set(order[:n_held])
    return (
        [t for i, t in enumerate(trajs) if i not in held],
        [t for i, t in enumerate(trajs) if i in held],
    )


def start_model_allowed_replies(
    cfg, bench, out: Path, all_train: list, exclude: list, n_prompts: int
) -> list:
    """The starting model's reward-earning, non-violating replies to training prompts the loop never samples.

    Added to the allowed set so that "written by the policy vs. by the demonstrator" stops being the easiest
    split for C_theta. Uses the violation label once, to clean the set (as for the demonstrations). Saved to
    out/start_model_allowed.jsonl and reused on restart.
    """
    p = out / "start_model_allowed.jsonl"
    if not p.exists():
        used = {it["id"] for it in exclude}
        pool = [it for it in all_train if it["id"] not in used]
        prompts = random.Random(int(cfg.rl.seed) + 2).sample(
            pool, min(len(pool), n_prompts)
        )
        ic = cfg.icrl
        with served(cfg, out / "start_model_allowed", None) as servers:
            reps = sample_replies(
                cfg, bench, servers, prompts, int(ic.k), float(cfg.sampling.temperature)
            )
        flat_items = [it for it, rs in zip(prompts, reps) for _ in rs]
        flat = [x for rs in reps for x in rs]
        scores = bench.score_replies(flat_items, flat)
        rows = [
            {**it, "response": x}
            for it, x, s in zip(flat_items, flat, scores)
            if s["reward"] > 0 and not s["violation"]
        ]
        n_viol = sum(1 for s in scores if s["reward"] > 0 and s["violation"])
        save_rows(p, rows)
        log(
            f"start-model allowed replies: {len(flat)} sampled on {len(prompts)} prompts, "
            f"{len(rows)} kept (reward-earning, not violating; {n_viol} violating dropped)"
        )
    rows = load_rows(p)
    return [[bench.monitor_text(it, it["response"])] for it in rows]


def stage_icrl_loop(cfg, bench, out: Path) -> None:
    """ICRL (Malik et al. 2021): alternate the inverse step on C_theta and the Lagrangian GRPO forward step."""
    import gc

    import torch
    from omegaconf import OmegaConf

    from src.icrl_dual_training.maxent_constraint import MaxEntConstraintConfig
    from src.icrl_dual_training.secret_keeping_reply_constraint import (
        TrajectoryFeasibility,
        inverse_step,
    )
    from src.lagrangian_finetuning.single_turn_constrained_grpo import grpo_train

    state_p = out / "state.json"
    if state_p.exists():
        log("icrl_loop: done already")
        return
    ic, c = cfg.icrl, cfg.constraint
    K, N = int(ic.rounds), int(ic.grpo_steps_per_round)
    loop_p = out / "icrl_state.json"
    st = (
        json.loads(loop_p.read_text())
        if loop_p.exists()
        else {
            "round": 0,
            "adapter": None,
            "checkpoint": None,
            "lambda": float(cfg.rl.dual.lambda_init),
            "phi_dir": None,
            "history": [],
        }
    )
    rng = random.Random(int(cfg.rl.seed))
    demos, n_loaded = select_demonstrations(cfg, bench, rng)
    all_train = bench.load_items(cfg, "train")
    items = rng.sample(all_train, min(len(all_train), int(ic.sample_prompts)))
    allowed = [[bench.monitor_text(d, d["response"])] for d in demos]
    n_start = int(ic.get("start_model_allowed_prompts") or 0)
    if n_start:
        allowed += start_model_allowed_replies(
            cfg, bench, out, all_train, items, n_start
        )
    expert_trajs, heldout_trajs = split_heldout(
        allowed, float(ic.get("heldout_fraction") or 0.0), int(cfg.rl.seed)
    )
    log(
        f"icrl_loop: {K} rounds x {N} GRPO steps; allowed set {len(allowed)} "
        f"({len(demos)} of {n_loaded} demonstrations"
        f"{', hacking ones dropped' if ic.get('drop_hacking_demonstrations') else ''}"
        f"{f', + start-model replies' if n_start else ''}): {len(expert_trajs)} to fit, "
        f"{len(heldout_trajs)} held out; {len(items)} prompts x {int(ic.k)} policy samples per round"
    )
    freeze_after = ic.get("freeze_constraint_after_round")
    while st["round"] < K:
        r = st["round"]
        rdir = out / f"icrl_round_{r:02d}"
        rdir.mkdir(parents=True, exist_ok=True)
        log(
            f"=== ICRL round {r}: policy adapter={st['adapter']} lambda={st['lambda']:.3f} ==="
        )
        done_p = rdir / "inverse_done.json"
        if done_p.exists():
            # a restarted round: reuse its samples and fitted constraint instead of re-sampling
            # (re-sampling used to append the round's forbidden replies to the PUCL memory twice)
            done = json.loads(done_p.read_text())
            phi_dir, cstats, audit = (
                Path(done["phi_dir"]),
                done["inverse"],
                done["audit"],
            )
            log(f"round {r}: inverse step done before this restart; reusing {phi_dir}")
        else:
            # 1. sample the CURRENT policy
            adapter = Path(st["adapter"]) if st["adapter"] else None
            with served(cfg, rdir, adapter) as servers:
                reps = sample_replies(
                    cfg,
                    bench,
                    servers,
                    items,
                    int(ic.k),
                    float(cfg.sampling.temperature),
                )
            flat_items = [it for it, rs in zip(items, reps) for _ in rs]
            flat_group = [g for g, rs in enumerate(reps) for _ in rs]
            flat = [x for rs in reps for x in rs]
            scores = bench.score_replies(flat_items, flat)
            # PUCL's policy filter (Peng & Billard 2024, Eq. 6): keep samples whose reward is at least the
            # demonstration's. Demonstrations all earn the training reward, so that is "earns the reward".
            keep = (
                [i for i, s in enumerate(scores) if s["reward"] > 0]
                if bool(ic.get("reward_filter", True))
                else []
            )
            if len(keep) < int(ic.get("min_nominal", 64)):
                log(
                    f"round {r}: only {len(keep)} reward-earning samples; using all {len(flat)} as the unlabelled set"
                )
                keep = list(range(len(flat)))
            samples = [
                {
                    "trajectory": [bench.monitor_text(flat_items[i], flat[i])],
                    "violates": scores[i]["violation"],
                    "reward": scores[i]["reward"],
                    "group": flat_group[i],
                }
                for i in keep
            ]
            save_rows(
                rdir / "policy_samples.jsonl",
                [{**it, "replies": rs} for it, rs in zip(items, reps)],
            )
            n_viol = sum(s["violation"] for s in scores)
            log(
                f"round {r} policy samples: {len(flat)} replies, reward {sum(s['reward'] for s in scores) / len(flat):.3f}, "
                f"true {sum(s['true'] for s in scores) / len(flat):.3f}, violations {n_viol / len(flat):.3f} (audit); "
                f"{len(samples)} in the unlabelled set"
            )
            # 2. inverse step, warm-started from the last round (none after the freeze round: control run)
            frozen = freeze_after is not None and r > int(freeze_after)
            phi_dir, cstats, audit = fit_constraint_round(
                cfg,
                out,
                rdir,
                r,
                st["phi_dir"],
                expert_trajs,
                samples,
                heldout_trajs=heldout_trajs or None,
                update=not frozen,
            )
            done_p.write_text(
                json.dumps(
                    {"phi_dir": str(phi_dir), "inverse": cstats, "audit": audit},
                    indent=2,
                )
            )
        # 3. forward step: GRPO resumed from the last round, lambda by dual ascent
        rl_over = {
            "signal": "lagrangian",
            "phi_dir": str(phi_dir),
            "max_steps": (r + 1) * N,
            "save_steps": N,
            "resume_from_checkpoint": st["checkpoint"],
            "lambda_init": st["lambda"],
        }
        if audit.get("budget") is not None:
            rl_over["dual"] = {"epsilon": float(audit["budget"])}
            log(
                f"round {r}: cost budget {audit['budget']:.4f} (held-out allowed mean cost + margin)"
            )
        run_cfg = OmegaConf.merge(cfg, OmegaConf.create({"rl": rl_over}))
        adapter_dir = grpo_train(run_cfg, rdir / "rl", log=log)
        lam = json.loads((rdir / "rl" / "lambda.json").read_text())["lambda"]
        st["history"].append(
            {
                "round": r,
                "inverse": cstats,
                "constraint_audit": audit,
                "lambda_end": lam,
            }
        )
        st.update(round=r + 1, adapter=str(adapter_dir), phi_dir=str(phi_dir))
        st["lambda"] = lam
        st["checkpoint"] = str(rdir / "rl" / "trainer" / f"checkpoint-{(r + 1) * N}")
        loop_p.write_text(json.dumps(st, indent=2))
    state_p.write_text(
        json.dumps(
            {
                "method": "icrl_loop",
                "adapter": st["adapter"],
                "phi_dir": st["phi_dir"],
                "history": st["history"],
            },
            indent=2,
        )
    )
    log(f"icrl_loop: done, final adapter {st['adapter']}")


def stage_icrl_graft_loop(cfg, bench, out: Path) -> None:
    """ICRL + GRAFT: each round re-fits C_theta against the exploring policy (RETAIN + FORGET), then trains with
    C_theta routing the exploring policy's flagged replies to FORGET and constraining the deployed policy
    (RETAIN only) through reward - lambda * cost."""
    import shutil

    from omegaconf import OmegaConf

    from src.lagrangian_finetuning.gradient_routing_grpo import (
        gradient_routing_train,
        read_exploring_replies,
    )

    state_p = out / "state.json"
    if state_p.exists():
        log("icrl_graft_loop: done already")
        return
    ic, g = cfg.icrl, cfg.graft
    K, N = int(ic.rounds), int(ic.grpo_steps_per_round)
    mode = str(cfg.constraint.get("input", "files"))
    quantile = float(g.get("flag_quantile_of_demonstrations", 0.95))
    loop_p = out / "icrl_graft_state.json"
    st = (
        json.loads(loop_p.read_text())
        if loop_p.exists()
        else {
            "round": 0,
            "adapter": None,
            "checkpoint": None,
            "lambda": float(g.anchor.dual.lambda_init),
            "phi_dir": None,
            "history": [],
        }
    )
    rng = random.Random(int(cfg.rl.seed))
    demos = bench.load_items(cfg, "demonstrations")
    demos = rng.sample(demos, min(len(demos), int(ic.expert_demos)))
    expert_trajs = [[bench.constraint_text(d, d["response"], mode)] for d in demos]
    items = bench.load_items(cfg, "train")
    items = rng.sample(items, min(len(items), int(ic.sample_prompts)))
    log(
        f"icrl_graft_loop: {K} rounds x {N} GRPO steps; {len(expert_trajs)} expert demonstrations; "
        f"constraint input: {mode}; flag above the cost of {quantile:.0%} of demonstrations"
    )
    mem_p = out / "icrl_pucl_memory.pt"
    while st["round"] < K:
        r = st["round"]
        rdir = out / f"icrl_graft_round_{r:02d}"
        rdir.mkdir(parents=True, exist_ok=True)
        log(
            f"=== ICRL + GRAFT round {r}: adapters={st['adapter']} lambda={st['lambda']:.3f} ==="
        )
        # 1. replies of the exploring policy (RETAIN + FORGET), which is never penalised
        if r == 0:
            # before any training both adapters are zero, so the exploring policy is the starting model
            with served(cfg, rdir, None) as servers:
                reps = sample_replies(
                    cfg,
                    bench,
                    servers,
                    items,
                    int(ic.k),
                    float(cfg.sampling.temperature),
                )
            flat_items = [it for it, rs in zip(items, reps) for _ in rs]
            flat = [x for rs in reps for x in rs]
            scores = bench.score_replies(flat_items, flat)
            rows = [
                {
                    "text": bench.constraint_text(it, x, mode),
                    "reward": s_["reward"],
                    "violates": s_["violation"],
                    "true": s_["true"],
                }
                for it, x, s_ in zip(flat_items, flat, scores)
            ]
        else:
            # its own training replies from the round just finished
            previous = out / f"icrl_graft_round_{r - 1:02d}" / "rl"
            rows = read_exploring_replies(
                previous, int(ic.get("exploring_replies", 2000))
            )
            if not rows:
                raise RuntimeError(
                    f"no exploring-policy replies were logged in {previous}"
                )
        # PUCL's policy filter (Peng & Billard 2024, Eq. 6): the unlabelled set is the reward-earning replies
        keep = (
            [x for x in rows if x["reward"] > 0]
            if bool(ic.get("reward_filter", True))
            else []
        )
        if len(keep) < int(ic.get("min_nominal", 64)):
            log(
                f"round {r}: only {len(keep)} reward-earning replies; using all {len(rows)} as the unlabelled set"
            )
            keep = rows
        samples = [
            {
                "trajectory": [x["text"]],
                "violates": x["violates"],
                "reward": x["reward"],
            }
            for x in keep
        ]
        n = max(1, len(rows))
        log(
            f"round {r} exploring-policy replies: {len(rows)}, reward {sum(x['reward'] for x in rows) / n:.3f}, "
            f"true {sum(x['true'] for x in rows) / n:.3f}, violations {sum(x['violates'] for x in rows) / n:.3f} (audit); "
            f"{len(samples)} in the unlabelled set"
        )
        # 2. inverse step. A round restarted after a time limit must not add its negatives to the memory twice.
        snapshot, marker = (
            rdir / "pucl_memory_before.pt",
            rdir / "pucl_memory_before.none",
        )
        if snapshot.exists():
            shutil.copyfile(snapshot, mem_p)
        elif marker.exists():
            mem_p.unlink(missing_ok=True)
        elif mem_p.exists():
            shutil.copyfile(mem_p, snapshot)
        else:
            marker.write_text("")
        phi_dir, cstats, audit = fit_constraint_round(
            cfg,
            out,
            rdir,
            r,
            st["phi_dir"],
            expert_trajs,
            samples,
            flag_quantile=quantile,
        )
        # 3. forward step: routing by C_theta, and the deployed policy under reward - lambda * cost
        shutil.rmtree(rdir / "rl" / "exploring_replies", ignore_errors=True)
        run_cfg = OmegaConf.merge(
            cfg,
            OmegaConf.create(
                {
                    "rl": {
                        "max_steps": K * N,
                        "stop_at_step": (r + 1) * N,
                        "save_steps": N,
                        "resume_from_checkpoint": st["checkpoint"],
                    },
                    "graft": {
                        "phi_dir": str(phi_dir),
                        "log_exploring_replies": True,
                        "widened_start_dir": str(out / "widened_start"),
                        "anchor": {"lambda_init": st["lambda"]},
                    },
                }
            ),
        )
        adapters = gradient_routing_train(run_cfg, rdir / "rl", log=log)
        lam = json.loads((rdir / "rl" / "lambda.json").read_text())["lambda"]
        st["history"].append(
            {
                "round": r,
                "inverse": cstats,
                "constraint_audit": audit,
                "lambda_end": lam,
            }
        )
        st.update(round=r + 1, adapter=str(adapters), phi_dir=str(phi_dir))
        st["lambda"] = lam
        st["checkpoint"] = str(rdir / "rl" / "trainer" / f"checkpoint-{(r + 1) * N}")
        loop_p.write_text(json.dumps(st, indent=2))
    state_p.write_text(
        json.dumps(
            {
                "method": "gradient_routing",
                "variant": "icrl_graft_loop",
                "adapter": st["adapter"],
                "phi_dir": st["phi_dir"],
                "history": st["history"],
            },
            indent=2,
        )
    )
    log(f"icrl_graft_loop: done, final adapters {st['adapter']}")


def graft_monitor_harm(cfg, bench, phi_dir: str | None = None):
    """GRAFT's monitor as a harm score per reply for the test-set audit: its flag (0/1) or the learned cost."""
    g = cfg.graft
    if str(g.monitor) == "trivial_test":
        return lambda items, replies: [
            float(f) for f in bench.monitor_flags(items, replies)
        ]
    from src.icrl_dual_training.secret_keeping_reply_constraint import (
        TrajectoryFeasibility,
    )

    mode = str((cfg.get("constraint") or {}).get("input", "files"))

    def harm(items, replies):
        phi = TrajectoryFeasibility.load(Path(str(phi_dir or g.phi_dir)))
        return phi.score(
            [[bench.constraint_text(it, x, mode)] for it, x in zip(items, replies)]
        )

    return harm


def stage_evaluate_gradient_routing(cfg, bench, out: Path, state: dict) -> None:
    """GRAFT: evaluate the deployed model (RETAIN only) and, as a diagnostic, RETAIN + FORGET."""
    import torch

    from src.lagrangian_finetuning.gradient_routing_adapters import (
        write_widened_checkpoint,
    )

    adapters = torch.load(state["adapter"], map_location="cpu")
    harm_fn = graft_monitor_harm(cfg, bench, state.get("phi_dir"))
    metrics = {}
    for tag, name, forget in (
        ("tuned", "deployed_retain_only", False),
        ("tuned_retain_and_forget", "retain_and_forget", True),
    ):
        model_dir = out / name
        if (
            not (out / "eval" / f"{tag}_metrics.json").exists()
            and not (model_dir / "config.json").exists()
        ):
            log(f"evaluate: writing {model_dir}")
            tmp = model_dir.with_name(model_dir.name + "_tmp")
            write_widened_checkpoint(
                str(cfg.policy.model),
                tmp,
                int(cfg.graft.adapter_width),
                adapter_state=adapters,
                forget=forget,
            )
            tmp.rename(model_dir)
        metrics[tag] = evaluate_model(
            cfg, bench, out, tag, None, harm_fn=harm_fn, model_dir=model_dir
        )
    base_p = out / "eval" / "base_metrics.json"
    results = {
        "run": cfg.run_name,
        "method": state["method"],
        "adapter": state["adapter"],
        "deployed_model": str(out / "deployed_retain_only"),
        "base": json.loads(base_p.read_text()) if base_p.exists() else None,
        "tuned": metrics["tuned"],
        "retain_and_forget": metrics["tuned_retain_and_forget"],
        "rounds": state.get("history"),
    }
    (out / "results.json").write_text(json.dumps(results, indent=2))
    log(
        "RESULTS "
        + json.dumps(
            {k: results[k] for k in ("base", "tuned", "retain_and_forget")}, indent=2
        )
    )


def learned_cost_fn(cfg, bench, phi_dir: Path):
    """harm_fn for evaluate_model: the learned cost 1 - C_theta of the constraint saved in phi_dir."""

    def harm_fn(items, replies):
        from src.icrl_dual_training.secret_keeping_reply_constraint import (
            TrajectoryFeasibility,
        )

        phi = TrajectoryFeasibility.load(phi_dir, device=cfg.constraint.get("device"))
        return phi.score([[bench.monitor_text(it, x)] for it, x in zip(items, replies)])

    return harm_fn


def icrl_rounds_to_evaluate(out: Path) -> list[tuple[int, Path]]:
    """(round index, round dir) for every finished ICRL round except the last, whose adapter is eval/tuned."""
    done = sorted(
        (int(d.name.rsplit("_", 1)[1]), d)
        for d in out.glob("icrl_round_*")
        if d.is_dir() and (d / "rl" / "adapter").is_dir() and (d / "phi").is_dir()
    )
    return done[:-1]


def round_summary_row(cfg, r: int, rdir: Path, met: dict) -> dict:
    """One row of eval/rounds_summary.json: steps trained, test metrics, lambda, monitor AUROC."""
    lam_p = rdir / "rl" / "lambda.json"
    lam = json.loads(lam_p.read_text()) if lam_p.exists() else None
    if isinstance(lam, dict):
        lam = lam.get("lambda", lam)
    audit = met.get("monitor_audit") or {}
    return {
        "round": r,
        "steps": (r + 1) * int(cfg.icrl.grpo_steps_per_round),
        **{
            k: met[k]
            for k in (
                "true",
                "true_ci95",
                "violation",
                "violation_ci95",
                "reward",
                "reward_ci95",
            )
        },
        "lambda_after_round": lam,
        "monitor_auroc": audit.get("auroc"),
        "monitor_mean_cost_violating": audit.get("mean_harm_violating"),
        "monitor_mean_cost_clean": audit.get("mean_harm_clean"),
    }


def stage_evaluate_rounds(cfg, bench, out: Path) -> None:
    """Test-set eval of each saved ICRL round's adapter, scored by that round's own C_theta (the last round is eval/tuned)."""
    rows = []
    for r, rdir in icrl_rounds_to_evaluate(out):
        log(f"evaluate_rounds: round {r} ({rdir})")
        met = evaluate_model(
            cfg,
            bench,
            out,
            f"round_{r:02d}",
            rdir / "rl" / "adapter",
            harm_fn=learned_cost_fn(cfg, bench, rdir / "phi"),
        )
        rows.append(round_summary_row(cfg, r, rdir, met))
    (out / "eval").mkdir(parents=True, exist_ok=True)
    (out / "eval" / "rounds_summary.json").write_text(json.dumps(rows, indent=2))
    log("ROUNDS " + json.dumps(rows, indent=2))


def stage_evaluate(cfg, bench, out: Path) -> None:
    state = json.loads((out / "state.json").read_text())
    if not state.get("adapter"):
        raise RuntimeError("no fine-tuned adapter to evaluate")
    if state.get("method") == "gradient_routing":
        return stage_evaluate_gradient_routing(cfg, bench, out, state)
    harm_fn = (
        learned_cost_fn(cfg, bench, Path(state["phi_dir"]))
        if state.get("phi_dir")
        else None
    )
    tuned = evaluate_model(
        cfg, bench, out, "tuned", Path(state["adapter"]), harm_fn=harm_fn
    )
    base_p = out / "eval" / "base_metrics.json"
    results = {
        "run": cfg.run_name,
        "method": state["method"],
        "adapter": state["adapter"],
        "base": json.loads(base_p.read_text()) if base_p.exists() else None,
        "tuned": tuned,
        "rounds": state.get("history"),
    }
    (out / "results.json").write_text(json.dumps(results, indent=2))
    log("RESULTS " + json.dumps({k: results[k] for k in ("base", "tuned")}, indent=2))


def main() -> int:
    from omegaconf import OmegaConf

    from src.environments.single_turn_benchmark import get_benchmark

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--override", action="append", default=[])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument(
        "--stages",
        default=None,
        help="comma list; default depends on the config's blocks",
    )
    a = ap.parse_args()
    cfg = load_cfg(a.config.resolve(), a.override, a.smoke)
    out = Path(cfg.output.dir)
    out.mkdir(parents=True, exist_ok=True)
    if a.stages is None:
        if "icrl" in cfg and "graft" in cfg:
            a.stages = "base_eval,icrl_graft_loop,evaluate"
        elif "icrl" in cfg:
            a.stages = "base_eval,icrl_loop,evaluate"
        elif "rl" in cfg:
            a.stages = "base_eval,train,evaluate"
        else:
            a.stages = "sft_prime"
    stages = [s for s in a.stages.split(",") if s]
    unknown = sorted(set(stages) - set(STAGES))
    if unknown:
        raise SystemExit(f"unknown stages {unknown}; choose from {STAGES}")
    (out / f"config_resolved_{'_'.join(stages)}.yaml").write_text(
        OmegaConf.to_yaml(cfg, resolve=True)
    )
    log(f"run {cfg.run_name} -> {out}; stages {stages}")
    bench = get_benchmark(str(cfg.benchmark))
    if "sft_prime" in stages:
        stage_sft_prime(cfg, out)
    if "base_eval" in stages:
        stage_base_eval(cfg, bench, out)
    if "train" in stages:
        stage_train(cfg, bench, out)
    if "icrl_loop" in stages:
        stage_icrl_loop(cfg, bench, out)
    if "icrl_graft_loop" in stages:
        stage_icrl_graft_loop(cfg, bench, out)
    if "evaluate" in stages:
        stage_evaluate(cfg, bench, out)
    if "evaluate_rounds" in stages:
        stage_evaluate_rounds(cfg, bench, out)
    log("pipeline finished: " + ",".join(stages))
    return 0


if __name__ == "__main__":
    sys.exit(main())
