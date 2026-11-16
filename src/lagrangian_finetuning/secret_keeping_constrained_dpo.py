# ABOUTME: Lagrangian DPO for the secret-keeping benchmark: sample replies, score them with the helpfulness judge (R) and the learned constraint (C), build DPO pairs by R - lambda*C, and measure exact leak rates
# ABOUTME: Use: replies = sample_replies(items, policy, "nominal", k); R = judge_rewards(items, replies, judge); pairs, stats = build_pairs(rows, lam, cfg); reply_metrics(rows)
"""
Model access goes through OpenAI-compatible vLLM servers. RoundRobinPolicy
spreads requests over the replicas (every prompt here shares the same opening
lines, so routing by prompt text would pile everything on one server).

DPO pairs are made per training prompt from k sampled replies: each reply's
score is S = (R - mean R at that prompt) / reward_scale - lambda * C, and the
pair is (highest S, lowest S) when the gap clears pair_margin. Training reuses
machiavelli_constrained_dpo.dpo_train, which takes chat-format pairs.
"""

from __future__ import annotations

import itertools
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from src.environments.machiavelli.rollout_collector import (
    OpenAIChatPolicy,
    sample_policy,
)
from src.environments.secret_keeping.secret_keeping_adapter import (
    JUDGE_SYSTEM,
    judge_prompt,
    leaks,
    parse_score,
    system_prompt,
)


class RoundRobinPolicy:
    """Cycles requests over several policies (one per vLLM replica)."""

    def __init__(self, policies: list):
        self.policies, self.lock, self.it = (
            policies,
            threading.Lock(),
            itertools.cycle(policies),
        )

    def _next(self):
        with self.lock:
            return next(self.it)

    def __call__(self, system: str, user: str, n_choices: int = 0) -> str:
        return self._next()(system, user, n_choices)

    def sample(self, system: str, user: str, n_choices: int, k: int) -> list[str]:
        return sample_policy(self._next(), system, user, n_choices, k)


def make_clients(
    servers, temperature: float, max_tokens: int, judge_max_tokens: int = 4
):
    """(policy, judge): the policy is the served model (adapter if loaded), the judge always the base weights at temperature 0."""
    policy = RoundRobinPolicy(
        [
            OpenAIChatPolicy(
                s.base_url,
                s.served_name,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            for s in servers
        ]
    )
    judge = RoundRobinPolicy(
        [
            OpenAIChatPolicy(
                s.base_url, "base", temperature=0.0, max_tokens=judge_max_tokens
            )
            for s in servers
        ]
    )
    return policy, judge


def _pmap(fn, xs, workers: int):
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(fn, xs))


def sample_replies(
    items: list[dict], policy, kind: str, k: int, workers: int = 64
) -> list[list[str]]:
    """k replies per prompt under the `kind` system prompt (one batched request each)."""
    return _pmap(
        lambda it: sample_policy(policy, system_prompt(kind, it), it["user"], 0, k),
        items,
        workers,
    )


def judge_rewards(
    items: list[dict], replies: list[list[str]], judge, workers: int = 64
) -> list[list[float | None]]:
    """Judge score 0-10 per reply; None when the judge's answer does not parse."""
    flat = [(it["user"], r) for it, rs in zip(items, replies) for r in rs]
    scores = _pmap(
        lambda ur: parse_score(judge(JUDGE_SYSTEM, judge_prompt(*ur), 0)), flat, workers
    )
    out, i = [], 0
    for rs in replies:
        out.append(scores[i : i + len(rs)])
        i += len(rs)
    return out


def build_pairs(rows: list[dict], lam: float, cfg) -> tuple[list[dict], dict]:
    """DPO pairs from rows with replies, R (judge) and C (constraint) per reply."""
    f = cfg.finetune
    margin, scale = float(f.pair_margin), float(f.reward_scale)
    pairs, gaps, c_all = [], [], []
    n_prompts = n_chosen_safer = 0
    for row in rows:
        idx = [i for i, R in enumerate(row["R"]) if R is not None]
        c_all += [row["C"][i] for i in idx]
        if len(idx) < 2:
            continue
        n_prompts += 1
        R = np.array([row["R"][i] for i in idx], dtype=float)
        C = np.array([row["C"][i] for i in idx], dtype=float)
        S = (R - R.mean()) / scale - lam * C
        b, w = int(np.argmax(S)), int(np.argmin(S))
        gap = float(S[b] - S[w])
        if b == w or gap < margin or row["replies"][idx[b]] == row["replies"][idx[w]]:
            continue
        gaps.append(gap)
        n_chosen_safer += int(C[b] < C[w])
        best, worst = row["replies"][idx[b]], row["replies"][idx[w]]
        pairs.append(
            {
                "prompt": [
                    {"role": "system", "content": system_prompt("nominal", row)},
                    {"role": "user", "content": row["user"]},
                ],
                "chosen": [{"role": "assistant", "content": best}],
                "rejected": [{"role": "assistant", "content": worst}],
                "id": row["id"],
                "kind": row["kind"],
                "R_chosen": float(R[b]),
                "R_rejected": float(R[w]),
                "C_chosen": float(C[b]),
                "C_rejected": float(C[w]),
                # ground truth, for audit only; never used to build the pair
                "gt_chosen_leaks": leaks(best, row["secret"]),
                "gt_rejected_leaks": leaks(worst, row["secret"]),
            }
        )
    stats = {
        "lambda": lam,
        "prompts": n_prompts,
        "pairs": len(pairs),
        "mean_C_sampled": float(np.mean(c_all)) if c_all else None,
        "pairs_where_chosen_has_lower_C": n_chosen_safer,
        "mean_pair_gap": float(np.mean(gaps)) if gaps else None,
        "pairs_gt_chosen_clean_rejected_leaks": sum(
            (not p["gt_chosen_leaks"]) and p["gt_rejected_leaks"] for p in pairs
        ),
        "pairs_gt_chosen_leaks": sum(p["gt_chosen_leaks"] for p in pairs),
    }
    return pairs, stats


def _boot(per_prompt: list[float], n: int = 1000) -> list[float]:
    if len(per_prompt) < 2:
        return [float(np.mean(per_prompt)) if per_prompt else 0.0] * 2
    rng = np.random.default_rng(0)
    a = np.array(per_prompt)
    b = [rng.choice(a, size=len(a)).mean() for _ in range(n)]
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def reply_metrics(rows: list[dict]) -> dict:
    """Exact leak rates and judge helpfulness per prompt kind, with 95% bootstrap intervals over prompts."""
    out = {}
    for kind in ("attack", "benign"):
        rs = [r for r in rows if r["kind"] == kind]
        leak = [
            float(np.mean([leaks(x, r["secret"]) for x in r["replies"]])) for r in rs
        ]
        helpful = [
            float(np.mean([R for R in r["R"] if R is not None]))
            for r in rs
            if any(R is not None for R in r["R"])
        ]
        out[kind] = {
            "prompts": len(rs),
            "replies": sum(len(r["replies"]) for r in rs),
            "leak_rate": float(np.mean(leak)) if leak else 0.0,
            "leak_rate_ci95": _boot(leak),
            "helpfulness": float(np.mean(helpful)) if helpful else None,
            "helpfulness_ci95": _boot(helpful),
            "judge_unparsed": sum(R is None for r in rs for R in r["R"]),
        }
    return out
