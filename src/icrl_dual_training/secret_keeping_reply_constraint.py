# ABOUTME: ICRL constraint for secret-keeping replies: oracle-expert vs policy demos, the one-shot classifier C_theta, and the max-entropy (Malik et al. 2021) feasibility phi used by the alternating ICRL loop
# ABOUTME: Use: expert, unsafe = oracle_demo_pairs(rows, pair_filter="higher_return"); train with machiavelli_step_constraint.train_step_constraint; reply_detector_eval(model, rows)
"""
One demo row per training prompt holds 4 expert-prompt replies and 4 policy
(nominal-prompt) replies, each with its judge reward. The oracle expert knows
the rule: of its replies it keeps the highest-reward one that does not leak.
Unsafe demos are the policy's replies on the same prompt that beat the expert
reply's reward (`higher_return`) or simply differ from it (`any_different`).
Neither filter reads a leak label; leak labels on policy replies are recorded
for reporting only.

The constraint model reads the nominal system prompt, the user message and the
reply, so the expert and unsafe rows of a pair differ only in the reply.

Two constraint models live here. ReplyFeasibility + maxent_round is the ICRL
one (Malik et al. 2021 max-entropy update against the CURRENT policy's samples,
used by the icrl_loop stage). oracle_demo_pairs feeds the BASELINE one-shot
supervised classifier (machiavelli_step_constraint.train_step_constraint),
which is not ICRL.
"""

from __future__ import annotations

from src.environments.secret_keeping.secret_keeping_adapter import leaks, system_prompt
from src.icrl_dual_training.machiavelli_step_constraint import (
    auroc,
    within_state_accuracy,
)

PAIR_FILTERS = ("higher_return", "any_different")


def constraint_text(item: dict, reply: str) -> str:
    return f"SYSTEM: {system_prompt('nominal', item)}\nUSER: {item['user']}\nREPLY: {reply}"


def oracle_expert_pick(
    replies: list[str], rewards: list[float | None], secret: str
) -> int | None:
    """Index of the highest-reward expert reply that does not leak (first on ties); None if all leak or are unscored."""
    ok = [
        i
        for i, (r, R) in enumerate(zip(replies, rewards))
        if R is not None and not leaks(r, secret)
    ]
    return max(ok, key=lambda i: (rewards[i], -i)) if ok else None


def oracle_pair_indices(row: dict, pair_filter: str) -> list[int]:
    """Policy replies at this prompt that become unsafe demos. Reads no leak label."""
    if pair_filter not in PAIR_FILTERS:
        raise ValueError(f"pair_filter must be one of {PAIR_FILTERS}")
    e = row.get("expert_pick")
    if e is None:
        return []
    e_text, e_R = row["expert_replies"][e], row["expert_R"][e]
    out, seen = [], set()
    for i, (txt, R) in enumerate(zip(row["policy_replies"], row["policy_R"])):
        if txt in seen or txt.strip() == e_text.strip():
            continue
        seen.add(txt)
        if pair_filter == "higher_return" and not (R is not None and R > e_R):
            continue
        out.append(i)
    return out


def oracle_demo_pairs(
    rows: list[dict], *, pair_filter: str
) -> tuple[list[dict], list[dict]]:
    """Expert and unsafe demos aligned row by row; row i of both is the same prompt."""
    expert, unsafe = [], []
    for row in rows:
        e = row.get("expert_pick")
        for i in oracle_pair_indices(row, pair_filter):
            expert.append(
                {
                    "state": row["id"],
                    "text": constraint_text(row, row["expert_replies"][e]),
                }
            )
            unsafe.append(
                {
                    "state": row["id"],
                    "text": constraint_text(row, row["policy_replies"][i]),
                }
            )
    return expert, unsafe


def reply_rows(rows: list[dict], replies_key: str = "replies") -> list[dict]:
    """Every reply at every prompt with its ground-truth leak, in within_state_accuracy's format."""
    out = []
    for row in rows:
        labels = [leaks(r, row["secret"]) for r in row[replies_key]]
        kind = "mixed" if any(labels) and not all(labels) else "uniform"
        for r, lab in zip(row[replies_key], labels):
            out.append(
                {
                    "state": row["id"],
                    "state_kind": kind,
                    "prompt_kind": row["kind"],
                    "violates": lab,
                    "text": constraint_text(row, r),
                }
            )
    return out


def reply_detector_eval(model, rows: list[dict], replies_key: str = "replies") -> dict:
    """C_theta against the exact leak check: AUROC over all replies, and leaking-above-clean within a prompt."""
    rr = reply_rows(rows, replies_key)
    scores = model.score([r["text"] for r in rr])
    labels = [r["violates"] for r in rr]
    att = [i for i, r in enumerate(rr) if r["prompt_kind"] == "attack"]
    return {
        "n_replies": len(rr),
        "n_leaking": int(sum(labels)),
        "auroc_all_replies": auroc(scores, labels),
        "auroc_attack_prompts": auroc(
            [scores[i] for i in att], [labels[i] for i in att]
        ),
        "within_prompt_leaking_above_clean": within_state_accuracy(rr, scores),
        "prompts_with_both": len(
            {r["state"] for r in rr if r["state_kind"] == "mixed"}
        ),
        "mean_score_leaking": sum(s for s, l in zip(scores, labels) if l)
        / max(1, sum(labels)),
        "mean_score_clean": sum(s for s, l in zip(scores, labels) if not l)
        / max(1, len(labels) - sum(labels)),
    }


# ---- Max-entropy ICRL (Malik et al. 2021) feasibility over replies ----------
#
# The genuine ICRL constraint: a feasibility function phi(prompt, reply) in (0, 1),
# cost = 1 - phi, fitted by maxent_constraint.maxent_constraint_update (expert
# log-likelihood: raise phi on expert replies, lower it on the CURRENT policy's
# replies, sparsity regulariser toward feasible). A reply is a one-step
# trajectory, so each TrajectoryBatch row is its own trajectory.


def one_step_batch(embeddings):
    """A TrajectoryBatch where every row is a one-step trajectory."""
    import torch

    from src.icrl_dual_training.maxent_constraint import TrajectoryBatch

    n = embeddings.size(0)
    return TrajectoryBatch(
        embeddings=embeddings,
        owner=torch.arange(n, device=embeddings.device),
        n_trajectories=n,
    )


class ReplyFeasibility:
    """Frozen encoder (embeds constraint_text) + StepFeasibilityHead phi; cost = 1 - phi."""

    def __init__(self, encoder, head, meta: dict):
        self.encoder, self.head, self.meta = encoder, head, meta

    @classmethod
    def build(
        cls, encoder_model: str, max_length: int, head_hidden: int, device=None
    ):
        from src.icrl_dual_training.machiavelli_step_constraint import StepConstraint
        from src.icrl_dual_training.maxent_constraint import StepFeasibilityHead

        enc = StepConstraint.build(encoder_model, max_length, head_hidden, device=device)
        dev = next(enc.encoder.backbone.parameters()).device
        head = StepFeasibilityHead(enc.encoder.hidden_size, head_hidden).to(dev)
        meta = {
            "encoder_model": encoder_model,
            "max_length": max_length,
            "head_hidden": head_hidden,
        }
        return cls(enc, head, meta)

    @classmethod
    def load(cls, out_dir, device=None):
        import json
        from pathlib import Path

        import torch

        meta = json.loads((Path(out_dir) / "phi_meta.json").read_text())
        obj = cls.build(
            meta["encoder_model"], meta["max_length"], meta["head_hidden"], device=device
        )
        obj.head.load_state_dict(
            torch.load(Path(out_dir) / "phi_head.pt", map_location="cpu")
        )
        return obj

    def load_head(self, out_dir) -> None:
        from pathlib import Path

        import torch

        self.head.load_state_dict(
            torch.load(Path(out_dir) / "phi_head.pt", map_location="cpu")
        )

    def save(self, out_dir) -> None:
        import json
        from pathlib import Path

        import torch

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.head.state_dict(), out_dir / "phi_head.pt")
        (out_dir / "phi_meta.json").write_text(json.dumps(self.meta, indent=2))

    def embed(self, texts: list[str], batch_size: int = 16):
        return self.encoder.embed(texts, batch_size)

    def costs_from_embeddings(self, emb) -> list[float]:
        import torch

        with torch.no_grad():
            dev = next(self.head.parameters()).device
            return (1.0 - self.head(emb.to(dev))).float().cpu().tolist()

    def score(self, texts: list[str], batch_size: int = 16) -> list[float]:
        """Cost 1 - phi per text (high = infeasible); the interface reply_detector_eval expects."""
        if not texts:
            return []
        return self.costs_from_embeddings(self.embed(texts, batch_size))


def maxent_round(phi: ReplyFeasibility, expert_emb, nominal_emb, mcfg) -> dict:
    """One inverse step of Malik et al. on cached embeddings; updates phi.head in place."""
    from src.icrl_dual_training.maxent_constraint import maxent_constraint_update

    return maxent_constraint_update(
        phi.head, one_step_batch(expert_emb), one_step_batch(nominal_emb), mcfg
    )


INVERSE_METHODS = ("maxent", "maxent_paired", "pucl")


def paired_maxent_update(head, expert_emb, nominal_emb, pair_expert_idx, cfg, rank_weight: float) -> dict:
    """Malik et al. max-entropy update plus a same-context ranking term.

    The plain update can lower feasibility on a whole CONTEXT (every reply to an
    attack chat) instead of on the replies the expert avoids, which leaves no
    contrast inside a GRPO group. The ranking term -log sigma(logit phi(expert
    reply) - logit phi(policy reply)), over pairs from the same chat, forces the
    feasibility to differ between replies to the same context.
    pair_expert_idx[i] is the expert row for nominal row i's chat, or -1.
    """
    import torch
    import torch.nn.functional as F

    device = next(head.parameters()).device
    e, n = expert_emb.to(device), nominal_emb.to(device)
    paired = (pair_expert_idx >= 0).nonzero(as_tuple=True)[0]
    opt = torch.optim.AdamW(head.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    b = cfg.batch_trajectories
    for _ in range(cfg.n_steps):
        ei = torch.randint(0, e.shape[0], (min(b, e.shape[0]),), device=device)
        ni = torch.randint(0, n.shape[0], (min(b, n.shape[0]),), device=device)
        log_e = torch.log(head(e[ei]).clamp(1e-6, 1 - 1e-6))
        log_n = torch.log(head(n[ni]).clamp(1e-6, 1 - 1e-6))
        likelihood = cfg.beta * (log_e.mean() - log_n.mean())
        sparsity = (1 - log_e.exp()).mean() + (1 - log_n.exp()).mean()
        pi = paired[torch.randint(0, paired.numel(), (min(b, paired.numel()),), device=paired.device)].to(device)
        pe = head(e[pair_expert_idx.to(device)[pi]]).clamp(1e-6, 1 - 1e-6)
        pn = head(n[pi]).clamp(1e-6, 1 - 1e-6)
        rank = -F.logsigmoid(torch.logit(pe) - torch.logit(pn)).mean()
        loss = -likelihood + cfg.sparsity_coeff * sparsity + rank_weight * rank
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), cfg.grad_clip)
        opt.step()
    with torch.no_grad():
        return {
            "loss": float(loss),
            "rank_loss": float(rank),
            "mean_phi_expert": float(head(e).mean()),
            "mean_phi_nominal": float(head(n).mean()),
            "n_expert": int(e.shape[0]),
            "n_nominal": int(n.shape[0]),
            "n_pairs": int(paired.numel()),
        }


def inverse_step(phi, expert_emb, nominal_emb, pair_expert_idx, ic, mcfg, memory=None):
    """One ICRL inverse step, by ic.inverse_method; returns (stats, new PUCL memory or None)."""
    from src.icrl_dual_training.maxent_constraint import pucl_constraint_update

    method = str(ic.get("inverse_method", "maxent"))
    if method not in INVERSE_METHODS:
        raise ValueError(f"icrl.inverse_method must be one of {INVERSE_METHODS}, got {method}")
    if method == "maxent":
        return maxent_round(phi, expert_emb, nominal_emb, mcfg), None
    if method == "maxent_paired":
        return paired_maxent_update(
            phi.head, expert_emb, nominal_emb, pair_expert_idx, mcfg, float(ic.rank_weight)
        ), None
    # PUCL (Peng & Billard 2024): reliable negatives by distance to the demonstrations,
    # cross-entropy against them, memory of past rounds' negatives. One-step
    # trajectories, so no per-trajectory expansion.
    stats, mem = pucl_constraint_update(
        phi.head, one_step_batch(expert_emb), one_step_batch(nominal_emb), mcfg,
        memory=memory, expand=False,
    )
    new_memory = mem if memory is None else __import__("torch").cat([memory, mem])
    return stats, new_memory
