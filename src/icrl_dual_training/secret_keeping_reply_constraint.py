# ABOUTME: ICRL constraint for secret-keeping chats: oracle-expert vs policy demos, the one-shot classifier baseline, and the max-entropy (Malik et al. 2021) feasibility C_theta(tau) = prod_t phi(s_t, a_t) over chat trajectories of any length, used by the alternating ICRL loop
# ABOUTME: Use: expert, unsafe = oracle_demo_pairs(rows, pair_filter="higher_return"); train with machiavelli_step_constraint.train_step_constraint; reply_detector_eval(model, rows)
"""
One demo row per training prompt holds 4 expert-prompt replies and 4 policy
(nominal-prompt) replies, each with its judge reward. The oracle expert knows
the rule: of its replies it keeps the highest-reward one that does not leak.
Unsafe demos are the policy's replies on the same prompt that beat the expert
reply's reward (`higher_return`) or simply differ from it (`any_different`).
Neither filter reads a leak label; leak labels on policy replies are recorded
for reporting only.

A chat is a trajectory tau = (s_1, a_1, ..., s_T, a_T); single-turn secret
keeping is T = 1. Each step's text holds the nominal system prompt, the
conversation so far and the reply, so the expert and policy trajectories of a
pair differ only in their replies.

Two constraint models live here. TrajectoryFeasibility + inverse_step is the
ICRL one (Malik et al. 2021 max-entropy update against the CURRENT policy's
samples, used by the icrl_loop stage). oracle_demo_pairs feeds the BASELINE one-shot
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
    """The encoder text of a T = 1 chat step; equals single_turn_trajectory(item, reply)[0]."""
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


# ---- Max-entropy ICRL (Malik et al. 2021) feasibility over chat trajectories ----
#
# A chat trajectory is tau = (s_1, a_1, ..., s_T, a_T): s_t is the system prompt,
# the earlier turns and user message t; a_t is assistant reply t. The constraint
# factorises over steps (Malik et al. 2021; survey Section 4.2),
#
#     C_theta(tau) = prod_t phi(s_t, a_t),   phi(s, a) = sigmoid(f(enc(s, a))),
#
# with cost 1 - C_theta(tau). Single-turn secret keeping is the case T = 1, where
# C_theta(tau) = phi(x, y). phi is fitted by maxent_constraint.maxent_constraint_update
# (raise feasibility on expert trajectories, lower it on the CURRENT policy's,
# sparsity regulariser toward feasible), which already sums log phi per trajectory.


def step_texts(item: dict, turns: list[tuple[str, str]]) -> list[str]:
    """One encoder text per step of a chat trajectory; turns[t] = (user message t, reply t).

    Step t reads the nominal system prompt, the earlier turns, user message t and
    reply t. For T = 1 this is exactly constraint_text(item, reply).
    """
    head = f"SYSTEM: {system_prompt('nominal', item)}"
    out, history = [], ""
    for user, reply in turns:
        out.append(f"{head}\n{history}USER: {user}\nREPLY: {reply}")
        history += f"USER: {user}\nREPLY: {reply}\n"
    return out


def single_turn_trajectory(item: dict, reply: str) -> list[str]:
    """The T = 1 trajectory of a single-turn chat: one step, (item's user message, reply)."""
    return step_texts(item, [(item["user"], reply)])


def trajectory_batch(step_embeddings, lengths):
    """A TrajectoryBatch from per-step embeddings, trajectory i owning the next lengths[i] rows."""
    import torch

    from src.icrl_dual_training.maxent_constraint import TrajectoryBatch

    lengths = torch.as_tensor(lengths, dtype=torch.long)
    if (lengths < 1).any():
        raise ValueError("every trajectory needs at least one step")
    if int(lengths.sum()) != step_embeddings.size(0):
        raise ValueError(
            f"lengths sum to {int(lengths.sum())} but there are {step_embeddings.size(0)} step embeddings"
        )
    owner = torch.repeat_interleave(torch.arange(lengths.numel()), lengths)
    return TrajectoryBatch(
        embeddings=step_embeddings,
        owner=owner.to(step_embeddings.device),
        n_trajectories=int(lengths.numel()),
    )


def trajectory_lengths(batch):
    """Number of steps in each trajectory of a TrajectoryBatch."""
    import torch

    return torch.bincount(batch.owner, minlength=batch.n_trajectories)


class TrajectoryFeasibility:
    """C_theta(tau) = prod_t phi(s_t, a_t): frozen encoder + StepFeasibilityHead phi; cost = 1 - C_theta."""

    def __init__(self, encoder, head, meta: dict):
        self.encoder, self.head, self.meta = encoder, head, meta

    @classmethod
    def build(cls, encoder_model: str, max_length: int, head_hidden: int, device=None):
        from src.icrl_dual_training.machiavelli_step_constraint import StepConstraint
        from src.icrl_dual_training.maxent_constraint import StepFeasibilityHead

        enc = StepConstraint.build(
            encoder_model, max_length, head_hidden, device=device
        )
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
            meta["encoder_model"],
            meta["max_length"],
            meta["head_hidden"],
            device=device,
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

    def embed(self, trajectories, batch_size: int = 16):
        """TrajectoryBatch of step embeddings; each trajectory is a list of step texts (a str is T = 1)."""
        trajs = [[t] if isinstance(t, str) else list(t) for t in trajectories]
        flat = [s for t in trajs for s in t]
        emb = self.encoder.embed(flat, batch_size)
        return trajectory_batch(emb, [len(t) for t in trajs])

    def costs(self, batch) -> list[float]:
        """1 - C_theta(tau) = 1 - prod_t phi(s_t, a_t) per trajectory (high = infeasible)."""
        import torch

        from src.icrl_dual_training.maxent_constraint import trajectory_log_feasibility

        with torch.no_grad():
            dev = next(self.head.parameters()).device
            log_c = trajectory_log_feasibility(self.head, batch.to(dev))
            return (1.0 - log_c.exp()).float().cpu().tolist()

    def score(self, trajectories, batch_size: int = 16) -> list[float]:
        """Cost per trajectory; a plain str is a T = 1 trajectory (the interface reply_detector_eval uses)."""
        if not trajectories:
            return []
        return self.costs(self.embed(trajectories, batch_size))


INVERSE_METHODS = ("maxent", "maxent_paired", "pucl")


def _logit_from_log(log_c):
    """logit C from log C, i.e. log C - log(1 - C); finite because phi is clamped below 1."""
    import torch

    return log_c - torch.log1p(-log_c.exp())


def paired_maxent_update(
    head, expert, nominal, pair_expert_idx, cfg, rank_weight: float
) -> dict:
    """Malik et al. max-entropy update plus a same-context ranking term, over trajectories.

    The plain update can lower feasibility on a whole CONTEXT (every reply to an
    attack chat) instead of on the trajectories the expert avoids, which leaves no
    contrast inside a GRPO group. The ranking term
    -log sigma(logit C(expert tau) - logit C(policy tau)), over pairs from the same
    chat, forces the feasibility to differ between trajectories from one context.
    expert and nominal are TrajectoryBatches; pair_expert_idx[i] is the expert
    trajectory for nominal trajectory i's chat, or -1.
    """
    import torch
    import torch.nn.functional as F

    from src.icrl_dual_training.maxent_constraint import trajectory_log_feasibility

    device = next(head.parameters()).device
    e, n = expert.to(device), nominal.to(device)
    pair = pair_expert_idx.to(device)
    paired = (pair >= 0).nonzero(as_tuple=True)[0]
    opt = torch.optim.AdamW(head.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    b = cfg.batch_trajectories
    for _ in range(cfg.n_steps):
        # log C for every trajectory, then index: keeps pair order and repeated draws
        lc_e = trajectory_log_feasibility(head, e)
        lc_n = trajectory_log_feasibility(head, n)
        ei = torch.randint(
            0, e.n_trajectories, (min(b, e.n_trajectories),), device=device
        )
        ni = torch.randint(
            0, n.n_trajectories, (min(b, n.n_trajectories),), device=device
        )
        log_e, log_n = lc_e[ei], lc_n[ni]
        likelihood = cfg.beta * (log_e.mean() - log_n.mean())
        sparsity = (1 - log_e.exp()).mean() + (1 - log_n.exp()).mean()
        pi = paired[
            torch.randint(0, paired.numel(), (min(b, paired.numel()),), device=device)
        ]
        rank = -F.logsigmoid(
            _logit_from_log(lc_e[pair[pi]]) - _logit_from_log(lc_n[pi])
        ).mean()
        loss = -likelihood + cfg.sparsity_coeff * sparsity + rank_weight * rank
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), cfg.grad_clip)
        opt.step()
    with torch.no_grad():
        return {
            "loss": float(loss),
            "rank_loss": float(rank),
            "mean_C_expert": float(trajectory_log_feasibility(head, e).exp().mean()),
            "mean_C_nominal": float(trajectory_log_feasibility(head, n).exp().mean()),
            "n_expert": int(e.n_trajectories),
            "n_nominal": int(n.n_trajectories),
            "n_pairs": int(paired.numel()),
        }


def inverse_step(
    phi,
    expert,
    nominal,
    pair_expert_idx,
    ic,
    mcfg,
    memory=None,
    heldout=None,
    return_selected=False,
):
    """One ICRL inverse step on TrajectoryBatches, by ic.inverse_method; returns (stats, new PUCL memory or None).

    The new memory is the old one with this round's reliable negatives appended at the end.
    `heldout` (held-out demonstration step embeddings) is needed by mcfg.pucl_selection='heldout_quantile'.
    With `return_selected`, PUCL's stats keep "selected_indices" (into `nominal`) for audits.
    """
    import torch

    from src.icrl_dual_training.maxent_constraint import (
        maxent_constraint_update,
        pucl_constraint_update,
    )

    method = str(ic.get("inverse_method", "maxent"))
    if method not in INVERSE_METHODS:
        raise ValueError(
            f"icrl.inverse_method must be one of {INVERSE_METHODS}, got {method}"
        )
    if method == "maxent":
        return maxent_constraint_update(phi.head, expert, nominal, mcfg), None
    if method == "maxent_paired":
        return paired_maxent_update(
            phi.head, expert, nominal, pair_expert_idx, mcfg, float(ic.rank_weight)
        ), None
    # PUCL (Peng & Billard 2024): reliable negative steps by distance to the
    # demonstrations, cross-entropy against them, memory of past rounds' negatives.
    # Eq. 4's expansion adds one step per policy trajectory; with T = 1 that would add
    # every step, so it runs only when some trajectory has more than one step.
    expand = bool((trajectory_lengths(nominal) > 1).any())
    stats, mem = pucl_constraint_update(
        phi.head, expert, nominal, mcfg, memory=memory, expand=expand, heldout=heldout
    )
    selected = stats.pop("selected_indices", None)
    if return_selected:
        stats["selected_indices"] = selected
    new_memory = mem if memory is None else torch.cat([memory, mem])
    return stats, new_memory
