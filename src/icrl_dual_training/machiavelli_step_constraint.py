# ABOUTME: The MACHIAVELLI constraint C_theta(state, action): builds expert and unsafe step demos from collected episodes, trains the head with the ICRL constraint update, and scores options
# ABOUTME: Use: expert, unsafe = demo_steps(expert_eps, nominal_eps); head = train_step_constraint(cfg, expert, unsafe, out_dir); scores = StepConstraint.load(out_dir).score(texts)
"""
Demos, in the sense ICRL needs them:

  expert demos  actions taken by the policy model under the expert prompt that
                did NOT violate (ground truth from the benchmark's annotations)
  unsafe demos  actions taken by the same model under the nominal prompt that
                DID violate when a non-violating option existed at that state,
                i.e. avoidable violations. Forced violations (every option
                violates) are excluded: the story did it, not the agent.

Both halves come from the same model, on the same games, with the same output
format; only the goal paragraph of the prompt differs.

C_theta is a frozen Qwen encoder, mean-pooled over the end of the scene plus the
chosen option, with a small MLP head trained by
`constraint_trainer.constraint_update` (expert scores pushed down, unsafe
pushed up). The head is the only trained part.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np

from src.environments.machiavelli.machiavelli_adapter import constraint_text


def accept_expert_step(step: dict) -> bool:
    """An expert demo: a parsed answer under the expert prompt whose action did not violate."""
    return bool(step["valid_response"]) and not step["violated"]


def accept_unsafe_step(step: dict) -> bool:
    """An unsafe demo: a parsed answer under the nominal prompt that violated when a clean option existed."""
    return bool(step["valid_response"]) and bool(step["avoidable"])


def demo_steps(
    expert_eps: list[dict], nominal_eps: list[dict], *, scene_chars: int = 2000
) -> tuple[list[dict], list[dict]]:
    def rows(eps, accept):
        return [
            {
                "game": e["game"],
                "text": constraint_text(
                    s["scene"], s["choices"][s["action"]], scene_chars
                ),
            }
            for e in eps
            for s in e["steps"]
            if accept(s)
        ]

    return rows(expert_eps, accept_expert_step), rows(nominal_eps, accept_unsafe_step)


def option_rows(episodes: list[dict], *, scene_chars: int = 2000) -> list[dict]:
    """Every option at every recorded state, with its ground-truth violation: the held-out test set for C_theta."""
    rows = []
    for e in episodes:
        for s in e["steps"]:
            if s["forced"] or not any(s["option_harms"]):
                state_kind = "forced" if s["forced"] else "clean"
            else:
                state_kind = "mixed"
            for i, (txt, harms) in enumerate(zip(s["choices"], s["option_harms"])):
                rows.append(
                    {
                        "game": e["game"],
                        "state": f"{e['game']}|{e['seed']}|{s['t']}",
                        "state_kind": state_kind,
                        "option": i,
                        "violates": bool(harms),
                        "text": constraint_text(s["scene"], txt, scene_chars),
                    }
                )
    return rows


def auroc(scores, labels) -> float:
    pos = [s for s, l in zip(scores, labels) if l]
    neg = [s for s, l in zip(scores, labels) if not l]
    if not pos or not neg:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]))
    ranks = np.empty(len(order))
    ranks[order] = np.arange(1, len(order) + 1)
    return float(
        (ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2)
        / (len(pos) * len(neg))
    )


def within_state_accuracy(rows: list[dict], scores: list[float]) -> float:
    """At mixed states: how often a violating option scores above a clean one (ties count half)."""
    by_state: dict[str, list[tuple[float, bool]]] = {}
    for r, sc in zip(rows, scores):
        if r["state_kind"] == "mixed":
            by_state.setdefault(r["state"], []).append((sc, r["violates"]))
    wins = total = 0.0
    for opts in by_state.values():
        bad = [s for s, v in opts if v]
        good = [s for s, v in opts if not v]
        for b in bad:
            for g in good:
                total += 1
                wins += 1.0 if b > g else 0.5 if b == g else 0.0
    return wins / total if total else float("nan")


class StepConstraint:
    """Frozen encoder + trained head; scores constraint texts in [0, 1] (high = violation)."""

    def __init__(self, encoder):
        self.encoder = encoder

    @classmethod
    def build(
        cls,
        encoder_model: str,
        max_length: int,
        head_hidden: int,
        device: str | None = None,
    ):
        import torch

        from src.trajectory_embedding.trajectory_encoder import load_qwen_encoder

        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        enc = load_qwen_encoder(
            encoder_model, max_length=max_length, head_hidden=head_hidden, device=device
        )
        enc.text_mode = "machiavelli_step"
        enc.head.to(next(enc.backbone.parameters()).device)
        return cls(enc)

    @classmethod
    def load(
        cls, out_dir: Path, encoder_model: str | None = None, device: str | None = None
    ):
        meta = json.loads((Path(out_dir) / "constraint_meta.json").read_text())
        obj = cls.build(
            encoder_model or meta["encoder_model"],
            meta["max_length"],
            meta["head_hidden"],
            device=device,
        )
        import torch

        obj.encoder.head.load_state_dict(
            torch.load(Path(out_dir) / "constraint_head.pt", map_location="cpu")
        )
        obj.encoder.head.to(next(obj.encoder.backbone.parameters()).device)
        return obj

    def embed(self, texts: list[str], batch_size: int = 16):
        return self.encoder.embed_texts(texts, batch_size=batch_size)

    def score_embeddings(self, emb) -> list[float]:
        import torch

        with torch.no_grad():
            dev = next(self.encoder.head.parameters()).device
            return self.encoder.head(emb.to(dev)).squeeze(-1).float().cpu().tolist()

    def score(self, texts: list[str], batch_size: int = 16) -> list[float]:
        if not texts:
            return []
        return self.score_embeddings(self.embed(texts, batch_size))


def train_step_constraint(
    cfg, expert: list[dict], unsafe: list[dict], out_dir: Path, log=print
) -> StepConstraint:
    """Fit the head on cached embeddings; save head + metadata to out_dir."""
    import torch

    from src.icrl_dual_training.constraint_trainer import constraint_update

    c = cfg.constraint
    model = StepConstraint.build(
        str(c.encoder_model),
        int(c.max_length),
        int(c.head_hidden),
        device=c.get("device"),
    )
    rng = random.Random(int(c.seed))
    cap = int(c.max_demos_per_class)
    expert = rng.sample(expert, min(cap, len(expert)))
    unsafe = rng.sample(unsafe, min(cap, len(unsafe)))
    log(f"constraint demos: {len(expert)} expert steps, {len(unsafe)} unsafe steps")
    if not expert or not unsafe:
        raise RuntimeError("constraint training needs both expert and unsafe demos")
    torch.manual_seed(int(c.seed))
    e_emb = model.embed([d["text"] for d in expert])
    u_emb = model.embed([d["text"] for d in unsafe])
    hist = []
    for it in range(int(c.n_iterations)):
        m = constraint_update(
            model.encoder,
            policy_embeddings=u_emb,
            expert_embeddings=e_emb,
            beta=float(c.beta),
            lambda_c=float(c.lambda_c),
            n_steps=int(c.n_constraint_steps),
            lr=float(c.lr),
            weight_decay=float(c.weight_decay),
            batch_size=int(c.batch_size),
            loss_kind=str(c.loss),
        )
        hist.append(m)
        log(
            f"constraint iter {it}: {json.dumps({k: round(v, 4) for k, v in m.items()})}"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.encoder.head.state_dict(), out_dir / "constraint_head.pt")
    (out_dir / "constraint_meta.json").write_text(
        json.dumps(
            {
                "encoder_model": str(c.encoder_model),
                "max_length": int(c.max_length),
                "head_hidden": int(c.head_hidden),
                "loss": str(c.loss),
                "n_expert": len(expert),
                "n_unsafe": len(unsafe),
                "history": hist,
            },
            indent=2,
        )
    )
    return model


def evaluate_step_constraint(model: StepConstraint, rows: list[dict]) -> dict:
    """AUROC over all options, AUROC over options at mixed states, and within-state ranking accuracy."""
    scores = model.score([r["text"] for r in rows])
    labels = [r["violates"] for r in rows]
    mixed = [i for i, r in enumerate(rows) if r["state_kind"] == "mixed"]
    return {
        "n_options": len(rows),
        "n_violating": int(sum(labels)),
        "auroc_all_options": auroc(scores, labels),
        "auroc_mixed_states": auroc(
            [scores[i] for i in mixed], [labels[i] for i in mixed]
        ),
        "within_state_violating_above_clean": within_state_accuracy(rows, scores),
        "mean_score_violating": float(np.mean([s for s, l in zip(scores, labels) if l]))
        if any(labels)
        else None,
        "mean_score_clean": float(np.mean([s for s, l in zip(scores, labels) if not l]))
        if not all(labels)
        else None,
    }
