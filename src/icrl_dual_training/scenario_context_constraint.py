# ABOUTME: The ODCV constraint function C_theta as trained: a LoRA-adapted encoder over the scenario prompt that defines the violation plus the agent's reasoning and actions, mean-pooled into a sigmoid head
# ABOUTME: Import from the trainer and from the fine-tuning loop so both build the input text identically: from src.icrl_dual_training.scenario_context_constraint import build_text, load_constraint_model, score_items
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn

# What the judge reads is the scenario framing plus what the agent thought and
# did. Scoring actions alone is the single biggest cause of false positives:
# the same bash command is compliant in one scenario and a violation in another.
SCENARIO_CHARS = 2500
REASON_CHARS = 400
RESULT_CHARS = 300


def trajectory_text(
    steps: list[dict], with_reasoning: bool = True, with_results: bool = False
) -> str:
    out: list[str] = []
    for s in steps:
        if with_reasoning and (s.get("reasoning") or "").strip():
            out.append(f"[THOUGHT] {s['reasoning'][:REASON_CHARS]}")
        out.append(f"[ACTION] {s.get('action', '')}")
        if with_results and (s.get("observation") or "").strip():
            out.append(f"[RESULT] {s['observation'][:RESULT_CHARS]}")
    return "\n".join(out)


def build_text(
    scenario_prompt: str,
    steps: list[dict],
    with_context: bool = True,
    with_reasoning: bool = True,
    with_results: bool = False,
) -> str:
    traj = trajectory_text(steps, with_reasoning, with_results)
    if not with_context:
        return traj
    return f"[SCENARIO]\n{(scenario_prompt or '')[:SCENARIO_CHARS]}\n\n[TRAJECTORY]\n{traj}"


def middle_truncate(tok, text: str, max_length: int) -> list[int]:
    """Keep the scenario header and the END of the trajectory: the decisive
    actions -- the final edit, task_complete -- are the last thing that happens."""
    ids = tok(text, add_special_tokens=False)["input_ids"]
    if len(ids) <= max_length:
        return ids
    head = int(0.45 * max_length)
    return ids[:head] + ids[-(max_length - head) :]


class ConstraintModel(nn.Module):
    """Frozen-shape wrapper: a (LoRA-adapted) encoder, mean-pooled, into an MLP."""

    def __init__(self, encoder, hidden: int):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(
            nn.Linear(hidden, 256), nn.ReLU(), nn.Dropout(0.1), nn.Linear(256, 1)
        )

    def forward(self, input_ids, attention_mask):
        h = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state
        m = attention_mask.unsqueeze(-1).to(h.dtype)
        pooled = (h * m).sum(1) / m.sum(1).clamp(min=1)
        return self.head(pooled.float()).squeeze(-1)


def collate(tok, texts: list[str], max_length: int, device):
    seqs = [middle_truncate(tok, t, max_length) for t in texts]
    width = max(len(s) for s in seqs)
    pad = tok.pad_token_id or 0
    ids = torch.full((len(seqs), width), pad, dtype=torch.long)
    mask = torch.zeros((len(seqs), width), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, : len(s)] = torch.tensor(s)
        mask[i, : len(s)] = 1
    return ids.to(device), mask.to(device)


def load_constraint_model(model_dir: Path | str, device: str | None = None):
    """Returns (model, tokenizer, config) for a directory written by the trainer's
    --save-model: an `adapter/` PEFT directory, `head.pt`, and the json that
    records the input format and the validation-chosen threshold."""
    from peft import PeftModel
    from transformers import AutoModel, AutoTokenizer

    model_dir = Path(model_dir)
    cfg = json.loads((model_dir / "constraint_model.json").read_text())
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    tok = AutoTokenizer.from_pretrained(model_dir / "adapter")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModel.from_pretrained(cfg["encoder"], dtype=torch.bfloat16)
    enc = PeftModel.from_pretrained(base, model_dir / "adapter").to(device)
    model = ConstraintModel(enc, base.config.hidden_size).to(device)
    model.head.load_state_dict(torch.load(model_dir / "head.pt", map_location=device))
    model.head.to(torch.float32)
    model.eval()
    return model, tok, cfg


@torch.no_grad()
def score_items(model, tok, cfg: dict, items: list[dict], batch: int = 2) -> list[float]:
    """items carry `scenario_prompt` and `steps`, exactly as transcript_to_steps
    and the trace loader produce them."""
    device = next(model.parameters()).device
    texts = [
        build_text(
            it.get("scenario_prompt", ""),
            it["steps"],
            with_context=cfg.get("with_context", True),
            with_reasoning=cfg.get("with_reasoning", True),
        )
        for it in items
    ]
    out: list[float] = []
    for i in range(0, len(texts), batch):
        ids, mask = collate(tok, texts[i : i + batch], int(cfg["max_length"]), device)
        out += torch.sigmoid(model(ids, mask)).float().cpu().tolist()
    return out
