#!/usr/bin/env python3
"""
Pre-compute and save trajectory embeddings using a frozen Qwen backbone.

Reads trajectory files (task_*_trace_*.json from collect_safe_trajectories.py,
or *.jsonl from data/demos/), encodes each with mean-pooled last hidden states,
and saves a .pt bundle that the ICRL trainer can load directly without re-running
the backbone.

Usage:
    # Local test (CPU/MPS, auto-selects Qwen/Qwen2.5-1.5B):
    python scripts/encode_trajectories.py \\
        --input-dir trajectories/safe \\
        --label safe \\
        --output embeddings/safe.pt

    # On SLURM (set by the job script, uses all visible GPUs):
    python scripts/encode_trajectories.py \\
        --input-dir $SCRATCH/trajectories/safe \\
        --label safe \\
        --output $SCRATCH/embeddings/safe.pt \\
        --model Qwen/Qwen2.5-1.5B \\
        --max-length 2048 \\
        --batch-size 16 \\
        --flash-attn

Output .pt file contains a dict:
    {
        "embeddings":  (N, H) float32 tensor,
        "task_ids":    [str, ...],
        "is_safe":     [bool, ...],
        "label":       "safe" | "unsafe",
        "model_name":  str,
        "max_length":  int,
        "n_tokens_truncated": int,   # trajectories that hit the token cap
    }

Load with:
    bundle = torch.load("embeddings/safe.pt", weights_only=False)
    embs   = bundle["embeddings"]   # (N, H)
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Text serialisation ────────────────────────────────────────────────────────

def traj_json_to_text(traj: dict) -> str:
    """
    Convert a task_*_trace_*.json dict to a single string for the encoder.

    Format:
        [GOAL] <intent>
        [POLICIES] policy1 | policy2 | ...
        [STEP 0] [OBS] <axtree> [ACTION] <action>
        [STEP 1] ...
    """
    parts: list[str] = []

    # Pull goal from steps[0].observation if available, otherwise leave empty
    goal = traj.get("goal", "")
    if not goal and traj.get("steps"):
        # steps[0].observation is the AXTree; the goal is in the env, not here
        # Use task_id as a minimal identifier
        goal = f"Task {traj.get('task_id', '')}"
    if goal:
        parts.append(f"[GOAL] {goal}")

    # Policy text — concatenate descriptions across all three levels
    policies = traj.get("policies", [])
    if policies:
        policy_strs = []
        for p in policies:
            src = p.get("source", "")
            desc = p.get("description", "")
            if desc:
                policy_strs.append(f"({src}) {desc}")
        if policy_strs:
            parts.append("[POLICIES] " + " | ".join(policy_strs))

    # Steps
    for step in traj.get("steps", []):
        idx = step.get("step_idx", "?")
        obs = step.get("observation", "").strip()
        action = step.get("action", "").strip()
        parts.append(f"[STEP {idx}] [OBS] {obs} [ACTION] {action}")

    return "\n".join(parts)


def jsonl_traj_to_text(traj: dict) -> str:
    """
    Convert an old-format JSONL trajectory dict (from data/demos/) to text.
    Mirrors Trajectory.to_text().
    """
    parts: list[str] = []
    for step in traj.get("steps", []):
        action = step.get("action", "")
        obs = step.get("observation", "")
        parts.append(f"[ACTION] {action}")
        parts.append(f"[OBS] {obs}")
    return " ".join(parts)


# ── File loading ──────────────────────────────────────────────────────────────

def load_traj_jsons(input_dir: Path, label: str) -> list[dict]:
    """Load task_*_trace_*.json files. Each file = one trajectory."""
    files = sorted(input_dir.glob("task_*_trace_*.json"))
    if not files:
        logger.warning("No task_*_trace_*.json files found in %s", input_dir)
        return []

    trajs = []
    for p in files:
        try:
            with open(p) as f:
                d = json.load(f)
            d["_source_file"] = str(p)
            d["_is_safe"] = label == "safe"
            trajs.append(d)
        except Exception as e:
            logger.warning("Skipping %s: %s", p, e)
    logger.info("Loaded %d trajectory JSON files from %s", len(trajs), input_dir)
    return trajs


def load_traj_jsonls(jsonl_path: Path, is_safe: bool) -> list[dict]:
    """Load *.jsonl trajectories from data/demos/."""
    trajs = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                d["_source_file"] = str(jsonl_path)
                d["_is_safe"] = is_safe
                trajs.append(d)
            except json.JSONDecodeError as e:
                logger.warning("Bad JSONL line: %s", e)
    logger.info("Loaded %d JSONL trajectories from %s", len(trajs), jsonl_path)
    return trajs


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Pre-compute trajectory embeddings with a frozen Qwen backbone",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input-dir", type=Path, default=None,
                        help="Directory of task_*_trace_*.json files (new format)")
    parser.add_argument("--jsonl", type=Path, default=None,
                        help="Path to a *.jsonl file (old data/demos/ format)")
    parser.add_argument("--label", choices=["safe", "unsafe"], required=True,
                        help="Whether these trajectories are safe or unsafe demos")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output .pt file path")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B",
                        help="HuggingFace model ID (default: Qwen/Qwen2.5-1.5B)")
    parser.add_argument("--max-length", type=int, default=2048,
                        help="Token limit per trajectory (default: 2048)")
    parser.add_argument("--head-hidden", type=int, default=256,
                        help="MLP head hidden size (default: 256)")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Encoding batch size (default: 8; raise for multi-GPU)")
    parser.add_argument("--device", default=None,
                        help="Force device: cuda / cpu / mps (default: auto)")
    parser.add_argument("--flash-attn", action="store_true",
                        help="Use flash_attention_2 (A100/H100 only)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Load model and print stats without encoding")
    args = parser.parse_args()

    if args.input_dir is None and args.jsonl is None:
        parser.error("Provide --input-dir (new JSON format) or --jsonl (old JSONL format)")

    import torch
    from src.constraint.encoder import load_qwen_encoder

    # ── Load model ────────────────────────────────────────────────────────────
    attn = "flash_attention_2" if args.flash_attn else None
    encoder = load_qwen_encoder(
        model_name=args.model,
        max_length=args.max_length,
        head_hidden=args.head_hidden,
        device=args.device,
        attn_implementation=attn,
    )
    encoder.eval()

    backbone_device = next(encoder.backbone.parameters()).device
    logger.info("Backbone device: %s  bfloat16: %s",
                backbone_device,
                next(encoder.backbone.parameters()).dtype == torch.bfloat16)

    if args.dry_run:
        # Encode one dummy text to verify the pipeline end-to-end
        dummy = "[GOAL] test [STEP 0] [OBS] hello [ACTION] click('1')"
        with torch.no_grad():
            emb = encoder._mean_pool([dummy])
        logger.info("Dry-run embedding shape: %s  dtype: %s", tuple(emb.shape), emb.dtype)
        logger.info("Dry-run OK — exiting without writing output.")
        return

    # ── Load trajectories ─────────────────────────────────────────────────────
    if args.input_dir is not None:
        trajs = load_traj_jsons(args.input_dir, args.label)
        texts = [traj_json_to_text(t) for t in trajs]
    else:
        is_safe = args.label == "safe"
        trajs = load_traj_jsonls(args.jsonl, is_safe)
        texts = [jsonl_traj_to_text(t) for t in trajs]

    if not trajs:
        logger.error("No trajectories found — nothing to encode.")
        sys.exit(1)

    logger.info("Trajectories to encode: %d", len(trajs))

    # Count trajectories that will hit the token cap (rough estimate: 4 chars/token)
    cap_chars = args.max_length * 4
    n_truncated = sum(1 for t in texts if len(t) > cap_chars)
    logger.info("Approx. truncated (>%d chars): %d / %d",
                cap_chars, n_truncated, len(texts))

    # ── Encode ───────────────────────────────────────────────────────────────
    logger.info("Encoding in batches of %d ...", args.batch_size)
    with torch.no_grad():
        embeddings = encoder.embed_texts(texts, batch_size=args.batch_size)

    # Move to CPU for saving (safe across all setups)
    embeddings_cpu = embeddings.float().cpu()
    logger.info("Embeddings shape: %s", tuple(embeddings_cpu.shape))

    # ── Save bundle ───────────────────────────────────────────────────────────
    args.output.parent.mkdir(parents=True, exist_ok=True)

    task_ids = [str(t.get("task_id", t.get("task_instance_id", "?"))) for t in trajs]
    is_safe_flags = [bool(t.get("_is_safe", args.label == "safe")) for t in trajs]

    bundle = {
        "embeddings": embeddings_cpu,
        "task_ids": task_ids,
        "is_safe": is_safe_flags,
        "label": args.label,
        "model_name": args.model,
        "max_length": args.max_length,
        "n_tokens_truncated": n_truncated,
        "texts": texts,  # keep so you can sanity-check what was encoded
    }
    torch.save(bundle, args.output)
    logger.info("Saved → %s  (%d embeddings, dim=%d)",
                args.output, len(trajs), embeddings_cpu.shape[1])

    # Quick stats
    mean_norm = embeddings_cpu.norm(dim=1).mean().item()
    logger.info("Embedding L2 norm mean: %.3f  (typical Qwen2.5-1.5B: ~100-150)",
                mean_norm)


if __name__ == "__main__":
    main()
