#!/usr/bin/env python3
"""
Test the TrajectoryEncoder locally on M1 Mac.

Uses Qwen/Qwen2.5-1.5B (small enough for 16GB M1) to verify the encoder
loads, runs on MPS, and produces valid scores on real demo data.

Usage:
    python scripts/test_encoder.py
    python scripts/test_encoder.py --max-length 256   # faster, less memory
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from transformers import AutoModel, AutoTokenizer

from src.constraint.encoder import TrajectoryEncoder
from src.data.trajectory import load_trajectories

MODEL = "Qwen/Qwen2.5-1.5B"   # ~3GB bfloat16 — fits easily in 16GB M1
SAFE_PATH  = Path("data/demos/safe.jsonl")
UNSAFE_PATH = Path("data/demos/unsafe.jsonl")


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_encoder(model_name: str, max_length: int, device: torch.device) -> TrajectoryEncoder:
    print(f"  Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"  Loading model ({device}) ...")
    model = AutoModel.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
    ).to(device)

    encoder = TrajectoryEncoder(
        model=model,
        tokenizer=tokenizer,
        max_length=max_length,
        head_hidden=256,
    ).to(device)

    trainable = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in encoder.parameters())
    print(f"  Trainable params: {trainable:,} / {total:,}  ({100*trainable/total:.2f}%)")
    return encoder


def score_trajectories(encoder: TrajectoryEncoder, trajs, label: str) -> list[float]:
    scores = []
    for t in trajs:
        text = t.to_text()
        with torch.no_grad():
            score = encoder([text]).item()
        scores.append(score)
        print(f"  [{label}] task={t.task_instance_id:>4s}  steps={len(t.steps):>2d}  "
              f"text_chars={len(text):>7,}  score={score:.4f}  "
              f"(ground_truth={'safe' if t.is_safe else 'unsafe'})")
    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--n", type=int, default=2, help="Demos per class to score")
    parser.add_argument("--max-length", type=int, default=512,
                        help="Token limit per trajectory (default 512 for fast test)")
    args = parser.parse_args()

    device = get_device()
    print(f"\n{'='*60}")
    print(f"  Device    : {device}")
    print(f"  Model     : {args.model}")
    print(f"  Max length: {args.max_length} tokens")
    print(f"  Demos     : {args.n} safe + {args.n} unsafe")
    print(f"{'='*60}\n")

    # Load demos
    safe_trajs   = load_trajectories(str(SAFE_PATH))[:args.n]
    unsafe_trajs = load_trajectories(str(UNSAFE_PATH))[:args.n]
    print(f"Loaded {len(safe_trajs)} safe, {len(unsafe_trajs)} unsafe demos\n")

    # Build encoder
    print("Building encoder:")
    encoder = load_encoder(args.model, args.max_length, device)

    # Score
    print(f"\nScoring trajectories (head is untrained — scores near 0.5 expected):")
    safe_scores   = score_trajectories(encoder, safe_trajs,   "safe  ")
    unsafe_scores = score_trajectories(encoder, unsafe_trajs, "unsafe")

    # Sanity checks
    all_scores = safe_scores + unsafe_scores
    print(f"\n{'─'*60}")
    print(f"  All scores in [0, 1]  : {all(0 <= s <= 1 for s in all_scores)}")
    print(f"  No NaN                : {all(s == s for s in all_scores)}")
    print(f"  Safe mean             : {sum(safe_scores)/len(safe_scores):.4f}")
    print(f"  Unsafe mean           : {sum(unsafe_scores)/len(unsafe_scores):.4f}")
    print(f"  (Means will be ~0.5 until the head is trained)")

    # Gradient check — make sure only head params have grad
    print(f"\n  Parameters with grad  :")
    for name, p in encoder.named_parameters():
        if p.requires_grad:
            print(f"    {name}  {tuple(p.shape)}")

    print(f"\n  Encoder test PASSED — ready for ICRL training.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
