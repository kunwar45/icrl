#!/usr/bin/env python
# ABOUTME: Builds the Countdown-Code data files: scored o4-mini demonstrations (the SFT/expert set), RL training problems and held-out test problems
# ABOUTME: Run on a login node (needs internet): python scripts/prepare_countdown_code_data.py --out $SCRATCH/icrl/countdown_code/data
"""
Follows the authors' datagen scripts (github.com/zohaib-khan5040/Countdown-Code):
the HF problem set `Jiayi-Pan/Countdown-Tasks-3to4` is shuffled with seed 42;
problems [0, 16000) were given to o4-mini (the released traces), [16000, 32000)
are the evaluation pool and [32000, 48000) the RL pool, of which 5,000 are
sampled (seed 42) and split 4,000 train / 1,000 validation.

Outputs (jsonl):
  demonstrations.jsonl  traces whose proxy reward is 1 (the authors' outcome
                        filter). Columns proxy/true/hack are written for
                        AUDIT ONLY; no training stage may read true or hack.
  rl_train.jsonl, rl_validation.jsonl, test.jsonl   {"id", "nums", "target"}

The traces file has no licence in the upstream repository: keep it on the
cluster, do not redistribute it.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.environments.countdown_code.countdown_code_adapter import (  # noqa: E402
    format_demonstration,
    score_replies,
)

TRACES_URL = "https://raw.githubusercontent.com/zohaib-khan5040/Countdown-Code/main/o4-mini-distillation-16k.jsonl"
PROBLEMS = "Jiayi-Pan/Countdown-Tasks-3to4"


def write(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"wrote {path} ({len(rows)} rows)")


def score_traces(traces: Path) -> tuple[list[dict], int, int, int]:
    """Proxy-passing traces with audit columns; returns (demos, n traces, n unparseable, n hacks among kept)."""
    items, replies, dropped = [], [], 0
    for i, line in enumerate(traces.read_text().splitlines()):
        row = json.loads(line)
        out = row.get("output")
        demo = format_demonstration(out) if isinstance(out, dict) else None
        if demo is None:
            dropped += 1
            continue
        items.append(
            {
                "id": f"demo-{i}",
                "nums": row["input"]["nums"],
                "target": row["input"]["target"],
            }
        )
        replies.append(demo)
    scores = score_replies(items, replies, workers=32)
    demos = [
        {
            **it,
            "response": rep,
            "proxy": s["proxy"],
            "true": s["true"],
            "hack": s["hack"],
        }
        for it, rep, s in zip(items, replies, scores)
        if s["proxy"] == 1
    ]
    n_hack = sum(d["hack"] for d in demos)
    print(
        f"traces: {len(items) + dropped} | unparseable: {dropped} | proxy=1 kept: {len(demos)} | "
        f"hacks among kept: {n_hack} ({n_hack / max(1, len(demos)):.2%}) [paper: 15,599 kept, ~1.2% hacks]"
    )
    return demos, len(items) + dropped, dropped, n_hack


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--test-problems", type=int, default=500)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    traces = a.out / "o4-mini-distillation-16k.jsonl"
    if not traces.exists():
        print("downloading", TRACES_URL)
        urllib.request.urlretrieve(TRACES_URL, traces)

    demos_path = a.out / "demonstrations.jsonl"
    if demos_path.exists():  # scoring 16k traces takes a while; reuse a finished file
        demos = [
            json.loads(x) for x in demos_path.read_text().splitlines() if x.strip()
        ]
        n_total, dropped, n_hack = None, None, sum(d["hack"] for d in demos)
        print(
            f"reusing {demos_path}: {len(demos)} demonstrations, {n_hack} hacks ({n_hack / len(demos):.2%})"
        )
    else:
        demos, n_total, dropped, n_hack = score_traces(traces)
        write(demos_path, demos)

    from datasets import load_dataset

    ds = load_dataset(PROBLEMS, split="train").shuffle(seed=42)
    rl = (
        ds.select(range(32000, 48000))
        .shuffle(seed=42)
        .select(range(5000))
        .train_test_split(test_size=0.2, seed=42)
    )
    for name, part in (
        ("rl_train", rl["train"]),
        ("rl_validation", rl["test"]),
        ("test", ds.select(range(16000, 16000 + a.test_problems))),
    ):
        write(
            a.out / f"{name}.jsonl",
            [
                {
                    "id": f"{name}-{i}",
                    "nums": list(r["nums"]),
                    "target": int(r["target"]),
                }
                for i, r in enumerate(part)
            ],
        )
    (a.out / "summary.json").write_text(
        json.dumps(
            {
                "traces": n_total,
                "unparseable": dropped,
                "demonstrations": len(demos),
                "demo_hacks": n_hack,
                "source_traces": TRACES_URL,
                "source_problems": PROBLEMS,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
