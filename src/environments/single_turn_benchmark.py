# ABOUTME: The seam between single-turn benchmarks and the benchmark-neutral training code: one object per benchmark giving prompts, the training reward, the audit-only violation label and the monitor's input text
# ABOUTME: Use: bench = get_benchmark(cfg.benchmark); items = bench.load_items(cfg, "train"); bench.chat_messages(item); bench.score_replies(items, replies); bench.monitor_text(item, reply); bench.monitor_flags(items, replies)
"""
A single-turn benchmark is a set of prompts with one reply each. Training code
(GRPO, the ICRL loop, the VARL discriminator) only talks to this interface, so
nothing benchmark-specific lives outside an adapter or a config.

score_replies returns one dict per reply with
  reward     the TRAINING reward in [0, 1] (may be gameable)
  true       the intended-task reward, evaluation only
  violation  1 if the reply breaks the unstated rule (audit and evaluation only;
             no training stage may read it)
"""

from __future__ import annotations

from pathlib import Path


class CountdownCodeBenchmark:
    name = "countdown_code"
    item_columns = ("id", "nums", "target")

    def load_items(self, cfg, split: str) -> list[dict]:
        from src.environments.countdown_code.countdown_code_adapter import load_problems

        d = cfg.data
        files = {
            "train": ("rl_train.jsonl", d.get("train_prompts")),
            "test": ("test.jsonl", d.get("test_prompts")),
            "demonstrations": ("demonstrations.jsonl", d.get("demonstrations")),
        }
        name, n = files[split]
        return load_problems(Path(str(d.dir)) / name, int(n) if n else None)

    def chat_messages(self, item: dict) -> list[dict]:
        from src.environments.countdown_code.countdown_code_adapter import chat_messages

        return chat_messages(item)

    def score_replies(self, items: list[dict], replies: list[str]) -> list[dict]:
        from src.environments.countdown_code.countdown_code_adapter import score_replies

        return [
            {
                "reward": float(s["proxy"]),
                "true": float(s["true"]),
                "violation": int(s["hack"]),
                "parsed": s["parsed"],
            }
            for s in score_replies(items, replies)
        ]

    def monitor_text(self, item: dict, reply: str) -> str:
        from src.environments.countdown_code.countdown_code_adapter import monitor_text

        return monitor_text(reply)

    def constraint_text(self, item: dict, reply: str, mode: str = "files") -> str:
        """What a learned constraint reads. "files": the returned files (monitor_text). "scaffold_changes":
        only what the reply changed in the provided files, with the answer masked."""
        from src.environments.countdown_code import countdown_code_adapter as adapter

        if mode == "files":
            return adapter.monitor_text(reply)
        if mode == "scaffold_changes":
            return adapter.scaffold_changes_text(item, reply)
        raise ValueError(f"constraint.input must be files or scaffold_changes, got {mode!r}")

    def monitor_flags(self, items: list[dict], replies: list[str]) -> list[int]:
        """The benchmark's hand-written monitor (GRAFT): 1 for replies it flags. It sees only part of the hacks."""
        from src.environments.countdown_code.countdown_code_adapter import (
            trivial_test_flags,
        )

        return trivial_test_flags(items, replies)


BENCHMARKS = {"countdown_code": CountdownCodeBenchmark}


def get_benchmark(name: str):
    if name not in BENCHMARKS:
        raise ValueError(f"benchmark must be one of {sorted(BENCHMARKS)}, got {name!r}")
    return BENCHMARKS[name]()
