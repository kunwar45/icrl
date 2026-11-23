#!/usr/bin/env python
# ABOUTME: Builds the LeetCode anchor problems for GRAFT: medium and hard problems of LeetCodeDataset whose reference solution passes its own tests in our sandbox
# ABOUTME: Run on a login node (needs internet): python scripts/prepare_leetcode_anchor_data.py --out $SCRATCH/icrl/leetcode_anchor/data
"""
Source: the Hugging Face dataset `newfacade/LeetCodeDataset` (Xia et al. 2025,
arXiv 2504.14655; Apache-2.0), train split. GRAFT (Ward et al. 2026) uses the
medium and hard problems of this environment for its anchor steps.

Kept problems:
  * difficulty Medium or Hard
  * the test code is at most --max-test-chars long (a few problems carry
    hundreds of thousands of characters of tests, which would dominate the
    reward time)
  * the dataset's reference solution passes the tests in our sandbox within the
    limits the reward uses. A problem whose own solution fails here could never
    pay reward, so it is dropped.

Output: anchor_train.jsonl with {"id", "difficulty", "problem", "starter_code",
"entry_point", "test"} and manifest.json with the counts. The reference
solutions are NOT written out.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.environments.leetcode_anchor.leetcode_anchor_adapter import (  # noqa: E402
    passes_hidden_tests,
)

DATASET = "newfacade/LeetCodeDataset"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-test-chars", type=int, default=20000)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()

    from datasets import load_dataset

    ds = load_dataset(DATASET, split="train")
    print(f"{DATASET} train: {len(ds)} problems; columns {ds.column_names}", flush=True)
    counts = {"all": len(ds), "medium_or_hard": 0, "test_short_enough": 0, "reference_passes": 0}
    candidates = []
    for row in ds:
        if str(row["difficulty"]).lower() not in ("medium", "hard"):
            continue
        counts["medium_or_hard"] += 1
        if len(row["test"]) > a.max_test_chars:
            continue
        counts["test_short_enough"] += 1
        candidates.append(row)
    jobs = [{"solution": r["completion"], "test": r["test"], "entry_point": r["entry_point"]} for r in candidates]
    results = passes_hidden_tests(jobs, workers=a.workers)
    kept, errors = [], {}
    for row, res in zip(candidates, results):
        if res["passed"]:
            kept.append(
                {
                    "id": row["task_id"],
                    "difficulty": row["difficulty"],
                    "problem": row["problem_description"],
                    "starter_code": row["starter_code"],
                    "entry_point": row["entry_point"],
                    "test": row["test"],
                }
            )
        else:
            errors[res["error"]] = errors.get(res["error"], 0) + 1
    counts["reference_passes"] = len(kept)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "anchor_train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in kept))
    manifest = {
        "source": DATASET,
        "split": "train",
        "licence": "apache-2.0",
        "max_test_chars": a.max_test_chars,
        "counts": counts,
        "reference_failures_by_error": errors,
        "by_difficulty": {d: sum(1 for r in kept if r["difficulty"] == d) for d in sorted({r["difficulty"] for r in kept})},
    }
    (a.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2), flush=True)
    if kept:
        ex = kept[0]
        print(f"\nexample {ex['id']} ({ex['difficulty']}), entry point {ex['entry_point']}\n{ex['starter_code']}\n{ex['test'][:300]}")
    return 0 if kept else 1


if __name__ == "__main__":
    sys.exit(main())
