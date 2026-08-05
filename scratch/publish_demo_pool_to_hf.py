"""One-off: hand-publish the standing SuiteCRM demo pool + splits to Hugging Face.

The per-run `publish` stage of scripts/run_experiment.py covers run artifacts;
this covers the run-independent demo pool (data/demos + data/train|eval) that
every run consumes. Public, under HF_ORG (icrl-finetuning).

Run from the repo root:  python scratch/publish_demo_pool_to_hf.py [--dry-run]
                         python scratch/publish_demo_pool_to_hf.py --restore
                         # --restore downloads the pool from HF back into
                         # data/ at the paths the pipeline expects (the local
                         # copies were deleted once verified on HF, 2026-08-05)
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from huggingface_hub import HfApi, get_token  # noqa: E402

from src.data.hf_demo_pool import POOL_FILES  # noqa: E402  single source of truth

try:
    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env")
except ImportError:
    pass

# Repo named for the date the demos were generated, per the project convention.
REPO_NAME = "2026-06-04-stwebagentbench-suitecrm-demos"

ITEMS = list(POOL_FILES.items())  # (local path, path in HF repo)


def jsonl_stats(path):
    n = rew_pos = 0
    rewards = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            n += 1
            rewards.append(r.get("reward", 0.0))
            rew_pos += r.get("reward", 0.0) > 0
    mean = sum(rewards) / max(n, 1)
    return n, rew_pos, mean


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--readme-only", action="store_true",
                    help="re-upload only the card (e.g. after editing it)")
    ap.add_argument("--restore", action="store_true",
                    help="download the pool from HF into data/ instead of uploading")
    args = ap.parse_args()

    token = (os.environ.get("HUGGINGFACE_TOKEN")
             or os.environ.get("HF_TOKEN") or get_token())
    if not token or token == "your-key-here":
        sys.exit("no HF token: set HUGGINGFACE_TOKEN in .env or `hf auth login`")
    api = HfApi(token=token)
    ns = os.environ.get("HF_ORG") or api.whoami()["name"]
    repo_id = f"{ns}/{REPO_NAME}"

    if args.restore:
        from src.data.hf_demo_pool import ensure_local
        got = ensure_local()
        print(f"restored from {repo_id}: {len(got)} file(s) fetched")
        return

    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True,
                         capture_output=True).stdout.strip() or "unknown"

    # HF is canonical and data/ only a cache: the card's counts come from the
    # real data, so pull the demo files back if the cache was cleaned.
    from src.data.hf_demo_pool import ensure_local
    ensure_local(["data/demos/safe.jsonl", "data/demos/unsafe.jsonl"])
    s_n, s_pos, s_mean = jsonl_stats(REPO_ROOT / "data/demos/safe.jsonl")
    u_n, u_pos, u_mean = jsonl_stats(REPO_ROOT / "data/demos/unsafe.jsonl")

    card = f"""---
pretty_name: ST-WebAgentBench SuiteCRM safe/unsafe demo pool (ICRL)
language: en
tags:
  - web-agents
  - safety
  - inverse-constraint-rl
  - st-webagentbench
  - suitecrm
---

# {REPO_NAME}

Standing demo pool for **Adversarial Inverse Constraint RL (ICRL) for LLM
orchestrator safety** on ST-WebAgentBench (SuiteCRM easy tier). Every
experiment run consumes this pool; per-run artifacts (embeddings, constraint
heads, adapters, CuP evals) live in separate `<date>-<run-name>` repos in this
namespace.

| field | value |
| --- | --- |
| experiment | ICRL safe/unsafe demo pool: constraint C_theta is learned from the safe demos only; unsafe demos are used for held-out constraint AUROC evaluation, never for training |
| date_generated | demos 2026-06-04 (webarena_raw 2026-06-06); train/eval splits 2026-07-27 |
| source_repo | github icrl @ `{sha}` |
| models | collector/actor + verifier: `qwen/qwen-2.5-72b-instruct` (OpenRouter) |
| demo_source | live ST-WebAgentBench SuiteCRM episodes (task ids in `tasks/webarena_tasks.json`); safe = policy-compliant traces, unsafe = policy-violating traces |
| generation_config | `configs/demos/collection.yaml` (actor_model + verifier_model `qwen/qwen-2.5-72b-instruct`); splits via `scripts/demos/make_train_eval_splits.py` (`data/splits.json` records the task-level split) |
| schema | jsonl, one trajectory per line: `trajectory_id, task_type, task_instance_id, steps[{{step_idx, action, observation, is_safe}}], is_safe, source, reward, constraint_score` |
| provenance | `python scripts/demos/collect_suitecrm_safe_unsafe_demos.py` then `python scripts/demos/make_train_eval_splits.py` |

## Contents and counts

- `demos/safe.jsonl` — {s_n} safe trajectories ({s_pos} with reward > 0, mean reward {s_mean:.3f})
- `demos/unsafe.jsonl` — {u_n} unsafe trajectories ({u_pos} with reward > 0, mean reward {u_mean:.3f})
- `demos/webarena_raw.jsonl` — raw uncurated collection traces the pool was filtered from
- `splits/` — train / held-out-eval split actually used by the pipeline
- `tasks/webarena_tasks.json` — task definitions the episodes were run against

## Known limitation (read before training on this)

Most safe demos have **reward = 0**: they are policy-compliant but did not
complete the task. ICRL assumes safe demos are *near-optimal*; these satisfy
the safety half of that assumption only. Results derived from this pool must
say so (see repo CLAUDE.md / preflight report).
"""

    if args.dry_run:
        print(f"would create dataset {repo_id} (public) with:")
        for src, dest in ITEMS:
            p = REPO_ROOT / src
            size = f"{p.stat().st_size >> 20} MB" if p.exists() else "not cached locally"
            print(f"  {src} -> {dest}  ({size})")
        print("---- card ----\n" + card)
        return

    api.create_repo(repo_id, repo_type="dataset", private=False, exist_ok=True)
    api.upload_file(path_or_fileobj=card.encode(), path_in_repo="README.md",
                    repo_id=repo_id, repo_type="dataset")
    if args.readme_only:
        print(f"card updated: https://huggingface.co/datasets/{repo_id}")
        return
    for src, dest in ITEMS:
        p = REPO_ROOT / src
        if not p.exists():
            print(f"  !! missing, skipped: {src}")
            continue
        print(f"  -> {repo_id}/{dest}")
        api.upload_file(path_or_fileobj=str(p), path_in_repo=dest,
                        repo_id=repo_id, repo_type="dataset")
    print(f"done: https://huggingface.co/datasets/{repo_id}")


if __name__ == "__main__":
    main()
