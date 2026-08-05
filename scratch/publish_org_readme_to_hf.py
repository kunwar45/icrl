"""One-off: create/update the icrl-finetuning org card on Hugging Face.

Org cards live in a Space named `<org>/README`; its README.md is what renders
on https://huggingface.co/<org>.

Run from the repo root:  python scratch/publish_org_readme_to_hf.py
"""

import os
import sys
from pathlib import Path

from huggingface_hub import HfApi, get_token

REPO_ROOT = Path(__file__).resolve().parents[1]

try:
    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env")
except ImportError:
    pass

ORG = os.environ.get("HF_ORG", "icrl-finetuning")

CARD = """# Adversarial Inverse Constraint RL for LLM orchestrator safety

Artifacts from **adversarial ICRL on
[ST-WebAgentBench](https://github.com/segev-shlomov/ST-WebAgentBench)**: a
constraint function C&theta; is learned from safe demonstrations only (if safe
demos are near-optimal, any trajectory that beats their reward must have
skipped a required safety step), then drives a Lagrangian-constrained LoRA
fine-tune of the policy. Headline metric: **CuP (Completion under Policy)** on
held-out tasks, baseline vs tuned.

## What lives where

- **`<YYYY-MM-DD>-stwebagentbench-suitecrm-demos`** (dataset) — the standing
  safe/unsafe demo pool + train/eval splits every run consumes. Demo quality
  *is* the experiment: read the card's known-limitation note before training.
- **`<YYYY-MM-DD>-<run-name>`** (dataset, one per experiment run) — that run's
  embeddings, constraint head + metrics, CuP eval results, plots, and report.
- **`<YYYY-MM-DD>-<run-name>-policy-lora`** (model, one per run) — the
  Lagrangian-fine-tuned LoRA adapter.

Dates are when the artifacts were *generated*, not uploaded. Every repo card
records source commit, models, demo source, generation config, schema, and the
exact rerun command. Code and configs live in the `icrl` git repository; this
org is the canonical home for the data.

Mock-env (smoke-test) runs are never published here — they are test fixtures,
not results.
"""


def main():
    token = (os.environ.get("HUGGINGFACE_TOKEN")
             or os.environ.get("HF_TOKEN") or get_token())
    if not token or token == "your-key-here":
        sys.exit("no HF token: set HUGGINGFACE_TOKEN in .env or `hf auth login`")
    api = HfApi(token=token)
    rid = f"{ORG}/README"
    api.create_repo(rid, repo_type="space", space_sdk="static",
                    private=False, exist_ok=True)
    api.upload_file(path_or_fileobj=CARD.encode(), path_in_repo="README.md",
                    repo_id=rid, repo_type="space")
    print(f"org card updated: https://huggingface.co/{ORG}")


if __name__ == "__main__":
    main()
