#!/usr/bin/env python3
"""
Download every model the run needs, on a node that has internet.

Alliance compute nodes generally cannot reach the internet. The pipeline
otherwise resolves models from HuggingFace at runtime, so a job that queued
successfully dies minutes in with a connection error — after the allocation
started charging.

Run this on the LOGIN node, once per model, then submit with
HF_HUB_OFFLINE=1 (scripts/submit_experiment.sh sets it for you).

Usage:
    # Whatever the cluster profile uses:
    python scripts/prefetch_models.py --profile cluster

    # Explicit:
    python scripts/prefetch_models.py --models Qwen/Qwen2.5-1.5B Qwen/Qwen2.5-7B-Instruct

    # Check without downloading:
    python scripts/prefetch_models.py --profile cluster --check
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def profile_models(profile: str) -> list[str]:
    from scripts.run_experiment import PROFILES
    if profile not in PROFILES:
        raise SystemExit(f"Unknown profile {profile!r}. Known: {sorted(PROFILES)}")
    spec = PROFILES[profile]
    # dict.fromkeys keeps order and drops the duplicate when both are the same.
    return list(dict.fromkeys([spec["encoder_model"], spec["policy_model"]]))


def hf_home() -> str:
    home = os.environ.get("HF_HOME")
    if home:
        return home
    scratch = os.environ.get("SCRATCH")
    if scratch:
        return os.path.join(scratch, "hf_cache")
    return os.path.expanduser("~/.cache/huggingface")


def is_cached(model: str) -> bool:
    """True when the weights resolve with the network switched off."""
    env_before = os.environ.get("HF_HUB_OFFLINE")
    os.environ["HF_HUB_OFFLINE"] = "1"
    try:
        from transformers import AutoConfig
        AutoConfig.from_pretrained(model)
        from huggingface_hub import snapshot_download
        snapshot_download(model, local_files_only=True)
        return True
    except Exception:
        return False
    finally:
        if env_before is None:
            os.environ.pop("HF_HUB_OFFLINE", None)
        else:
            os.environ["HF_HUB_OFFLINE"] = env_before


def fetch(model: str, token: str | None) -> bool:
    from huggingface_hub import snapshot_download
    print(f"  downloading {model} ...", flush=True)
    try:
        # Skip the duplicate formats — the pipeline loads safetensors.
        snapshot_download(
            model,
            token=token,
            ignore_patterns=["*.msgpack", "*.h5", "*.ot", "*.pth", "*consolidated*"],
        )
        return True
    except Exception as e:
        print(f"  FAILED {model}: {type(e).__name__}: {e}", file=sys.stderr)
        return False


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", default=None,
                    help="take the model list from a run_experiment.py profile")
    ap.add_argument("--models", nargs="*", default=[])
    ap.add_argument("--check", action="store_true",
                    help="report cache status without downloading")
    args = ap.parse_args()

    models = list(args.models)
    if args.profile:
        models = profile_models(args.profile) + models
    if not models:
        ap.error("pass --profile and/or --models")

    cache = hf_home()
    os.environ.setdefault("HF_HOME", cache)
    Path(cache).mkdir(parents=True, exist_ok=True)

    token = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")
    print(f"HF_HOME : {cache}")
    print(f"token   : {'set' if token else 'not set (fine for public models)'}")
    print(f"models  : {len(models)}\n")

    missing = []
    for model in models:
        if is_cached(model):
            print(f"  cached  {model}")
            continue
        if args.check:
            print(f"  MISSING {model}")
            missing.append(model)
            continue
        if fetch(model, token) and is_cached(model):
            print(f"  ok      {model}")
        else:
            missing.append(model)

    print()
    if missing:
        verb = "not cached" if args.check else "could not be fetched"
        print(f"{len(missing)} model(s) {verb}:", file=sys.stderr)
        for m in missing:
            print(f"  - {m}", file=sys.stderr)
        if not args.check:
            print("\nIf these are gated repos, set HUGGINGFACE_TOKEN and accept the "
                  "licence on huggingface.co first.", file=sys.stderr)
        return 1

    print("All models are in the cache — jobs can run with HF_HUB_OFFLINE=1.")
    print(f"Export this in your job environment:  HF_HOME={cache}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
