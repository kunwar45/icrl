#!/usr/bin/env python3
# ABOUTME: Downloads every HuggingFace model a run needs into $HF_HOME so offline compute nodes can load them
# ABOUTME: Run on the login node: python scripts/prefetch_models.py --profile cluster [--check] or --models <names>
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


# TensorFlow / Flax / duplicate-format artifacts the pipeline never loads. The
# SAME list must gate the download and the verification: filtering the download
# but demanding a complete mirror when checking marks every model as missing.
IGNORE_PATTERNS = ["*.msgpack", "*.h5", "*.ot", "*.pth", "*consolidated*"]
WEIGHT_SUFFIXES = (".safetensors", ".bin", ".gguf")


def is_cached(model: str) -> bool:
    """
    True when the model resolves without touching the network.

    Checks what a job actually does — config, tokenizer, and at least one
    weights shard on disk — rather than asking for a byte-complete repo mirror.
    `snapshot_download(local_files_only=True)` without the ignore list raises
    IncompleteSnapshotError over files we intentionally skipped.

    Uses the per-call `local_files_only` flag rather than setting
    HF_HUB_OFFLINE: huggingface_hub reads that variable once, at import, into a
    module-level constant, so flipping it here would put the library in offline
    mode for the rest of the process — including the downloads this script
    exists to perform.
    """
    try:
        from transformers import AutoConfig, AutoTokenizer
        AutoConfig.from_pretrained(model, local_files_only=True)
        AutoTokenizer.from_pretrained(model, local_files_only=True)
    except Exception:
        return False

    try:
        from huggingface_hub import snapshot_download
        path = snapshot_download(model, local_files_only=True,
                                 ignore_patterns=IGNORE_PATTERNS)
    except Exception:
        return False

    return any(p.suffix in WEIGHT_SUFFIXES for p in Path(path).rglob("*") if p.is_file())


def fetch(model: str, token: str | None) -> bool:
    from huggingface_hub import snapshot_download
    print(f"  downloading {model} ...", flush=True)
    try:
        snapshot_download(model, token=token, ignore_patterns=IGNORE_PATTERNS)
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

    # This script only ever runs where there IS internet, and it must win over a
    # stale HF_HUB_OFFLINE=1 left in the shell by a previous submit. Set before
    # any huggingface import, since the library snapshots the value on import.
    if os.environ.get("HF_HUB_OFFLINE", "") not in ("", "0"):
        print("note: HF_HUB_OFFLINE was set — clearing it, this step needs the network")
    os.environ["HF_HUB_OFFLINE"] = "0"
    os.environ.pop("TRANSFORMERS_OFFLINE", None)

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
