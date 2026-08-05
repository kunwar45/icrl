"""Canonical demo-pool access — Hugging Face is the source of truth.

The gitignored ``data/`` tree is a local cache, not the canonical store: the
demo pool (safe/unsafe demos, train/eval splits, task definitions) lives in the
HF dataset repo named by ``HF_DEMO_POOL_REPO`` in ``.env`` (default: the
icrl-finetuning pool). Any pool file missing locally is fetched on demand; on
an offline compute node (``HF_HUB_OFFLINE=1``) the fetch fails fast with the
exact login-node command instead.

Fetch the whole pool by hand:  python -m src.data.hf_demo_pool
"""

import os
import shutil
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_POOL_REPO = "icrl-finetuning/2026-06-04-stwebagentbench-suitecrm-demos"

# local path relative to the repo root -> path inside the HF dataset repo
POOL_FILES = {
    "data/demos/safe.jsonl": "demos/safe.jsonl",
    "data/demos/unsafe.jsonl": "demos/unsafe.jsonl",
    "data/demos/webarena_raw.jsonl": "demos/webarena_raw.jsonl",
    "data/train/safe.jsonl": "splits/train/safe.jsonl",
    "data/train/unsafe.jsonl": "splits/train/unsafe.jsonl",
    "data/eval/safe_held_out.jsonl": "splits/eval/safe_held_out.jsonl",
    "data/eval/unsafe_held_out.jsonl": "splits/eval/unsafe_held_out.jsonl",
    "data/splits.json": "splits/splits.json",
    "data/webarena_tasks.json": "tasks/webarena_tasks.json",
}


def pool_repo() -> str:
    return os.environ.get("HF_DEMO_POOL_REPO", DEFAULT_POOL_REPO)


def _pool_key(path) -> str | None:
    """POOL_FILES key for *path*, or None if the file is not part of the pool."""
    p = Path(path)
    if not p.is_absolute():
        p = REPO_ROOT / p
    try:
        rel = p.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return None
    return rel if rel in POOL_FILES else None


def ensure_local(paths=None, verbose: bool = True) -> list[Path]:
    """Make the given pool files (default: all of them) exist locally.

    Missing files are downloaded from the pool's HF repo; present files are
    left untouched. Returns the list of freshly fetched paths. Paths that are
    not pool files are ignored — callers can pass anything path-shaped.
    """
    keys = ([k for k in (_pool_key(p) for p in paths) if k]
            if paths is not None else list(POOL_FILES))
    missing = [k for k in keys if not (REPO_ROOT / k).exists()]
    if not missing:
        return []

    if os.environ.get("HF_HUB_OFFLINE") == "1":
        raise RuntimeError(
            f"demo pool files missing locally ({', '.join(missing)}) and this "
            f"node is offline (HF_HUB_OFFLINE=1). From the login node run: "
            f"python -m src.data.hf_demo_pool"
        )

    try:
        from dotenv import load_dotenv
        load_dotenv(REPO_ROOT / ".env")
    except ImportError:
        pass
    from huggingface_hub import get_token, hf_hub_download
    token = (os.environ.get("HUGGINGFACE_TOKEN")
             or os.environ.get("HF_TOKEN") or get_token())
    if token == "your-key-here":
        token = None  # the pool repo is public; a placeholder token would 401

    repo = pool_repo()
    fetched = []
    # local_dir keeps downloads out of the shared HF cache so the only copy
    # ends up under data/ (HF stays the canonical store, data/ the cache).
    with tempfile.TemporaryDirectory() as tmp:
        for key in missing:
            dest = REPO_ROOT / key
            dest.parent.mkdir(parents=True, exist_ok=True)
            got = hf_hub_download(repo, POOL_FILES[key], repo_type="dataset",
                                  token=token, local_dir=tmp)
            shutil.move(got, dest)
            fetched.append(dest)
            if verbose:
                print(f"  fetched from HF pool {repo}: {key}")
    return fetched


if __name__ == "__main__":
    got = ensure_local()
    print(f"pool complete: {len(got)} file(s) fetched, "
          f"{len(POOL_FILES) - len(got)} already present")
