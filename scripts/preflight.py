#!/usr/bin/env python3
"""
Check everything the pipeline needs before a long job burns an allocation.

Every failure below has cost a real run: a missing SuiteCRM URL that only
surfaced after vLLM finished loading, `answer()` undefined because the env was
built without an action set, 0 tasks registered because the .pth file never
landed in site-packages.

Usage:
    python scripts/preflight.py                        # mock backend (no browser)
    python scripts/preflight.py --backend stwebagent \
        --task-ids 235 236 --policy-model Qwen/Qwen2.5-7B-Instruct

Exit codes:
    0  ready
    1  at least one hard requirement is missing
Warnings (degraded but runnable) never fail the run.
"""
from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

OK, WARN, FAIL = "  ok  ", " warn ", " FAIL "
_results: list[tuple[str, str, str]] = []


def record(status: str, name: str, detail: str = "") -> bool:
    _results.append((status, name, detail))
    print(f"[{status}] {name}" + (f" — {detail}" if detail else ""), flush=True)
    return status != FAIL


# ── Checks ────────────────────────────────────────────────────────────────────

def check_python_packages() -> None:
    required = ["torch", "transformers", "hydra", "omegaconf", "accelerate",
                "sklearn", "numpy"]
    optional = {"peft": "LoRA fine-tuning", "wandb": "experiment tracking",
                "matplotlib": "the plots stage"}

    for mod in required:
        try:
            m = importlib.import_module(mod)
            record(OK, f"import {mod}", getattr(m, "__version__", ""))
        except ImportError as e:
            record(FAIL, f"import {mod}", f"{e} — check the venv is activated")

    for mod, why in optional.items():
        try:
            importlib.import_module(mod)
            record(OK, f"import {mod}", why)
        except ImportError:
            record(WARN, f"import {mod}", f"missing — {why} unavailable")


def check_repo_importable() -> None:
    try:
        import src.finetune.rollout  # noqa: F401
        record(OK, "import src.*", "repo root is on sys.path")
    except ImportError as e:
        record(FAIL, "import src.*", f"{e} — add the repo ROOT (not src/) to PYTHONPATH")


def check_gpu(require: bool) -> None:
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        record(OK, "CUDA", f"{torch.cuda.device_count()}x {names[0]}")
    elif torch.backends.mps.is_available():
        record(WARN if require else OK, "CUDA", "none — using Apple MPS")
    else:
        record(FAIL if require else WARN, "CUDA",
               "no GPU visible — check --gres and that the job is on a compute node")


def check_hf_cache() -> None:
    hf_home = os.environ.get("HF_HOME")
    if not hf_home:
        record(WARN, "HF_HOME",
               "unset — weights go to ~/.cache (often slow NFS on the cluster); "
               "set HF_HOME=$SCRATCH/hf_cache")
        return
    path = Path(hf_home)
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".preflight_write_test"
        probe.write_text("ok")
        probe.unlink()
        record(OK, "HF_HOME", f"{hf_home} (writable)")
    except OSError as e:
        record(FAIL, "HF_HOME", f"{hf_home} is not writable: {e}")


def check_model_resolvable(label: str, model_name: str) -> None:
    """
    Resolve the model — cheap, and catches typos, gated repos and the big one:
    weights that are not in the local cache on a compute node with no internet.
    """
    if not model_name:
        return
    if Path(model_name).exists():
        record(OK, f"model {label}", f"{model_name} (local path)")
        return

    offline = os.environ.get("HF_HUB_OFFLINE", "") not in ("", "0")
    try:
        from transformers import AutoConfig
        AutoConfig.from_pretrained(model_name)
        record(OK, f"model {label}",
               f"{model_name}{' (from cache, offline)' if offline else ''}")
        return
    except Exception as e:
        msg = str(e).split("\n")[0][:120]

    lowered = msg.lower()
    if offline or "offline" in lowered or "couldn't connect" in lowered \
            or "connection" in lowered:
        record(FAIL, f"model {label}",
               f"{model_name} is not in the local cache ({os.environ.get('HF_HOME', '~/.cache')}). "
               f"Compute nodes have no internet — prefetch it on the LOGIN node: "
               f"python scripts/infra/prefetch_models.py --models {model_name}")
        return
    gated = "gated" in lowered or "401" in msg or "authorized" in lowered
    record(FAIL, f"model {label}", f"{model_name}: {msg}"
           + (" — set HUGGINGFACE_TOKEN" if gated else ""))


def check_demos(paths: list[str]) -> None:
    """Accept both a .jsonl and a directory of collection traces."""
    for p in paths:
        path = Path(p)
        if not path.exists():
            record(FAIL, f"demos {p}",
                   "not found — run the collection job, or pass --safe-demos / "
                   "--unsafe-demos")
            continue
        try:
            from src.data.trace_loader import load_demos
            trajectories = load_demos(path)
        except Exception as e:
            record(FAIL, f"demos {p}", f"{type(e).__name__}: {e}")
            continue

        n = len(trajectories)
        n_tasks = len({t.task_instance_id for t in trajectories})
        n_term = sum(1 for t in trajectories if t.terminated)
        if n == 0:
            record(FAIL, f"demos {p}", "no trajectories")
        elif n_tasks < 2:
            record(FAIL, f"demos {p}",
                   f"{n} trajectories across only {n_tasks} task(s) — a held-out "
                   f"split needs at least 2, so the AUROC gate cannot run")
        else:
            status = OK if n_term else WARN
            record(status, f"demos {p}",
                   f"{n} trajectories, {n_tasks} tasks, {n_term} terminated cleanly"
                   + ("" if n_term else " — none are near-optimal; ICRL expects "
                                        "experts that finish"))


def check_benchmark(task_ids: list[str]) -> None:
    try:
        import gymnasium as gym
        import browsergym.stwebagentbench  # noqa: F401
    except ImportError as e:
        record(FAIL, "browsergym.stwebagentbench",
               f"{e} — install the fork and check STWEBAGENT_ROOT is on PYTHONPATH")
        return

    registered = {e.split(".")[-1] for e in gym.envs.registry if "STWebAgent" in e}
    if not registered:
        record(FAIL, "task registration",
               "0 tasks — re-run scripts/infra/setup_cluster.sh; check stwebagentbench.pth")
        return
    record(OK, "task registration", f"{len(registered)} tasks")

    missing = [t for t in task_ids if str(t) not in registered]
    if missing:
        record(FAIL, "requested task ids", f"not registered: {missing}")
    elif task_ids:
        record(OK, "requested task ids", f"all {len(task_ids)} present")

    # The agent finishes a task by calling answer(). BrowserGym renders custom
    # actions with inspect.getsource(), so one defined inside a factory function
    # is emitted indented, never gets defined at module level, and raises
    # NameError the moment the agent tries to finish — after a full episode of
    # browser work. Compile the generated code and execute the definitions.
    try:
        from src.data.st_webagent import build_action_set
        code = build_action_set().to_python_code('answer("done")')
        if not code:
            record(FAIL, "action set", "answer() produced no code")
        else:
            compile(code, "<action>", "exec")
            defs = [ln for ln in code.splitlines() if ln.lstrip().startswith("def answer")]
            if not defs:
                record(FAIL, "action set", "generated code never defines answer()")
            elif defs[0].startswith((" ", "\t")):
                record(FAIL, "action set",
                       "def answer() is indented in the generated code — it must be "
                       "defined at module level (see src/data/st_webagent.py)")
            else:
                record(OK, "action set", "answer() compiles at module level")
    except SyntaxError as e:
        record(FAIL, "action set", f"generated action code does not compile: {e}")
    except Exception as e:
        record(FAIL, "action set", f"{type(e).__name__}: {e}")


def check_benchmark_env_vars() -> None:
    """
    ST-WebAgentBench validates every site URL at import, not just the ones the
    selected tasks use, and refuses to start when any is missing.
    """
    missing = [v for v in ("GITLAB", "SHOPPING_ADMIN") if not os.environ.get(v)]
    if missing:
        record(FAIL, "benchmark env vars",
               f"unset: {', '.join(missing)} — the benchmark requires all site URLs "
               f"even for SuiteCRM-only tasks. Any reachable placeholder works; set "
               f"them in .env and $STWEBAGENT_ROOT/.env")
    else:
        record(OK, "benchmark env vars", "GITLAB, SHOPPING_ADMIN set")


def check_playwright() -> None:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as e:
        record(FAIL, "playwright", f"{e} — pip install playwright")
        return
    try:
        with sync_playwright() as p:
            path = p.chromium.executable_path
        if path and Path(path).exists():
            record(OK, "playwright chromium", path)
        else:
            record(FAIL, "playwright chromium",
                   "browser not installed — run: playwright install chromium")
    except Exception as e:
        record(FAIL, "playwright chromium", f"{type(e).__name__}: {e} — "
               f"run: playwright install chromium")


def check_suitecrm() -> None:
    url = os.environ.get("WA_SUITECRM") or os.environ.get("SUITECRM")
    if not url:
        record(FAIL, "WA_SUITECRM",
               "unset — start SuiteCRM (scripts/infra/start_suitecrm_apptainer.sh) and "
               "put WA_SUITECRM=http://<login-node>:8080/public in .env")
        return
    import urllib.error
    import urllib.request
    try:
        with urllib.request.urlopen(url, timeout=20) as resp:
            code = resp.getcode()
        record(OK if code < 400 else FAIL, "SuiteCRM reachable", f"{url} → HTTP {code}")
    except urllib.error.HTTPError as e:
        # A 3xx/4xx still proves something is listening.
        record(OK if e.code < 500 else FAIL, "SuiteCRM reachable",
               f"{url} → HTTP {e.code}")
    except Exception as e:
        record(FAIL, "SuiteCRM reachable",
               f"{url} unreachable ({type(e).__name__}) — is it running, and is the "
               f"hostname resolvable from this node? (localhost will not work)")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=["mock", "stwebagent"], default="mock")
    ap.add_argument("--task-ids", nargs="*", default=[])
    ap.add_argument("--encoder-model", default=None)
    ap.add_argument("--policy-model", default=None)
    ap.add_argument("--demos", nargs="*", default=[])
    ap.add_argument("--require-gpu", action="store_true")
    ap.add_argument("--skip-models", action="store_true",
                    help="skip HuggingFace config lookups (offline nodes)")
    ap.add_argument("--skip-browser", action="store_true",
                    help="skip Playwright and SuiteCRM checks — for runs whose "
                         "stages never open a browser (constraint training, gate)")
    args = ap.parse_args()

    print(f"\n=== preflight (backend={args.backend}) ===\n")

    print("[ environment ]")
    check_python_packages()
    check_repo_importable()
    check_gpu(require=args.require_gpu)
    check_hf_cache()

    if not args.skip_models:
        print("\n[ models ]")
        check_model_resolvable("encoder", args.encoder_model)
        check_model_resolvable("policy", args.policy_model)

    if args.demos:
        print("\n[ demos ]")
        check_demos(args.demos)

    if args.backend == "stwebagent":
        print("\n[ benchmark ]")
        if args.skip_browser:
            print("  (browser + CRM checks skipped — no rollout stage selected)")
        else:
            check_benchmark_env_vars()
            check_benchmark([str(t) for t in args.task_ids])
            check_playwright()
            check_suitecrm()

    n_fail = sum(1 for s, _, _ in _results if s == FAIL)
    n_warn = sum(1 for s, _, _ in _results if s == WARN)
    print(f"\n{'=' * 60}")
    print(f"{len(_results)} checks — {n_fail} failed, {n_warn} warnings")
    if n_fail:
        print("\nBlocking:")
        for status, name, detail in _results:
            if status == FAIL:
                print(f"  - {name}: {detail}")
        print()
        return 1
    print("Ready.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
