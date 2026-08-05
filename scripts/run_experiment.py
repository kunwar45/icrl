#!/usr/bin/env python3
"""
End-to-end ICRL experiment driver.

Runs every stage in order, each as its own subprocess so the stage scripts stay
independently runnable and a failure points at one command you can re-run by
hand:

    0. preflight       scripts/preflight.py          environment is actually usable
    1. splits          scripts/demos/make_train_eval_splits.py        train / held-out demos
    2. encode          scripts/constraint/encode_trajectories.py cached backbone embeddings
    3. constraint      scripts/constraint/train_constraint.py   train C_theta
    4. gate            scripts/constraint/eval_constraint.py    held-out AUROC gate
    5. eval_base       scripts/finetune/eval_finetune.py      CuP of the untuned policy
    6. finetune        scripts/finetune/run_finetune.py       Lagrangian constrained PG
    7. eval_tuned      scripts/finetune/eval_finetune.py      CuP of the tuned policy
    8. plots           scripts/make_experiment_plots.py         figures + one HTML report

Profiles
--------
    smoke    tiny random model, mock env, a handful of steps. Minutes on a
             laptop. Proves the plumbing, says nothing about results.
    local    Qwen2.5-0.5B on CPU/MPS against the mock env. Hours on a laptop.
    cluster  the real configuration: Qwen backbone + real ST-WebAgentBench.

Every profile trains on one set of task ids and evaluates CuP on a disjoint
held-out set, so the headline number never comes from a task the policy has
already seen.

Usage
-----
    python scripts/run_experiment.py --profile smoke
    python scripts/run_experiment.py --profile cluster --run-name icrl_v1
    python scripts/run_experiment.py --profile smoke --stages constraint,gate
    python scripts/run_experiment.py --profile smoke --dry-run

    # WebArena human traces as the expert set instead of the weak safe demos:
    python scripts/run_experiment.py --profile cluster \
        --safe-demos data/demos/webarena_raw.jsonl

The AUROC gate is reported but not enforced by default — the pipeline is worth
verifying even while the safe demos are still poor. Pass --strict-gate to stop
the run when the gate fails.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

STAGES = ["preflight", "splits", "encode", "constraint", "gate",
          "eval_base", "finetune", "eval_tuned", "plots"]

# ── Profiles ──────────────────────────────────────────────────────────────────
# Each profile supplies Hydra overrides shared by every stage, plus a few
# driver-level knobs.

PROFILES: dict[str, dict] = {
    "smoke": {
        "description": "tiny random weights + mock env — verifies plumbing only",
        "encoder_model": "hf-internal-testing/tiny-random-gpt2",
        "policy_model": "hf-internal-testing/tiny-random-gpt2",
        "env_backend": "mock",
        "compute": "local",
        "train_task_ids": ["m001", "m002", "m003", "m005"],
        "eval_task_ids": ["m004", "m006"],
        "lora": False,
        "encode": True,
        "overrides": [
            "constraint.encoder.max_length=256",
            "constraint.training.n_iterations=20",
            "constraint.training.n_constraint_steps=20",
            "constraint.training.batch_size=8",
            "constraint.training.eval_every=10",
            "finetune.ppo.steps=3",
            "finetune.ppo.batch_size=2",
            "finetune.ppo.max_rollout_steps=6",
            "finetune.ppo.max_obs_tokens=256",
            "finetune.ppo.max_act_tokens=24",
            "finetune.ppo.learning_rate=1e-4",
            "finetune.env.max_steps=6",
            "finetune.checkpointing.save_every_n_steps=0",
            "finetune.eval.n_episodes_per_task=1",
        ],
    },
    "local": {
        "description": "Qwen2.5-0.5B on CPU/MPS + mock env — real weights, small",
        "encoder_model": "Qwen/Qwen2.5-0.5B",
        "policy_model": "Qwen/Qwen2.5-0.5B-Instruct",
        "env_backend": "mock",
        "compute": "local",
        "train_task_ids": ["m001", "m002", "m003", "m005"],
        "eval_task_ids": ["m004", "m006"],
        "lora": True,
        "encode": True,
        "overrides": [
            "constraint.encoder.max_length=1024",
            "constraint.training.n_iterations=50",
            "constraint.training.n_constraint_steps=30",
            "finetune.ppo.steps=20",
            "finetune.ppo.batch_size=4",
            "finetune.ppo.max_rollout_steps=10",
            "finetune.ppo.max_obs_tokens=768",
            "finetune.env.max_steps=10",
            "finetune.checkpointing.save_every_n_steps=0",
            "finetune.eval.n_episodes_per_task=2",
        ],
    },
    "cluster": {
        "description": "full configuration — real benchmark, GPU",
        "encoder_model": "Qwen/Qwen2.5-1.5B",
        "policy_model": "Qwen/Qwen2.5-7B-Instruct",
        "env_backend": "stwebagent",
        # Any $SCRATCH-based compute group works; killarney and carleton are
        # identical in shape. Override with --compute.
        "compute": "killarney",
        # Easy-tier SuiteCRM tasks (the IDs collect_safe_trajectories_job.sh collects), split so
        # the CuP number comes from tasks the policy never trained on.
        "train_task_ids": [str(i) for i in range(235, 250)],
        "eval_task_ids": [str(i) for i in range(250, 255)],
        "lora": True,
        "encode": True,
        "overrides": [
            "constraint.encoder.max_length=2048",
            "constraint.training.n_iterations=100",
            "finetune.ppo.steps=200",
            "finetune.ppo.batch_size=8",
            "finetune.checkpointing.save_every_n_steps=50",
            "finetune.eval.n_episodes_per_task=3",
        ],
    },
}

# ── Stage runner ──────────────────────────────────────────────────────────────

class StageFailure(RuntimeError):
    pass


def run_cmd(cmd: list[str], *, dry_run: bool, allow_fail: bool = False) -> tuple[int, float]:
    printable = " ".join(shlex.quote(c) for c in cmd)
    print(f"\n$ {printable}\n", flush=True)
    if dry_run:
        return 0, 0.0

    start = time.time()
    proc = subprocess.run(cmd, cwd=REPO_ROOT, env=_child_env())
    elapsed = time.time() - start
    if proc.returncode != 0 and not allow_fail:
        raise StageFailure(f"exit {proc.returncode}: {printable}")
    return proc.returncode, elapsed


def load_dotenv(path: Path) -> dict:
    """
    Read .env into a dict without adding a dependency.

    Only the collection scripts call python-dotenv, so WA_SUITECRM / GITLAB /
    SHOPPING_ADMIN never reached the fine-tuning and evaluation stages — they
    roll out real browser episodes too and fail without them. Existing
    environment variables win, so `WA_SUITECRM=... python ...` still overrides.
    """
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            values[key] = value
    return values


def _child_env() -> dict:
    env = os.environ.copy()

    for key, value in load_dotenv(REPO_ROOT / ".env").items():
        env.setdefault(key, value)

    existing = env.get("PYTHONPATH", "")
    parts = [str(REPO_ROOT)]
    stweb_root = env.get("STWEBAGENT_ROOT")
    if stweb_root:
        # `stwebagentbench` (the evaluators) lives at the fork's root, separately
        # from the pip-installed `browsergym.stwebagentbench` shim.
        parts.append(stweb_root)
    if existing:
        parts.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(parts)
    # Hydra writes its own run dirs otherwise; keep artifacts under the repo.
    env.setdefault("HYDRA_FULL_ERROR", "1")
    return env


def python_bin() -> str:
    return sys.executable


# ── Override assembly ─────────────────────────────────────────────────────────

def common_overrides(profile: dict, args) -> list[str]:
    ov = [
        "+constraint=icrl_default",
        "+finetune=lagrangian_ppo",
        f"+compute={args.compute}",
        f"run_name={args.run_name}",
        f"constraint.encoder.model_name={profile['encoder_model']}",
        f"finetune.policy.model_name={profile['policy_model']}",
        f"finetune.env.backend={profile['env_backend']}",
        f"finetune.policy.lora.enabled={str(profile['lora']).lower()}",
        f"seed={args.seed}",
    ]
    # Only override paths when the user asked for it — otherwise the compute
    # group's paths win (on the cluster those point at $SCRATCH, not the repo).
    for key, value in args.explicit_paths.items():
        if value is not None:
            ov.append(f"paths.{key}={value}")

    ov += profile["overrides"]
    ov.append("finetune.env.task_ids=[" + ",".join(profile["train_task_ids"]) + "]")

    if args.wandb:
        ov.append("wandb.enabled=true")
        if args.wandb_entity:
            ov.append(f"wandb.entity={args.wandb_entity}")

    ov += args.override or []
    return ov


def hydra_cmd(script: str, overrides: list[str]) -> list[str]:
    return [python_bin(), f"scripts/{script}"] + overrides


# ── Stages ────────────────────────────────────────────────────────────────────

# Stages that actually drive a browser; only these need Playwright + a live CRM.
ROLLOUT_STAGES = {"eval_base", "finetune", "eval_tuned"}


def stage_preflight(args, profile, results):
    """Fail in seconds on a broken environment rather than hours into the run."""
    # Checking for SuiteCRM when the run is only training C_theta would block a
    # perfectly valid CPU/GPU-only job on a browser it never opens.
    needs_browser = bool(ROLLOUT_STAGES & set(args.selected_stages))

    cmd = [python_bin(), "scripts/preflight.py",
           "--backend", profile["env_backend"],
           "--encoder-model", profile["encoder_model"],
           "--policy-model", profile["policy_model"],
           "--demos", args.safe_demos, args.unsafe_demos,
           "--task-ids", *profile["train_task_ids"], *profile["eval_task_ids"]]
    if not needs_browser:
        cmd.append("--skip-browser")
    if profile["env_backend"] == "stwebagent":
        cmd.append("--require-gpu")
    code, elapsed = run_cmd(cmd, dry_run=args.dry_run, allow_fail=True)
    results["preflight"] = {"status": "ok" if code == 0 else "failed",
                            "seconds": elapsed}
    if code != 0:
        raise StageFailure(
            "preflight found blocking problems (above). Fix them, or skip this "
            "stage with --stages splits,encode,constraint,gate,eval_base,"
            "finetune,eval_tuned"
        )


def stage_splits(args, profile, results):
    cmd = [python_bin(), "scripts/demos/make_train_eval_splits.py",
           "--safe", args.safe_demos, "--unsafe", args.unsafe_demos,
           "--train-dir", os.path.join(args.data_root, "train"),
           "--eval-dir", os.path.join(args.data_root, "eval"),
           "--manifest", os.path.join(args.data_root, "splits.json"),
           "--seed", str(args.seed)]
    if args.demo_limit:
        cmd += ["--limit", str(args.demo_limit)]
    _, elapsed = run_cmd(cmd, dry_run=args.dry_run)
    results["splits"] = {"status": "ok", "seconds": elapsed}


def stage_encode(args, profile, results):
    """Cache backbone embeddings so constraint training skips the backbone."""
    if not profile.get("encode"):
        results["encode"] = {"status": "skipped"}
        return
    emb_dir = Path(args.embeddings_dir)
    total = 0.0
    for label, jsonl in (("safe", os.path.join(args.data_root, "train", "safe.jsonl")),
                         ("unsafe", os.path.join(args.data_root, "train", "unsafe.jsonl"))):
        out = emb_dir / f"{label}.pt"
        cmd = [python_bin(), "scripts/constraint/encode_trajectories.py",
               "--jsonl", jsonl, "--label", label, "--output", str(out),
               "--model", profile["encoder_model"],
               "--max-length", _override_value(profile, "constraint.encoder.max_length", "2048"),
               "--batch-size", str(args.encode_batch_size)]
        _, elapsed = run_cmd(cmd, dry_run=args.dry_run)
        total += elapsed
    results["encode"] = {"status": "ok", "seconds": total,
                         "safe": str(emb_dir / "safe.pt"),
                         "unsafe": str(emb_dir / "unsafe.pt")}


def stage_constraint(args, profile, results):
    ov = common_overrides(profile, args)
    if profile.get("encode"):
        emb_dir = Path(args.embeddings_dir)
        ov += [
            f"constraint.encoder.safe_embeddings_path={emb_dir / 'safe.pt'}",
            f"constraint.encoder.unsafe_embeddings_path={emb_dir / 'unsafe.pt'}",
        ]
    _, elapsed = run_cmd(hydra_cmd("constraint/train_constraint.py", ov), dry_run=args.dry_run)
    results["constraint"] = {"status": "ok", "seconds": elapsed}


def stage_gate(args, profile, results):
    ov = common_overrides(profile, args)
    code, elapsed = run_cmd(hydra_cmd("constraint/eval_constraint.py", ov),
                            dry_run=args.dry_run, allow_fail=not args.strict_gate)
    metrics = _read_json(Path(args.checkpoint_dir) / args.run_name / "held_out_metrics.json")
    passed = bool(metrics.get("passed")) if metrics else code == 0
    results["gate"] = {"status": "ok" if code == 0 else "gate_failed",
                       "seconds": elapsed, "passed": passed,
                       "auroc": metrics.get("auroc") if metrics else None}
    if code != 0:
        print("\n!! Constraint gate FAILED. Continuing anyway (pass --strict-gate "
              "to stop here). Downstream numbers are pipeline checks, not results.\n")


def _eval_stage(args, profile, results, key: str, run_suffix: str, policy_path: str | None):
    ov = common_overrides(profile, args)
    ov = [o for o in ov if not o.startswith("run_name=")]
    ov.append(f"run_name={args.run_name}_{run_suffix}")
    # Evaluate on the held-out tasks: drop the training restriction so the env
    # exposes them, then point the evaluator at exactly that set.
    ov = [o for o in ov if not o.startswith("finetune.env.task_ids=")]
    ov.append("finetune.eval.task_ids=[" + ",".join(profile["eval_task_ids"]) + "]")
    if policy_path:
        ov.append(f"finetune.eval.policy_path={policy_path}")
    _, elapsed = run_cmd(hydra_cmd("finetune/eval_finetune.py", ov), dry_run=args.dry_run)

    summary = _read_json(
        Path(args.checkpoint_dir) / f"{args.run_name}_{run_suffix}" / "cup_eval.json"
    ).get("summary", {})
    results[key] = {"status": "ok", "seconds": elapsed, **_cup_fields(summary)}


def stage_eval_base(args, profile, results):
    _eval_stage(args, profile, results, "eval_base", "eval_base", None)


def stage_finetune(args, profile, results):
    ov = common_overrides(profile, args)
    _, elapsed = run_cmd(hydra_cmd("finetune/run_finetune.py", ov), dry_run=args.dry_run)
    results["finetune"] = {"status": "ok", "seconds": elapsed}


def stage_eval_tuned(args, profile, results):
    ckpt = os.path.join(args.checkpoint_dir, args.run_name, "final")
    if not args.dry_run and not os.path.exists(ckpt):
        raise StageFailure(f"No fine-tuned checkpoint at {ckpt}")
    _eval_stage(args, profile, results, "eval_tuned", "eval_tuned", ckpt)


def stage_plots(args, profile, results):
    """
    Figures + a single self-contained HTML report.

    Never fails the run: a job that got as far as producing numbers should not
    be marked failed because a chart could not be drawn, and make_experiment_plots.py
    already skips whatever the run did not produce.
    """
    # Flush stage timings first — the timings figure reads this file.
    if not args.dry_run:
        write_run_report(results, args)

    out_dir = os.path.join(args.log_dir, args.run_name, "plots")
    cmd = [python_bin(), "scripts/make_experiment_plots.py",
           "--run-name", args.run_name,
           "--log-dir", args.log_dir,
           "--checkpoint-dir", args.checkpoint_dir,
           "--out-dir", out_dir,
           "--theme", args.plot_theme,
           "--allow-empty"]
    if args.pdf:
        cmd.append("--pdf")
    code, elapsed = run_cmd(cmd, dry_run=args.dry_run, allow_fail=True)
    results["plots"] = {"status": "ok" if code == 0 else "failed",
                        "seconds": elapsed, "out_dir": out_dir}


STAGE_FNS = {
    "preflight": stage_preflight,
    "splits": stage_splits,
    "encode": stage_encode,
    "constraint": stage_constraint,
    "gate": stage_gate,
    "eval_base": stage_eval_base,
    "finetune": stage_finetune,
    "eval_tuned": stage_eval_tuned,
    "plots": stage_plots,
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _override_value(profile: dict, key: str, default: str) -> str:
    for o in profile["overrides"]:
        if o.startswith(f"{key}="):
            return o.split("=", 1)[1]
    return default


def resolve_effective_paths(args) -> None:
    """
    Fill in the paths the driver itself needs to read artifacts back.

    Anything the user passed explicitly is kept (and forwarded as a Hydra
    override); anything left as None is taken from the selected compute group,
    so the driver looks for artifacts exactly where the stages wrote them.
    """
    from omegaconf import OmegaConf

    args.explicit_paths = {
        "data_root": args.data_root,
        "checkpoint_dir": args.checkpoint_dir,
        "log_dir": args.log_dir,
    }

    defaults = {"data_root": "data", "checkpoint_dir": "checkpoints", "log_dir": "logs"}
    compute_file = REPO_ROOT / "configs" / "compute" / f"{args.compute}.yaml"
    if compute_file.exists():
        try:
            group = OmegaConf.load(compute_file)
            paths = OmegaConf.to_container(group.get("paths", {}), resolve=True) or {}
            defaults.update({k: v for k, v in paths.items() if k in defaults and v})
        except Exception as e:
            print(f"warning: could not read {compute_file} ({e}); "
                  f"falling back to repo-relative paths")

    for key, fallback in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, os.path.expandvars(os.path.expanduser(str(fallback))))


def resolve_demo_paths(args) -> None:
    """
    Point --safe-demos / --unsafe-demos at whatever actually exists.

    The SLURM collection job writes task_*_trace_*.json to
    $SCRATCH/trajectories/{safe,unsafe}, while the repo default is a .jsonl under
    data/demos. Rather than fail on a fresh cluster checkout, fall back to the
    collection output — and say so, because which demos were used changes the
    result.
    """
    scratch = os.environ.get("SCRATCH", f"/scratch/{os.environ.get('USER', '')}")
    for attr, default, fallback in (
        ("safe_demos", "data/demos/safe.jsonl",
         os.path.join(scratch, "trajectories", "safe")),
        ("unsafe_demos", "data/demos/unsafe.jsonl",
         os.path.join(scratch, "trajectories", "unsafe")),
    ):
        chosen = getattr(args, attr)
        if chosen is not None:
            continue
        if Path(default).exists():
            setattr(args, attr, default)
        elif Path(fallback).exists():
            setattr(args, attr, fallback)
            print(f"note: {default} not found — using collection output {fallback}")
        else:
            setattr(args, attr, default)  # let preflight report it


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _cup_fields(summary: dict) -> dict:
    return {k: summary.get(k) for k in
            ("cup", "completion_rate", "violation_rate", "mean_steps", "n_episodes")}


def _fmt(v) -> str:
    if v is None:
        return "—"
    return f"{v:.3f}" if isinstance(v, float) else str(v)


def print_summary(results: dict, args) -> None:
    print("\n" + "=" * 72)
    print(f"EXPERIMENT SUMMARY — run_name={args.run_name}  profile={args.profile}")
    print("=" * 72)
    for stage in STAGES:
        r = results.get(stage)
        if r is None:
            print(f"  {stage:12s} not run")
            continue
        secs = r.get("seconds")
        extra = ""
        if stage == "gate":
            extra = f"  AUROC={_fmt(r.get('auroc'))} passed={r.get('passed')}"
        elif stage in ("eval_base", "eval_tuned"):
            extra = (f"  CuP={_fmt(r.get('cup'))} "
                     f"completion={_fmt(r.get('completion_rate'))} "
                     f"violations={_fmt(r.get('violation_rate'))}")
        print(f"  {stage:12s} {r['status']:12s} "
              f"{f'{secs:6.1f}s' if secs else '':>8s}{extra}")

    base, tuned = results.get("eval_base"), results.get("eval_tuned")
    if base and tuned and base.get("cup") is not None and tuned.get("cup") is not None:
        delta = tuned["cup"] - base["cup"]
        print("-" * 72)
        print(f"  CuP  baseline {base['cup']:.3f}  →  tuned {tuned['cup']:.3f}  "
              f"({delta:+.3f})")
    print("=" * 72)

    print(f"Report: {write_run_report(results, args)}\n")


def write_run_report(results: dict, args) -> Path:
    """
    Persist the per-stage report.

    Called before the plots stage as well as at the end, because the stage
    timings figure reads this file — writing it only on exit means the plot is
    always one run behind.
    """
    out = Path(args.log_dir) / f"{args.run_name}_experiment.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "run_name": args.run_name,
        "profile": args.profile,
        "epsilon": _profile_epsilon(args),
        "stages": results,
    }, indent=2))
    return out


def _profile_epsilon(args) -> float:
    """The constraint budget ε, so the plots can draw it without the Hydra config."""
    for override in reversed(args.override or []):
        if override.startswith("finetune.constraint.epsilon="):
            try:
                return float(override.split("=", 1)[1])
            except ValueError:
                pass
    return 0.1  # configs/finetune/lagrangian_ppo.yaml default


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", choices=sorted(PROFILES), default="smoke")
    ap.add_argument("--run-name", default=None,
                    help="default: icrl_<profile>")
    ap.add_argument("--stages", default=",".join(STAGES),
                    help=f"comma-separated subset of: {','.join(STAGES)}")
    ap.add_argument("--compute", default=None,
                    help="Hydra compute group (default: the profile's — "
                         "local for smoke/local, carleton for cluster)")
    ap.add_argument("--data-root", default=None,
                    help="override paths.data_root (default: the compute group's)")
    ap.add_argument("--checkpoint-dir", default=None,
                    help="override paths.checkpoint_dir")
    ap.add_argument("--log-dir", default=None,
                    help="override paths.log_dir")
    ap.add_argument("--embeddings-dir", default=None,
                    help="default: embeddings/<run_name>")
    ap.add_argument("--safe-demos", default=None,
                    help="a .jsonl, or a directory of task_*_trace_*.json from "
                         "the collection job (default: data/demos/safe.jsonl, "
                         "falling back to $SCRATCH/trajectories/safe)")
    ap.add_argument("--unsafe-demos", default=None,
                    help="same formats (default: data/demos/unsafe.jsonl, "
                         "falling back to $SCRATCH/trajectories/unsafe)")
    ap.add_argument("--demo-limit", type=int, default=None,
                    help="use only the first N demos of each label")
    ap.add_argument("--encode-batch-size", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--strict-gate", action="store_true",
                    help="stop when the held-out AUROC gate fails")
    ap.add_argument("--plot-theme", choices=["light", "dark"], default="light",
                    help="figure theme (default: light, for papers)")
    ap.add_argument("--pdf", action="store_true",
                    help="also write vector PDFs alongside the PNGs")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb-entity", default=None)
    ap.add_argument("--override", action="append", default=[],
                    help="extra Hydra override, repeatable")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the commands without running them")
    args = ap.parse_args()

    profile = PROFILES[args.profile]
    args.run_name = args.run_name or f"icrl_{args.profile}"
    args.compute = args.compute or profile["compute"]
    resolve_effective_paths(args)
    resolve_demo_paths(args)
    args.embeddings_dir = args.embeddings_dir or os.path.join(
        args.data_root, "embeddings", args.run_name)

    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    args.selected_stages = stages
    unknown = [s for s in stages if s not in STAGE_FNS]
    if unknown:
        print(f"Unknown stage(s): {unknown}. Valid: {STAGES}", file=sys.stderr)
        return 2

    print(f"Profile : {args.profile} — {profile['description']}")
    print(f"Run name: {args.run_name}")
    print(f"Compute : {args.compute}")
    print(f"Paths   : data={args.data_root}  ckpt={args.checkpoint_dir}  "
          f"logs={args.log_dir}")
    print(f"Tasks   : train={profile['train_task_ids']}  "
          f"held-out={profile['eval_task_ids']}")
    print(f"Stages  : {', '.join(stages)}")

    results: dict = {}
    try:
        for stage in stages:
            print(f"\n{'#' * 72}\n# STAGE: {stage}\n{'#' * 72}")
            STAGE_FNS[stage](args, profile, results)
    except StageFailure as e:
        print(f"\nSTAGE FAILED: {e}", file=sys.stderr)
        print_summary(results, args)
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        print_summary(results, args)
        return 130

    print_summary(results, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
