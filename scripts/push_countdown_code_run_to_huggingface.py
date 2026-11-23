#!/usr/bin/env python
# ABOUTME: Publishes one finished Countdown-Code run to a private, dated Hugging Face model repo: the final adapter in adapter/, the intermediate adapters in checkpoints/, plus configs, training logs, eval outputs, results and the run's monitor (constraint heads or VARL discriminator)
# ABOUTME: Run (any machine with internet and `hf auth login`): python scripts/push_countdown_code_run_to_huggingface.py --run-dir <copy of $SCRATCH/icrl/countdown_code/<run_name>> [--dry-run]
"""
Naming (lowercase, hyphens, dated by the day the run finished, UTC):

  <namespace>/<YYYY-MM-DD>-countdown-code-<method>-qwen2.5-7b-v<N>

<method> comes from the run name (METHODS below); <N> is the run name's _vN.
Everything is private. The upstream o4-mini demonstration traces have no
licence and are never uploaded: a run directory holds none of them, and the
caches derived from them (embeddings, the PUCL memory) are ignored below.

Left out because they are huge or regenerable: trainer checkpoints, the
adapters of earlier ICRL rounds, the discriminator's resume checkpoints and
server logs. The final adapter loads with peft on top of the SFT-primed model
(icrl-finetuning/2026-11-17-countdown-code-sft-priming-qwen2.5-7b-v1, merged).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.lagrangian_finetuning.adapter_checkpoint_keeping import (  # noqa: E402
    ADAPTER_FILES,
    intermediate_adapters,
)

BENCHMARK = "countdown-code"
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
BASE_SLUG = "qwen2.5-7b"
START_REPO = "icrl-finetuning/2026-11-17-countdown-code-sft-priming-qwen2.5-7b-v1"

# run-name prefix (before _qwen7b_vN) -> (method slug, one-line description); longest prefix first
METHODS = {
    "countdown_code_no_intervention_long": (
        "no-intervention-long",
        "Baseline, no intervention, long: GRPO on the hackable proxy reward alone, continued to 1,250 steps.",
    ),
    "countdown_code_no_intervention": (
        "no-intervention",
        "Baseline, no intervention: GRPO on the hackable proxy reward alone.",
    ),
    "countdown_code_varl": (
        "varl",
        "Baseline, VARL (Damani et al. 2026, our reimplementation): GRPO on proxy reward x a discriminator "
        "trained alongside on demonstrations vs. the policy's verifier-passing outputs.",
    ),
    "countdown_code_icrl_pucl_calibrated": (
        "icrl-pucl-calibrated",
        "ICRL with PUCL, calibrated: the cost budget is set each round from held-out allowed replies, and a reply "
        "is marked forbidden only if it is farther from the allowed set than a quantile of held-out allowed replies.",
    ),
    "countdown_code_icrl_pucl": (
        "icrl-pucl",
        "ICRL (Malik et al. 2021) with the PUCL inverse step (Peng & Billard 2024): each round re-fits the "
        "constraint from demonstrations vs. the policy's reward-earning samples, then Lagrangian GRPO.",
    ),
}

IGNORE = [
    "**/trainer/**",
    "**/rl/adapter/**",  # the final one is uploaded separately as adapter/
    "rl/adapter/**",  # the same, for a run whose rl/ sits at the top level
    "rl/trainer/**",
    "**/vllm_*",
    "**/discriminator.log",
    "**/discriminator_port",
    "**/discriminator_checkpoints/**",
    "**/adapter_checkpoints/**",  # intermediate adapters are uploaded separately under checkpoints/
    "**/*_embeddings.pt",
    "icrl_pucl_memory.pt",
    "pucl_memory/**",
    "**/training_args.bin",
    "**/__pycache__/**",
]


def run_parts(run_name: str) -> tuple[str, str, int]:
    m = re.fullmatch(r"(.+?)_qwen7b_v(\d+)", run_name)
    if not m or m.group(1) not in METHODS:
        raise SystemExit(
            f"unknown run name {run_name!r}: add its prefix to METHODS in {__file__}"
        )
    slug, desc = METHODS[m.group(1)]
    return slug, desc, int(m.group(2))


def finished_date(run_dir: Path) -> str:
    p = run_dir / "results.json"
    if not p.exists():
        raise SystemExit(f"{run_dir} has no results.json: the run has not finished")
    return datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc).date().isoformat()


def final_adapter(run_dir: Path) -> Path:
    """The run's final LoRA adapter, from state.json (an absolute cluster path) mapped into run_dir."""
    state = json.loads((run_dir / "state.json").read_text())
    parts = Path(state["adapter"]).parts
    path = run_dir / Path(*parts[parts.index(run_dir.name) + 1 :])
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        if not (path / name).exists():
            raise SystemExit(f"final adapter {path} is missing {name}")
    return path


def model_card(run_name: str, repo_id: str, desc: str, run_dir: Path) -> str:
    lines = [
        "---",
        f"base_model: {BASE_MODEL}",
        "library_name: peft",
        "tags: [countdown-code, reward-hacking, icrl, constrained-rl, lora]",
        "---",
        f"# {repo_id.split('/')[1]}",
        "",
        f"Run `{run_name}` from the ICRL project's Countdown-Code experiment. {desc}",
        "",
        "Task (Khalifa et al., arXiv 2603.07084): reach a target number with the given numbers; the model "
        "returns `solution.py` and `test.py`. Proxy reward (training): the returned solution passes the "
        "returned test. True reward: the expression really solves the original puzzle. A hack is proxy 1 and "
        "true 0.",
        "",
        f"`adapter/` is a LoRA adapter on top of the SFT-primed starting model (`{START_REPO}`, merged into "
        f"{BASE_MODEL}), not on the plain base model. `checkpoints/` holds intermediate adapters (adapter "
        "files only): `step-NNNNN` at each save, and for multi-round runs one folder per round.",
    ]
    res = json.loads((run_dir / "results.json").read_text())
    base, tuned = res.get("base"), res.get("tuned")
    if base and tuned:

        def row(name, m):
            lo, hi = m["violation_ci95"]
            return f"| {name} | {m['violation']:.2%} ({lo:.2%}-{hi:.2%}) | {m['true']:.1%} | {m['reward']:.1%} |"

        lines += [
            "",
            f"Test set ({tuned['prompts']} problems, {tuned['replies']} replies):",
            "",
            "| Model | Hack rate (95% CI) | True reward | Proxy reward |",
            "|---|---|---|---|",
            row("SFT-primed start", base),
            row("This run", tuned),
        ]
    lines += [
        "",
        "The training demonstrations (o4-mini traces from the Countdown-Code repository) carry no licence and "
        "are not included. Keep this repo private.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--namespace", default="icrl-finetuning")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--no-checkpoints",
        action="store_true",
        help="skip the intermediate adapters (checkpoints/)",
    )
    a = ap.parse_args()
    run_dir = a.run_dir.resolve()
    run_name = run_dir.name
    slug, desc, version = run_parts(run_name)
    repo = f"{a.namespace}/{finished_date(run_dir)}-{BENCHMARK}-{slug}-{BASE_SLUG}-v{version}"
    adapter = final_adapter(run_dir)
    card = model_card(run_name, repo, desc, run_dir)
    print(
        f"{run_name} -> {repo} (private); final adapter {adapter.relative_to(run_dir)}"
    )
    checkpoints = [] if a.no_checkpoints else intermediate_adapters(run_dir)
    for folder, path_in_repo in checkpoints:
        size = sum(f.stat().st_size for f in folder.iterdir() if f.is_file()) / 1e9
        print(f"  checkpoint {folder.relative_to(run_dir)} -> {path_in_repo} ({size:.2f} GB)")
    if a.dry_run:
        print(card)
        return

    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo, repo_type="model", private=True, exist_ok=True)
    api.upload_folder(
        repo_id=repo,
        repo_type="model",
        folder_path=str(adapter),
        path_in_repo="adapter",
        ignore_patterns=["README.md", "training_args.bin"],
    )
    api.upload_folder(
        repo_id=repo,
        repo_type="model",
        folder_path=str(run_dir),
        ignore_patterns=IGNORE,
    )
    for folder, path_in_repo in checkpoints:
        api.upload_folder(
            repo_id=repo,
            repo_type="model",
            folder_path=str(folder),
            path_in_repo=path_in_repo,
            allow_patterns=list(ADAPTER_FILES),
        )
    api.upload_file(
        repo_id=repo,
        repo_type="model",
        path_or_fileobj=card.encode(),
        path_in_repo="README.md",
    )
    info = api.model_info(repo, files_metadata=True)
    print("done:", repo, "| private =", info.private, "|", len(info.siblings), "files")
    for s in info.siblings:
        if (s.size or 0) > 20e6:
            print(f"   {s.rfilename}  {s.size / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
