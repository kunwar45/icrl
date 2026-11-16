#!/usr/bin/env python
# ABOUTME: Publishes one finished secret-keeping run to a private, dated Hugging Face model repo (adapters, constraint heads, configs, eval outputs, results) and its demos to a dataset repo
# ABOUTME: Run (any machine with internet and `hf auth login`): python scripts/push_secret_keeping_run_to_huggingface.py --run-dir <$SCRATCH/icrl/secret_keeping/<run_name>> [--with-demos] [--dry-run]
"""
Naming (lowercase, hyphens, dated by the day the run finished, UTC):

  model    <namespace>/<YYYY-MM-DD>-secret-keeping-<method>-qwen2.5-7b-v<N>
  dataset  <namespace>/<YYYY-MM-DD>-secret-keeping-demos

<method> comes from the run name (METHODS below) and matches the names used in
docs/secret-keeping.md; <N> is the run name's _vN. Everything is private.

The model repo holds the run directory minus what is either huge or
regenerable: trainer checkpoints and optimiser state, server logs, cached
embeddings. Adapters load with peft: PeftModel.from_pretrained(base,
"<repo>", subfolder="adapter"). The demos (training chats with the expert's and
the policy's replies, judge scores and constraint labels) go to the dataset repo
once (--with-demos), from the run that generated them; later runs reuse copies.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

BENCHMARK = "secret-keeping"
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
BASE_SLUG = "qwen2.5-7b"

# run-name prefix (before _qwen7b_vN) -> (method slug, one-line description)
METHODS = {
    "secret_keeping_icrl_loop": (
        "icrl",
        "ICRL (Malik et al. 2021): alternating max-entropy constraint updates on the current policy's "
        "replies and Lagrangian GRPO with dual ascent on lambda.",
    ),
    "secret_keeping_constrained_grpo": (
        "classifier-grpo",
        "Baseline (not ICRL): a supervised constraint classifier trained once, then GRPO with a fixed "
        "penalty.",
    ),
    "secret_keeping_constrained_dpo": (
        "classifier-dpo",
        "Baseline (superseded, not ICRL): the one-shot constraint classifier with Lagrangian DPO rounds.",
    ),
    "secret_keeping_expert_sft": (
        "expert-sft",
        "Baseline: supervised fine-tuning on the expert demonstrations.",
    ),
    "secret_keeping_grpo_exact_leak_cost": (
        "exact-check-grpo",
        "Baseline: GRPO with the ground-truth constraint check as the cost.",
    ),
    "secret_keeping_grpo_llm_judge_cost": (
        "llm-judge-grpo",
        "Baseline: GRPO with a prompted LLM safety judge as the cost.",
    ),
}

IGNORE = [
    "**/trainer/**",
    "**/*.log",
    "**/vllm_*",
    "**/*_embeddings.pt",
    "**/dpo_pairs.json",
    "demos/**",
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
    for name in ("results.json", "state.json"):
        p = run_dir / name
        if p.exists():
            return (
                datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc)
                .date()
                .isoformat()
            )
    raise SystemExit(
        f"{run_dir} has no results.json or state.json: the run has not finished"
    )


def final_adapter(run_dir: Path) -> Path:
    """The run's final LoRA adapter, from state.json (an absolute cluster path) mapped into run_dir."""
    state = json.loads((run_dir / "state.json").read_text())
    parts = Path(state["adapter"]).parts
    rel = Path(*parts[parts.index(run_dir.name) + 1:])
    path = run_dir / rel
    if not (path / "adapter_config.json").exists():
        raise SystemExit(f"final adapter {path} is missing adapter_config.json")
    return path


def model_card(run_name: str, repo_id: str, desc: str, run_dir: Path) -> str:
    lines = [
        "---",
        f"base_model: {BASE_MODEL}",
        "library_name: peft",
        "tags: [secret-keeping, icrl, constrained-rl, lora]",
        "---",
        f"# {repo_id.split('/')[1]}",
        "",
        f"Run `{run_name}` from the ICRL secret-keeping project. {desc}",
        "",
        "Task: a customer-service chatbot holds a secret code in its system prompt and is never told to "
        "keep it private; an LLM judge rewards doing what the user asks. Metric: the share of attack-chat "
        "replies that reveal the code (constraint violation rate) on held-out businesses and attack styles.",
        "",
        "```python",
        "from peft import PeftModel",
        "from transformers import AutoModelForCausalLM",
        f"base = AutoModelForCausalLM.from_pretrained('{BASE_MODEL}')",
        f"model = PeftModel.from_pretrained(base, '{repo_id}', subfolder='adapter')",
        "```",
    ]
    results = run_dir / "results.json"
    if results.exists():
        cmp = json.loads(results.read_text()).get("comparison", {})
        att = cmp.get("attack", {}).get("leak_rate", {})
        if "tuned_nominal" in att:
            v, (lo, hi) = att["tuned_nominal"][0], att["tuned_nominal"][1]
            b = att["base_nominal"][0]
            lines += [
                "",
                f"Test constraint violation rate on attack chats: **{v:.1%}** "
                f"(95% CI {lo:.1%}-{hi:.1%}); base model {b:.1%}. Details in `results.json`.",
            ]
    lines += [
        "",
        "Full method and caveats: `docs/secret-keeping.md` in the code repository.",
    ]
    return "\n".join(lines) + "\n"


def demos_card(repo_id: str, run_name: str) -> str:
    return (
        "\n".join(
            [
                "---",
                "tags: [secret-keeping, icrl, demonstrations]",
                "---",
                f"# {repo_id.split('/')[1]}",
                "",
                f"Training chats for the secret-keeping ICRL project, from run `{run_name}`. One row per chat: the "
                "system prompt fields (business, secret type, secret), the user message, 4 replies from the model "
                "told the confidentiality rule (`expert_replies`) and 4 from the model without it "
                "(`policy_replies`), their LLM-judge rewards, the oracle expert's chosen reply (`expert_pick`), "
                "and ground-truth constraint labels (`expert_leaks`, `policy_leaks`, audit only).",
                "",
            ]
        )
        + "\n"
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--namespace", default="icrl-finetuning")
    ap.add_argument(
        "--with-demos",
        action="store_true",
        help="also publish demos/ as the dataset; only for the run that generated them (later runs copy them)",
    )
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    run_dir = a.run_dir.resolve()
    run_name = run_dir.name
    slug, desc, version = run_parts(run_name)
    day = finished_date(run_dir)
    model_repo = f"{a.namespace}/{day}-{BENCHMARK}-{slug}-{BASE_SLUG}-v{version}"
    print(f"{run_name} -> {model_repo} (private)")

    demos_repo = None
    rows = run_dir / "demos" / "rows.jsonl"
    if a.with_demos and rows.exists():
        demos_day = (
            datetime.fromtimestamp(rows.stat().st_mtime, tz=timezone.utc)
            .date()
            .isoformat()
        )
        demos_repo = f"{a.namespace}/{demos_day}-{BENCHMARK}-demos"
        print(f"demos -> {demos_repo} (private dataset)")
    if a.dry_run:
        return

    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(model_repo, repo_type="model", private=True, exist_ok=True)
    api.upload_folder(
        repo_id=model_repo,
        folder_path=str(run_dir),
        ignore_patterns=IGNORE,
        commit_message=f"Upload run {run_name}",
    )
    final = final_adapter(run_dir)
    api.upload_folder(repo_id=model_repo, folder_path=str(final), path_in_repo="adapter",
                      commit_message="Final adapter at adapter/")
    api.upload_file(
        path_or_fileobj=model_card(run_name, model_repo, desc, run_dir).encode(),
        path_in_repo="README.md",
        repo_id=model_repo,
        commit_message="Model card",
    )
    if demos_repo:
        api.create_repo(demos_repo, repo_type="dataset", private=True, exist_ok=True)
        existing = api.list_repo_files(demos_repo, repo_type="dataset")
        if "rows.jsonl" not in existing:
            api.upload_folder(
                repo_id=demos_repo,
                repo_type="dataset",
                folder_path=str(run_dir / "demos"),
                allow_patterns=["rows.jsonl", "summary.json"],
                commit_message=f"Demos from run {run_name}",
            )
            api.upload_file(
                path_or_fileobj=demos_card(demos_repo, run_name).encode(),
                path_in_repo="README.md",
                repo_id=demos_repo,
                repo_type="dataset",
                commit_message="Dataset card",
            )
    print(f"published https://huggingface.co/{model_repo}")


if __name__ == "__main__":
    main()
