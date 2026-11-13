#!/usr/bin/env python
# ABOUTME: Publishes the accepted MACHIAVELLI demos (expert: clean steps under the expert prompt; unsafe: avoidable violations under the nominal prompt) as two private, dated HF datasets
# ABOUTME: Run: python scripts/push_machiavelli_demos_to_huggingface.py --demos-dir <run>/demos --namespace kunwar45 [--date YYYY-MM-DD] [--dry-run]
"""
Accepted means exactly what the constraint trainer uses
(src/icrl_dual_training/machiavelli_step_constraint.py: accept_expert_step,
accept_unsafe_step), with labels from the benchmark's own scene annotations:

  expert  steps from expert-prompt episodes whose chosen action did not violate
  unsafe  steps from nominal-prompt episodes whose chosen action violated while a
          non-violating option existed at that state (story-forced harm excluded)

Each row keeps the full provenance: the exact system and user prompt the model
saw, its raw answer, the action, every option with its ground-truth harms, and
the run it came from. Datasets are named <namespace>/<date>-machiavelli-<set>
and are private by default: the scene text is from MACHIAVELLI's games, which
the benchmark distributes behind a password and without a licence for the text.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.environments.machiavelli.rollout_collector import load_episodes  # noqa: E402
from src.icrl_dual_training.machiavelli_step_constraint import (  # noqa: E402
    accept_expert_step,
    accept_unsafe_step,
)


def rows_for(episodes: list[dict], accept, set_name: str, run: str) -> list[dict]:
    out = []
    for e in episodes:
        for s in e["steps"]:
            if not accept(s):
                continue
            out.append(
                {
                    "set": set_name,
                    "run": run,
                    "game": e["game"],
                    "seed": e["seed"],
                    "prompt_kind": e["kind"],
                    "step": s["t"],
                    "system_prompt": s["system"],
                    "user_prompt": s["user"],
                    "model_output": s["response"],
                    "action": s["action"],
                    "chosen_option": s["choices"][s["action"]],
                    "options": s["choices"],
                    "chosen_harms": s["harms"],
                    "option_harms": s["option_harms"],
                    "violated": s["violated"],
                    "avoidable": s["avoidable"],
                    "forced": s["forced"],
                    "reward": s["reward"],
                }
            )
    return out


def card(set_name: str, rows: list[dict], src: dict) -> str:
    games = sorted({r["game"] for r in rows})
    rule = {
        "expert": "steps under the expert prompt whose chosen action did not violate any of the 13 morality labels (> 0.5)",
        "unsafe": "steps under the nominal prompt whose chosen action violated while a non-violating option existed",
    }[set_name]
    return (
        f"# MACHIAVELLI {set_name} demos\n\n"
        f"Accepted rule: {rule}. Labels are the benchmark's scene annotations, checked for every option by "
        f"save/restore of the game state.\n\n"
        f"- rows: {len(rows)}\n- games ({len(games)}): {', '.join(games)}\n"
        f"- policy: {src.get('model')}\n- run: {src.get('run')} (commit {src.get('commit')})\n\n"
        "Scene text comes from the MACHIAVELLI games (Pan et al., 2023), which the benchmark distributes without "
        "a licence for the text: keep this dataset private.\n"
    )


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--demos-dir",
        type=Path,
        required=True,
        help="the run's demos/ folder (expert.jsonl, nominal.jsonl)",
    )
    ap.add_argument("--namespace", default=os.environ.get("HF_NAMESPACE"))
    ap.add_argument(
        "--date",
        default=None,
        help="generation date; default: the demo files' modification date",
    )
    ap.add_argument("--model", default=None)
    ap.add_argument("--run", default=None)
    ap.add_argument("--commit", default=None)
    ap.add_argument(
        "--public", action="store_true", help="publish publicly (default private)"
    )
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--staging",
        type=Path,
        default=None,
        help="where to write the files before upload",
    )
    a = ap.parse_args()

    expert_eps = load_episodes(a.demos_dir / "expert.jsonl")
    nominal_eps = load_episodes(a.demos_dir / "nominal.jsonl")
    when = (
        a.date
        or date.fromtimestamp(
            (a.demos_dir / "expert.jsonl").stat().st_mtime
        ).isoformat()
    )
    src = {
        "model": a.model,
        "run": a.run or a.demos_dir.parent.name,
        "commit": a.commit,
    }
    sets = {
        "expert": rows_for(expert_eps, accept_expert_step, "expert", src["run"]),
        "unsafe": rows_for(nominal_eps, accept_unsafe_step, "unsafe", src["run"]),
    }
    n_steps = {
        "expert": sum(len(e["steps"]) for e in expert_eps),
        "unsafe": sum(len(e["steps"]) for e in nominal_eps),
    }
    for k, rows in sets.items():
        print(
            f"{k}: {len(rows)} accepted of {n_steps[k]} steps over {len({r['game'] for r in rows})} games"
        )
        if not rows:
            print(f"refusing to publish: no accepted {k} rows")
            return 1

    staging = a.staging or (a.demos_dir.parent / "hf_staging")
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    for k, rows in sets.items():
        d = staging / k
        d.mkdir(parents=True, exist_ok=True)
        with open(d / f"{k}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        (d / "README.md").write_text(card(k, rows, src))
        repo_id = f"{a.namespace}/{when}-machiavelli-{k}"
        vis = "PUBLIC" if a.public else "private"
        if a.dry_run:
            print(
                f"DRY RUN: would publish {len(rows)} rows to {repo_id} ({vis}) from {d}"
            )
            continue
        if not a.namespace or not token:
            print("need --namespace and HF_TOKEN / HUGGINGFACE_TOKEN to publish")
            return 1
        from huggingface_hub import HfApi

        api = HfApi(token=token)
        api.create_repo(
            repo_id, repo_type="dataset", private=not a.public, exist_ok=True
        )
        api.upload_folder(folder_path=str(d), repo_id=repo_id, repo_type="dataset")
        print(
            f"published {len(rows)} rows ({vis}) to https://huggingface.co/datasets/{repo_id}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
