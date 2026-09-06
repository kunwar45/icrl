#!/usr/bin/env python
# ABOUTME: Runs one ODCV-Bench cell (scenario + framing) through the Apptainer sandbox and prints the result — the smoke test for the port
# ABOUTME: Run: python scripts/run_odcv_scenario.py --bench-dir <odcv-bench> --scenario Ai-Pharma-Quality-Control-Data-Suppression --framing incentivized --base-url http://127.0.0.1:8000/v1 --model <served-name> --out <dir>
"""
One cell, end to end, with no Docker: start the sandbox instance, replay the
scenario's Dockerfile steps into it, run the vendored agent loop natively
against the served model, hash the protected files before and after, stop.

The served model is a local vLLM (compute node) or any OpenAI-compatible server (login node,
for a smoke test): pass --base-url and --api-key accordingly.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.environments.odcv.apptainer_sandbox import DEFAULT_BASE_SIF, run_scenario  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--bench-dir",
        type=Path,
        required=True,
        help="the vendored odcv-bench directory",
    )
    ap.add_argument(
        "--audit",
        type=Path,
        default=Path("configs/trajectory_collection/assets/odcv_scenario_audit.json"),
    )
    ap.add_argument("--scenario", required=True)
    ap.add_argument(
        "--framing", choices=["incentivized", "mandated"], default="incentivized"
    )
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument(
        "--api-key", default=os.environ.get("ODCV_API_KEY", "local-vllm-no-auth")
    )
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--base-sif", default=DEFAULT_BASE_SIF)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--timeout", type=int, default=2400)
    a = ap.parse_args()

    audit = json.loads(a.audit.read_text())
    row = next((r for r in audit["scenarios"] if r["scenario"] == a.scenario), None)
    if row is None:
        print(f"{a.scenario} is not in {a.audit}", file=sys.stderr)
        return 2
    scenario_dir = a.bench_dir / f"{a.framing}_scenarios" / a.scenario
    result = run_scenario(
        scenario_dir,
        a.framing,
        a.out,
        row,
        openai_base_url=a.base_url,
        model=a.model,
        api_key=a.api_key,
        temperature=a.temperature,
        base_sif=a.base_sif,
        executor_timeout_s=a.timeout,
    )
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if not k.startswith("integrity_") or k == "integrity_changed"
            },
            indent=2,
        )
    )
    return 0 if result.get("status") == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
