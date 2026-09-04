#!/usr/bin/env python
# ABOUTME: Collects ODCV-Bench rollouts on a SLURM node: N cells at a time through the Apptainer sandbox against a served model
# ABOUTME: Run: python scripts/collect_odcv_rollouts.py --config configs/trajectory_collection/odcv_rollouts.yaml [--override key=value]
"""
The ODCV counterpart of trajectory collection: roll the policy out on every
cell (scenario × framing) for `passes` passes at `temperature`, `concurrency`
sandboxes at once, and write one directory per rollout in the published
layout LASR uses:

    <output.dir>/rollouts/<framing>/<Scenario>/pass<N>/{messages_record.txt,result.json,...}
    <output.dir>/summary.json      per-cell and overall submission, integrity-violation and
                                   prescribed-tool rates — the guards the fine-tuning loop reads

No judge runs here. Labels come later from the constraint head (training) or
from the two-judge protocol on the login node (evaluation).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.environments.odcv.apptainer_sandbox import run_scenario  # noqa: E402


def load_cfg(path: Path, overrides: list[str]):
    from omegaconf import OmegaConf

    return OmegaConf.merge(OmegaConf.load(path), OmegaConf.from_dotlist(overrides))


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--override", action="append", default=[])
    a = ap.parse_args()
    cfg = load_cfg(a.config, a.override)

    bench = Path(cfg.benchmark.bench_dir)
    audit = json.loads(Path(cfg.benchmark.audit_file).read_text())
    rows = {r["scenario"]: r for r in audit["scenarios"]}
    scenarios = (
        list(cfg.benchmark.scenarios)
        if cfg.benchmark.get("scenarios")
        else sorted(rows)
    )
    exclude = set(cfg.benchmark.get("exclude_scenarios", []) or [])
    framings = list(cfg.benchmark.get("framings", ["incentivized", "mandated"]))
    out = Path(cfg.output.dir)
    out.mkdir(parents=True, exist_ok=True)

    jobs = []
    for s in scenarios:
        if s in exclude or s not in rows:
            continue
        for f in framings:
            for p in range(1, int(cfg.rollouts.passes) + 1):
                d = out / "rollouts" / f / s / f"pass{p}"
                if (d / "result.json").exists() and not cfg.rollouts.get(
                    "overwrite", False
                ):
                    continue
                jobs.append((s, f, p, d))
    print(
        f"{len(jobs)} rollouts to run ({len(scenarios) - len(exclude)} scenarios × {len(framings)} framings × "
        f"{cfg.rollouts.passes} passes, concurrency {cfg.rollouts.concurrency}, T={cfg.rollouts.temperature})",
        flush=True,
    )

    def one(job):
        s, f, p, d = job
        try:
            return run_scenario(
                bench / f"{f}_scenarios" / s,
                f,
                d,
                rows[s],
                openai_base_url=cfg.model.base_url,
                model=cfg.model.name,
                api_key=cfg.model.get("api_key", "local-vllm-no-auth"),
                temperature=float(cfg.rollouts.temperature),
                base_sif=cfg.benchmark.base_sif,
                executor_timeout_s=int(cfg.rollouts.timeout_s),
            )
        except Exception as e:  # one broken cell must not take the pass down
            return {
                "scenario": s,
                "framing": f,
                "status": "EXCEPTION",
                "error": str(e)[:500],
            }

    t0 = time.time()
    results = []
    with ThreadPoolExecutor(max_workers=int(cfg.rollouts.concurrency)) as ex:
        futs = {ex.submit(one, j): j for j in jobs}
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            results.append(r)
            print(
                f"[{i}/{len(jobs)} {time.time() - t0:6.0f}s] {r.get('framing')}/{r.get('scenario')}: {r.get('status')} "
                f"cmds={r.get('n_commands')} submitted={r.get('submitted')} integrity_changed={len(r.get('integrity_changed') or [])} "
                f"tool={r.get('prescribed_tool_ran')} {r.get('seconds')}s",
                flush=True,
            )

    # fold in rollouts that already existed on disk so the summary covers the whole grid
    for d in (out / "rollouts").glob("*/*/pass*/result.json"):
        r = json.loads(d.read_text())
        if not any(
            x.get("scenario") == r.get("scenario")
            and x.get("framing") == r.get("framing")
            and x.get("port") == r.get("port")
            for x in results
        ):
            results.append(r)
    per_cell = defaultdict(list)
    for r in results:
        per_cell[(r.get("framing"), r.get("scenario"))].append(r)

    def rate(key):
        cells = [
            sum(bool(x.get(key)) for x in v) / len(v) for v in per_cell.values() if v
        ]
        return round(100 * sum(cells) / len(cells), 1) if cells else None

    summary = {
        "n_rollouts": len(results),
        "n_cells": len(per_cell),
        "ok_pct": round(
            100 * sum(r.get("status") == "ok" for r in results) / max(1, len(results)),
            1,
        ),
        "submitted_pct": rate("submitted"),
        "integrity_violated_pct": rate("integrity_violated"),
        "prescribed_tool_ran_pct": rate("prescribed_tool_ran"),
        "validator_ran_pct": rate("validator_ran"),
        "seconds": round(time.time() - t0, 1),
        "model": cfg.model.name,
        "temperature": cfg.rollouts.temperature,
        "per_cell": {
            f"{k[0]}/{k[1]}": {
                "n": len(v),
                "submitted": sum(bool(x.get("submitted")) for x in v),
                "integrity_violated": sum(bool(x.get("integrity_violated")) for x in v),
                "prescribed_tool_ran": sum(
                    bool(x.get("prescribed_tool_ran")) for x in v
                ),
            }
            for k, v in sorted(per_cell.items())
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "per_cell"}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
