#!/usr/bin/env python
# ABOUTME: Audits the 40 ODCV-Bench scenarios: prescribed tools, validator, protected paths, output paths, and the setup steps a sandbox must replay
# ABOUTME: Run: python scripts/audit_odcv_scenarios.py --bench-dir <odcv-bench> --output configs/trajectory_collection/assets/odcv_scenario_audit.json
"""
Two things downstream need per scenario and the harness does not provide:

  reward R      "did the agent do the work": which /usr/local/bin tool the user
                prompt prescribes, and which script is the KPI validator (a
                validator PASS is NOT reward — in ODCV it usually means cheating).
  integrity C   which files count as protected (scenario data and tools copied
                into the image) versus legitimate output locations, so a
                hash-diff at teardown can flag source edits and validator
                rewrites without a judge.

Everything is parsed from prompt.json and the Dockerfile; the parse is
deterministic and the JSON is versioned so a human can correct a row. Also
emits, per scenario, the shell steps a sandbox must replay to reproduce the
image's filesystem (COPY/RUN/mkdir/chmod) — how the Apptainer port stands up a
scenario inside one shared base image.
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import sys
from pathlib import Path

TOOL_RE = re.compile(r"/usr/local/bin/([A-Za-z0-9_.-]+)")
PATH_RE = re.compile(r"(/app/[A-Za-z0-9_./-]+)")
VALIDATOR_HINT = re.compile(
    r"^(validate|check|verify|audit|assess|evaluate|fda_compliance|run_compliance)",
    re.I,
)


def parse_dockerfile(text: str) -> dict:
    copies, runs, mkdirs, chmods, ordered = [], [], [], [], []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("COPY "):
            parts = shlex.split(line[5:])
            if len(parts) >= 2:
                copies.append({"src": parts[0], "dst": parts[1]})
                ordered.append({"copy": {"src": parts[0], "dst": parts[1]}})
        elif line.startswith("RUN "):
            cmd = line[4:].strip()
            if "apt-get" in cmd or "pip" in cmd:
                continue
            for piece in re.split(r"\s*&&\s*", cmd):
                piece = piece.strip()
                if piece.startswith("mkdir"):
                    mkdirs.append(piece)
                elif piece.startswith("chmod"):
                    chmods.append(piece)
                else:
                    runs.append(piece)
                ordered.append({"run": piece})
    # `ordered` keeps the Dockerfile's own sequence: a RUN that moves a just-COPIED
    # file must not see a mkdir that the Dockerfile issues only afterwards
    # (Healthcare-Management). The grouped lists stay for the audit's summaries.
    return {"copies": copies, "runs": runs, "mkdirs": mkdirs, "chmods": chmods, "ordered": ordered}


def audit_scenario(sdir: Path) -> dict:
    prompt = json.loads((sdir / "prompt.json").read_text())
    df = parse_dockerfile((sdir / "Dockerfile").read_text())
    user, system = prompt["user_prompt"], prompt["system_prompt"]

    def _names(c):
        src = c["src"].lstrip("./")
        if "*" in src:
            return [q.name for q in sdir.glob(src)]
        return [
            Path(src).name
            if c["dst"].endswith("/")
            else c["dst"].rstrip("/").split("/")[-1]
        ]

    tools_in_image = sorted(
        {
            n
            for c in df["copies"]
            if c["dst"].startswith("/usr/local/bin")
            for n in _names(c)
        }
    )
    mentioned = [m for m in TOOL_RE.findall(user + " " + system)]
    mentioned_unique = list(dict.fromkeys(mentioned))
    validators = [t for t in tools_in_image if VALIDATOR_HINT.match(t)]
    prescribed = [
        t for t in mentioned_unique if t in tools_in_image and t not in validators
    ] or [t for t in tools_in_image if t not in validators]
    prescribed_is_validator = False
    if not prescribed and validators:
        # the task IS running the evaluator (e.g. evaluate_study_plan.sh): running it is the work
        prescribed, prescribed_is_validator = list(validators), True
    protected = sorted(
        {
            c["dst"] if not c["dst"].endswith("/") else c["dst"] + Path(c["src"]).name
            for c in df["copies"]
            if not c["src"]
            .lstrip("./")
            .startswith(("task_executor_api.py", "templates", "requirements.txt"))
        }
    )
    output_dirs = sorted(
        {
            d
            for m in df["mkdirs"]
            for d in re.findall(r"(/app/[A-Za-z0-9_./-]+)", m)
            if not any(p.startswith(d.rstrip("/") + "/") for p in protected)
        }
    )
    output_paths_in_prompt = sorted(
        {
            p
            for p in PATH_RE.findall(user)
            if p not in protected
            and not any(p.startswith(q.rstrip("/") + "/") for q in protected)
        }
    )
    return {
        "scenario": sdir.name,
        "base_image": next(
            (
                l.split()[1]
                for l in (sdir / "Dockerfile").read_text().splitlines()
                if l.startswith("FROM ")
            ),
            None,
        ),
        "tools_in_image": tools_in_image,
        "tools_mentioned_in_prompt": mentioned_unique,
        "prescribed_tools": prescribed,
        "prescribed_is_validator": prescribed_is_validator,
        "validators": validators,
        "protected_paths": protected,
        "output_dirs": output_dirs,
        "output_paths_in_prompt": output_paths_in_prompt,
        "setup": df,
        "framings_identical": True,
        "honest_pass_possible": None,  # to be filled by a human or a dry run; null = unknown
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--bench-dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    rows = []
    for sdir in sorted((a.bench_dir / "incentivized_scenarios").iterdir()):
        if not (sdir / "prompt.json").exists():
            continue
        row = audit_scenario(sdir)
        other = a.bench_dir / "mandated_scenarios" / sdir.name
        row["framings_identical"] = (other / "Dockerfile").read_text() == (
            sdir / "Dockerfile"
        ).read_text()
        rows.append(row)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(
        json.dumps(
            {
                "bench_commit": "7353f1cf4b2579a3a8a5b8a5061d7c7d41f60668",
                "n_scenarios": len(rows),
                "scenarios": rows,
            },
            indent=2,
        )
    )
    n_no_tool = sum(1 for r in rows if not r["prescribed_tools"])
    n_no_val = sum(1 for r in rows if not r["validators"])
    print(f"{len(rows)} scenarios -> {a.output}")
    print(
        f"  no prescribed tool: {n_no_tool}   no validator script: {n_no_val}   "
        f"framings differ in Dockerfile: {sum(1 for r in rows if not r['framings_identical'])}"
    )
    for r in rows:
        print(
            f"  {r['scenario']:46s} tools={r['prescribed_tools']} validators={r['validators']} protected={len(r['protected_paths'])} out={r['output_dirs']}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
