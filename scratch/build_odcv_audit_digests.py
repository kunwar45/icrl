# ABOUTME: Writes compact per-trace digests of an ODCV contrast set, chunked by scenario, for a label audit
# ABOUTME: Run: python scratch/build_odcv_audit_digests.py <set_root> <out_dir> [--chunks 12]
"""
An auditor should see a whole scenario at once — every expert and unsafe trace
of the same cell — so it can judge them against each other, not in isolation.
Traces are sorted by (scenario, framing, label) and cut into contiguous chunks.
Each digest carries the user prompt, every action with the reasoning that
preceded it and a slice of what came back, and the judge's verdict.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path


def digest(t: dict, path: Path) -> str:
    rep = t["safety_report"][0]
    prompt = t.get("scenario_prompt", "")
    user = prompt.split("[user]", 1)[1].strip() if "[user]" in prompt else prompt
    system = prompt.split("[user]", 1)[0].replace("[system]", "").strip()
    lines = [
        f"### FILE {path.name}",
        f"label={t['set']} | judge_severity={rep.get('severity_score')} | scenario={t['task_id']} | "
        f"framing={t['task_variant']} | steps={t['n_steps']} | submitted={t['submitted']}",
        f"SYSTEM (first 400 chars): {system[:400]}",
        f"USER: {user[:1200]}",
        "STEPS:",
    ]
    for s in t["steps"]:
        r = (s.get("reasoning") or "").replace("\n", " ")
        a = s["action"].replace("\n", " ⏎ ")
        o = (s.get("observation") or "").replace("\n", " ")
        if r:
            lines.append(f"  [{s['step_idx']}] reason: {r[:220]}")
        lines.append(f"  [{s['step_idx']}] ACTION: {a[:400]}")
        if o:
            lines.append(f"       saw-before-acting: {o[:200]}")
    lines.append(f"JUDGE ({rep.get('judge_model')}): {rep.get('reason', '')[:500]}")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("set_root", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--chunks", type=int, default=12)
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)

    traces = []
    for label in ("expert", "unsafe"):
        for p in sorted((a.set_root / label).glob("task_*_trace_*.json")):
            traces.append((json.loads(p.read_text()), p))
    traces.sort(
        key=lambda x: (x[0]["task_id"], x[0]["task_variant"], x[0]["set"], x[1].name)
    )

    # contiguous chunks that never split a (scenario, framing) cell
    cells = defaultdict(list)
    order = []
    for t, p in traces:
        k = (t["task_id"], t["task_variant"])
        if k not in cells:
            order.append(k)
        cells[k].append((t, p))
    target = len(traces) / a.chunks
    chunks, cur = [], []
    for k in order:
        cur.extend(cells[k])
        if len(cur) >= target and len(chunks) < a.chunks - 1:
            chunks.append(cur)
            cur = []
    if cur:
        chunks.append(cur)

    index = []
    for i, ch in enumerate(chunks):
        out = a.out_dir / f"chunk_{i:02d}.md"
        out.write_text("".join(digest(t, p) + "\n" for t, p in ch))
        n_e = sum(t["set"] == "expert" for t, _ in ch)
        index.append(
            {
                "chunk": out.name,
                "n": len(ch),
                "expert": n_e,
                "unsafe": len(ch) - n_e,
                "cells": sorted({f"{t['task_id']}/{t['task_variant']}" for t, _ in ch}),
                "chars": out.stat().st_size,
            }
        )
    (a.out_dir / "index.json").write_text(json.dumps(index, indent=2))
    for row in index:
        print(
            f"{row['chunk']}: n={row['n']} expert={row['expert']} unsafe={row['unsafe']} "
            f"cells={len(row['cells'])} chars={row['chars']}"
        )


if __name__ == "__main__":
    main()
