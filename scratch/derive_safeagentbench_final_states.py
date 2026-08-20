#!/usr/bin/env python3
# ABOUTME: Derives final_state specs for SafeAgentBench tasks that ship none, by replaying their reference plan
# ABOUTME: Run on a GPU node: python scratch/derive_safeagentbench_final_states.py [--limit N]
"""
Widen the verifiable task pool.

Only 149/299 hazardous and 88/299 benign tasks ship a `final_state`, and without
one an episode cannot be proven, so the rest are unusable. That left 21 benign
tasks the 32B can actually complete — few enough that C_theta memorised them
(training AUROC 0.828, held-out 0.511 on 2026-08-19).

A task's `final_state` is just "what the world looks like once the task is done",
and the benchmark ships the reference action sequence that does it. So: snapshot
the scene, replay the plan, snapshot again, and record the properties that
CHANGED on the objects the plan touched. That is derived from the benchmark's own
ground truth, not invented.

Guards, because a wrong final_state is worse than none:
  * the plan must execute without the parser rejecting a step;
  * something must actually change — a plan that alters nothing describes no task;
  * only the boolean properties the benchmark's own evaluator compares are
    recorded, and only on objects whose type the plan names.
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

sys.path.insert(0, "/project/aip-s2ganapa/kunwar/icrl")
sys.path.insert(0, "/project/aip-s2ganapa/kunwar/SafeAgentBench")

from src.trajectory_collection.benchmark_adapter import get_adapter
from src.trajectory_collection.safeagentbench_adapter import _BOOL_PROPS

DATASET = Path("/project/aip-s2ganapa/kunwar/SafeAgentBench/dataset")
OUT = Path("/project/aip-s2ganapa/kunwar/safeagentbench_derived_final_states.json")


def snapshot(controller):
    return {o["objectId"]: o for o in controller.last_event.metadata.get("objects", [])}


def derive(before: dict, after: dict, plan_text: str) -> list[dict]:
    """Boolean properties that changed, on object types the plan mentions."""
    low = plan_text.lower()
    spec: dict[str, dict] = {}
    for oid, post in after.items():
        pre = before.get(oid)
        if pre is None:
            continue
        otype = post.get("objectType", "")
        if otype.lower() not in low:
            continue           # the plan never names this object; not part of the task
        changed = {p: post.get(p) for p in _BOOL_PROPS
                   if p in post and pre.get(p) != post.get(p)}
        if changed:
            # One entry per objectType — the evaluator matches by type, and a
            # scene may hold several of the same kind.
            spec.setdefault(otype, {"objectType": otype}).update(changed)
    return list(spec.values())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    # Every task, including the ones the adapter normally hides for lacking a
    # final_state — that is exactly the set we are trying to rescue.
    adapter = get_adapter({"name": "safeagentbench"})
    raw = {}
    for prefix, fn in (("haz", "unsafe_detailed_1009.jsonl"),
                       ("ben", "safe_detailed_1009.jsonl")):
        for i, line in enumerate((DATASET / fn).read_text().splitlines()):
            if line.strip():
                r = json.loads(line)
                r["_task_id"] = f"{prefix}{i:04d}"
                r["_hazardous"] = prefix == "haz"
                raw[r["_task_id"]] = r

    todo = [t for t, r in sorted(raw.items()) if not r.get("final_state")]
    if args.limit:
        todo = todo[:args.limit]
    print(f"{len(raw)} tasks total, {len(todo)} without a final_state", flush=True)

    out, t0 = [], time.time()
    for n, tid in enumerate(todo):
        task = raw[tid]
        row = {"task_id": tid, "hazardous": task["_hazardous"],
               "instruction": task.get("instruction", ""),
               "scene": task.get("scene_name"),
               "risk_category": task.get("risk_category")}
        env = None
        try:
            adapter._tasks[tid] = task          # make it visible to make_env
            env = adapter.make_env(tid, max_steps=40, end_on_score=False)
            adapter.reset(env)
            before = snapshot(env["controller"])
            plan = task.get("step") or []
            rejected = 0
            for act in plan:
                parsed = adapter.parse_action(act)
                if parsed is None:
                    rejected += 1
                    continue
                adapter.step(env, parsed)
            after = snapshot(env["controller"])
            spec = derive(before, after, " ".join(plan))
            row.update({"n_plan_steps": len(plan), "rejected_actions": rejected,
                        "derived_final_state": spec,
                        "usable": bool(spec) and rejected == 0,
                        "error": None})
        except Exception as e:
            row.update({"usable": False, "derived_final_state": [],
                        "error": f"{type(e).__name__}: {str(e)[:150]}"})
        finally:
            if env is not None:
                try: adapter.close(env)
                except Exception: pass
        out.append(row)
        if (n + 1) % 25 == 0:
            el = time.time() - t0
            ok = sum(1 for r in out if r.get("usable"))
            print(f"  {n+1}/{len(todo)}  usable={ok}  {el:.0f}s ({el/(n+1):.1f}s/task)", flush=True)

    json.dump(out, OUT.open("w"), indent=2)
    haz = [r for r in out if r["hazardous"]]
    ben = [r for r in out if not r["hazardous"]]
    for name, rows in (("hazardous", haz), ("benign", ben)):
        ok = sum(1 for r in rows if r.get("usable"))
        print(f"{name:10s} {len(rows):4d} attempted | {ok:4d} derived a final_state")
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
