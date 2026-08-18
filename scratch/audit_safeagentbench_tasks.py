#!/usr/bin/env python3
# ABOUTME: Replays every SafeAgentBench task's own reference plan in AI2-THOR to find which are usable
# ABOUTME: Run on a GPU node: python scratch/audit_safeagentbench_tasks.py (writes the audit JSON)
import json, sys, time, traceback
sys.path.insert(0, "/project/aip-s2ganapa/kunwar/icrl")
sys.path.insert(0, "/project/aip-s2ganapa/kunwar/SafeAgentBench")
from src.trajectory_collection.benchmark_adapter import get_adapter

OUT = "/project/aip-s2ganapa/kunwar/safeagentbench_task_audit.json"
a = get_adapter({"name": "safeagentbench"})
ids = a.task_ids()
print("auditing %d tasks" % len(ids), flush=True)

rows, t_start = [], time.time()
for n, tid in enumerate(ids):
    task = a.task(tid)
    row = {"task_id": tid, "hazardous": task["_hazardous"],
           "instruction": task["instruction"], "scene": task["scene_name"],
           "risk_category": task.get("risk_category"),
           "n_ref_steps": len(task.get("step") or [])}
    env = None
    try:
        env = a.make_env(tid, max_steps=40, end_on_score=False)
        a.reset(env)
        row["match_at_start"] = round(env["match_at_start"], 3)
        info = {}
        for act in (task.get("step") or []):
            parsed = a.parse_action(act) or act
            _obs, reward, _t, _tr, info = a.step(env, parsed)
        row["match_after_plan"] = round(info.get("state_match", 0.0), 3)
        row["reference_plan_reaches_target"] = bool(info.get("reached"))
        row["satisfied_before"] = bool(info.get("satisfied_before"))
        # Usable iff the target is not pre-satisfied AND the reference plan can
        # actually produce it. A task failing either cannot yield an unsafe
        # demonstration (the hazard never happens) or an attributable safe one.
        row["usable"] = bool(info.get("reached")) and not bool(info.get("satisfied_before"))
        row["error"] = None
    except Exception as e:
        row.update({"usable": False, "error": f"{type(e).__name__}: {str(e)[:160]}"})
    finally:
        if env is not None:
            try: a.close(env)
            except Exception: pass
    rows.append(row)
    if (n + 1) % 20 == 0:
        el = time.time() - t_start
        print("  %d/%d  %.0fs elapsed  (%.1fs/task)" % (n + 1, len(ids), el, el / (n + 1)), flush=True)

json.dump(rows, open(OUT, "w"), indent=2)

haz = [r for r in rows if r["hazardous"]]
ben = [r for r in rows if not r["hazardous"]]
def sm(rs, k): return sum(1 for r in rs if r.get(k))
print("\n" + "=" * 60)
print("hazardous : %3d total | %3d usable | %3d pre-satisfied | %3d plan-fails | %3d errors"
      % (len(haz), sm(haz, "usable"), sm(haz, "satisfied_before"),
         sum(1 for r in haz if not r.get("reference_plan_reaches_target") and not r.get("error")),
         sm(haz, "error")))
print("benign    : %3d total | %3d usable | %3d pre-satisfied | %3d plan-fails | %3d errors"
      % (len(ben), sm(ben, "usable"), sm(ben, "satisfied_before"),
         sum(1 for r in ben if not r.get("reference_plan_reaches_target") and not r.get("error")),
         sm(ben, "error")))
import collections
cats = collections.Counter(r["risk_category"] for r in haz if r.get("usable"))
print("\nusable hazardous by category (top 12):")
for k, v in cats.most_common(12):
    print("   %3d  %s" % (v, k))
print("\nwrote", OUT)
