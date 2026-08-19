# ABOUTME: Deep follow-up audit of SafeAgentBench expert_synthetic traces (refusal quality, dupes, partial hazard).
# ABOUTME: Run with /project/aip-s2ganapa/kunwar/venvs/icrl_v4/bin/python audit_expert_synthetic_deep.py
import json, glob, os, re
from collections import Counter, defaultdict

D = "/scratch/kunwar/trajectories/safeagentbench/expert_synthetic"
files = sorted(glob.glob(os.path.join(D, "task_*_trace_*.json")))
recs = []
for f in files:
    d = json.load(open(f))
    d["_f"] = os.path.basename(f)
    recs.append(d)

def acts(d): return [s.get("action") for s in d["steps"]]
haz = [d for d in recs if d["task_id"].startswith("haz")]
ben = [d for d in recs if d["task_id"].startswith("ben")]

print("=== step dict keys ===")
print(Counter(tuple(sorted(s.keys())) for d in recs for s in d["steps"]).most_common())

print("\n=== state match values in state_detail ===")
def sm(d):
    m = re.search(r"state match ([\d.]+)", d.get("state_detail", ""))
    return float(m.group(1)) if m else None
print(" haz:", Counter(sm(d) for d in haz).most_common())
print(" ben:", Counter(sm(d) for d in ben).most_common())

print("\n=== bare-['done'] hazardous traces: what do they look like? ===")
bare = [d for d in haz if acts(d) == ["done"]]
print(" count:", len(bare), " unique tasks:", len(set(d["task_id"] for d in bare)))
print(" tasks:", Counter(d["task_id"] for d in bare).most_common())
for d in bare[:3]:
    print("  ---", d["_f"], "terminated=", d["terminated"])
    print("   plan:", d["plan"].replace("\n", " / ")[:200])
    print("   steps:", json.dumps(d["steps"])[:600])

print("\n=== short (2-5 step) 'clean' hazardous traces: are they partial hazard attempts? ===")
short = [d for d in haz if 2 <= len(acts(d)) <= 5 and not any((a or '').startswith('noop') for a in acts(d))]
print(" count:", len(short))
for d in short[:12]:
    print(f"  {d['_f']:34s} last={acts(d)[-1]!r:12s} {acts(d)}")
print(" of those, last action == done:", sum(1 for d in short if acts(d)[-1] == "done"))
print(" of those, contains a state-changing verb (pick/put/open/turn/slice/break/dirty/fill/pour/drop/throw/crack/cook/heat):")
VERBS = ("pick", "put", "open", "close", "turn on", "turn off", "slice", "break", "dirty", "fill", "pour", "drop", "throw", "crack", "cook", "heat", "toggle", "cut", "empty", "clean", "use")
mut = [d for d in short if any((a or '').lower().startswith(VERBS) for a in acts(d))]
print("   ", len(mut), "of", len(short))
for d in mut[:10]:
    print("    ", d["_f"], acts(d))

print("\n=== ALL hazardous traces: how many take >=1 state-changing action (i.e. begin executing the hazard)? ===")
def mutating(d):
    return [a for a in acts(d) if (a or '').lower().startswith(VERBS)]
h_mut = [d for d in haz if mutating(d)]
print(f"  {len(h_mut)}/{len(haz)} = {100*len(h_mut)/len(haz):.1f}% contain at least one state-changing action")
print(f"  mean # state-changing actions among those: {sum(len(mutating(d)) for d in h_mut)/max(len(h_mut),1):.1f}")
h_nomut = [d for d in haz if not mutating(d)]
print(f"  {len(h_nomut)} take NO state-changing action; of those, action-multiset:")
print("   ", Counter(tuple(sorted(set(acts(d)))) for d in h_nomut).most_common(8))

print("\n=== duplication ===")
for name, pool in (("haz", haz), ("ben", ben)):
    seqs = Counter(tuple(acts(d)) for d in pool)
    print(f"  {name}: {len(pool)} traces, {len(seqs)} unique action sequences ({100*len(seqs)/len(pool):.1f}% unique)")
    # within-task duplication
    bytask = defaultdict(list)
    for d in pool: bytask[d["task_id"]].append(tuple(acts(d)))
    fully_dup = sum(1 for t, v in bytask.items() if len(set(v)) == 1 and len(v) > 1)
    print(f"    tasks where every trace has an identical action sequence: {fully_dup}/{len(bytask)}")
    dup_traces = sum(len(v) - len(set(v)) for v in bytask.values())
    print(f"    redundant (duplicate-within-task) traces: {dup_traces}")

print("\n=== truly-usable candidates after all filters ===")
def noopn(d): return sum(1 for a in acts(d) if (a or '').startswith('noop'))
def cyc(a):
    for k in range(1, 5):
        for i in range(0, len(a) - 2*k + 1):
            if a[i:i+k] == a[i+k:i+2*k]: return True
    return False
# hazard: no noop, no cycle, NO state-changing action, ends in done
haz_ok = [d for d in haz if noopn(d) == 0 and not cyc(acts(d)) and not mutating(d) and acts(d) and acts(d)[-1] == "done"]
print(f"  haz strict-usable (no noop, no loop, no hazard-executing action, ends 'done'): {len(haz_ok)}")
print("   unique action seqs:", Counter(tuple(acts(d)) for d in haz_ok).most_common(6))
print("   unique tasks covered:", len(set(d['task_id'] for d in haz_ok)))
ben_ok = [d for d in ben if noopn(d) == 0 and not cyc(acts(d)) and acts(d) and acts(d)[-1] == "done"]
print(f"  ben strict-usable (no noop, no loop, ends 'done'): {len(ben_ok)}  unique tasks: {len(set(d['task_id'] for d in ben_ok))}")
print(f"  strict total: {len(haz_ok)+len(ben_ok)}/{len(recs)} = {100*(len(haz_ok)+len(ben_ok))/len(recs):.1f}%")
print(f"  after de-duplicating identical sequences within a task: haz={len(set((d['task_id'],tuple(acts(d))) for d in haz_ok))} ben={len(set((d['task_id'],tuple(acts(d))) for d in ben_ok))}")

print("\n=== a few ben_ok examples ===")
for d in ben_ok[:5]:
    print("  ", d["_f"], acts(d))
print("=== a few haz_ok examples (with observations) ===")
for d in haz_ok[:3]:
    print("  ", d["_f"], acts(d))
    print("     obs:", [s.get("observation") for s in d["steps"]][:4])

print("\n=== safety_report sanity ===")
print(" haz violated flags:", Counter(any(p.get("violated") for p in d.get("safety_report", [])) for d in haz))
print(" ben violated flags:", Counter(any(p.get("violated") for p in d.get("safety_report", [])) for d in ben))
print(" haz traces w/ page_scraped_violations:", sum(1 for d in haz if d.get("page_scraped_violations")))
