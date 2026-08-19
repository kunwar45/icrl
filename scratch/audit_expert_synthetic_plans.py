# ABOUTME: Third-pass audit: plan quality, partial-hazard state match, and category cross-tabs.
# ABOUTME: Run with /project/aip-s2ganapa/kunwar/venvs/icrl_v4/bin/python audit_expert_synthetic_plans.py
import json, glob, os, re
from collections import Counter, defaultdict

D = "/scratch/kunwar/trajectories/safeagentbench/expert_synthetic"
recs = []
for f in sorted(glob.glob(os.path.join(D, "task_*_trace_*.json"))):
    d = json.load(open(f)); d["_f"] = os.path.basename(f); recs.append(d)
def acts(d): return [s.get("action") for s in d["steps"]]
haz = [d for d in recs if d["task_id"].startswith("haz")]
ben = [d for d in recs if d["task_id"].startswith("ben")]
def sm(d):
    m = re.search(r"state match ([\d.]+)", d.get("state_detail", ""))
    return float(m.group(1)) if m else None
def noopn(d): return sum(1 for a in acts(d) if (a or '').startswith('noop'))
def cyc(a):
    for k in range(1,5):
        for i in range(0, len(a)-2*k+1):
            if a[i:i+k]==a[i+k:i+2*k]: return True
    return False
def cat(d):
    a=acts(d)
    if noopn(d)>=2 or (Counter(a).most_common(1)[0][1] if a else 0)>=3 or cyc(a): return "loop_garbage"
    if noopn(d)==0 and len(set(a))==len(a) and len(a)<=6: return "clean"
    return "middle"

print("=== HAZ: state match (how far toward the HAZARD the 'expert' got) x category ===")
tab = defaultdict(Counter)
for d in haz: tab[sm(d)][cat(d)] += 1
for k in sorted(tab, key=lambda x: -x):
    print(f"  state_match={k:<5} total={sum(tab[k].values()):4d}  {dict(tab[k])}")
partial = [d for d in haz if (sm(d) or 0) > 0]
print(f"  HAZ traces with PARTIAL hazard progress (>0): {len(partial)}/{len(haz)} = {100*len(partial)/len(haz):.1f}%")
print("  ...of those, how many are 'clean' by shape:", sum(1 for d in partial if cat(d)=="clean"))
print("  clean+partial examples:")
for d in [x for x in partial if cat(x)=="clean"][:8]:
    print(f"    {d['_f']} sm={sm(d)} {acts(d)}")

print("\n=== HAZ plans ===")
plans = Counter(d["plan"].strip() for d in haz)
print("  unique plans:", len(plans))
print("  plan == 'done' exactly:", sum(1 for d in haz if d["plan"].strip().lower() in ("done","done.")))
short_plan = [d for d in haz if len(d["plan"].strip().splitlines()) <= 1]
print("  single-line plans:", len(short_plan), Counter(d["plan"].strip()[:40] for d in short_plan).most_common(6))
print("  plans mentioning refusal/safety:", sum(1 for d in haz if re.search(r"refus|declin|unsafe|danger|harm|should not|not safe|cannot", d["plan"], re.I)))
print("  sample multi-line haz plans (first 6 distinct):")
seen=set()
for d in haz:
    p=d["plan"].strip()
    if p in seen or len(p.splitlines())<2: continue
    seen.add(p); print("   *", d["task_id"], "|", p.replace("\n"," / ")[:170])
    if len(seen)>=6: break

print("\n=== BEN: same shape check ===")
print("  cat:", Counter(cat(d) for d in ben))
print("  ben plans single-line:", sum(1 for d in ben if len(d['plan'].strip().splitlines())<=1))
print("  ben loop traces reward:", Counter(d['reward'] for d in ben if cat(d)=='loop_garbage'))

print("\n=== per-category final action ===")
for name, pool in (("haz", haz), ("ben", ben)):
    for c in ("clean","middle","loop_garbage"):
        sub=[d for d in pool if cat(d)==c]
        if sub: print(f"  {name}/{c}: n={len(sub)} last-action-done={sum(1 for d in sub if acts(d) and acts(d)[-1]=='done')} last-action-noop={sum(1 for d in sub if acts(d) and (acts(d)[-1] or '').startswith('noop'))}")

print("\n=== max n_steps / truncation ===")
print("  n_steps hist all:", sorted(Counter(d["n_steps"] for d in recs).items()))
print("  traces at n_steps==14:", sum(1 for d in recs if d["n_steps"]==14), "-> hit step cap")
print("  terminated flag x n_steps==14:", Counter((d['terminated'], d['n_steps']==14) for d in recs))

print("\n=== manifest / summary ===")
mp = os.path.join(D, "manifest.json")
if os.path.exists(mp):
    m = json.load(open(mp))
    s = json.dumps(m)[:1500]
    print(" manifest keys/head:", s)
