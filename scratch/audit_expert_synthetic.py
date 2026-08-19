# ABOUTME: One-off audit of generated SafeAgentBench expert_synthetic traces on the cluster.
# ABOUTME: Run with /project/aip-s2ganapa/kunwar/venvs/icrl_v4/bin/python audit_expert_synthetic.py
import json, glob, os, re
from collections import Counter, defaultdict

D = "/scratch/kunwar/trajectories/safeagentbench/expert_synthetic"
files = sorted(glob.glob(os.path.join(D, "task_*_trace_*.json")))
print("N FILES", len(files))

recs = []
for f in files:
    d = json.load(open(f))
    acts = [s.get("action") for s in d.get("steps", [])]
    recs.append(dict(
        f=os.path.basename(f), tid=d.get("task_id", ""), acts=acts,
        n_steps=d.get("n_steps"), reward=d.get("reward"), cup=d.get("cup"),
        sv=d.get("state_verified"), sd=d.get("state_detail", ""),
        term=d.get("terminated"), fin=d.get("finished_deliberately"),
        goal=d.get("ended_on_goal_confirmed"), plan=d.get("plan", ""),
        sr=d.get("safety_report", []), psv=d.get("page_scraped_violations", []),
    ))

haz = [r for r in recs if r["tid"].startswith("haz")]
ben = [r for r in recs if r["tid"].startswith("ben")]
other = [r for r in recs if not r["tid"].startswith(("haz", "ben"))]
print("haz", len(haz), "ben", len(ben), "other", len(other), [r["tid"] for r in other][:5])
print("unique haz tasks", len(set(r["tid"] for r in haz)), "unique ben tasks", len(set(r["tid"] for r in ben)))

# ---------- overall field distributions ----------
def dist(rs, key, label, n=8):
    print(f"  {label}:", Counter(str(r[key]) for r in rs).most_common(n))

for name, rs in (("HAZ", haz), ("BEN", ben)):
    print(f"\n=== {name} fields ===")
    for k in ("sv", "reward", "cup", "term", "fin", "goal"):
        dist(rs, k, k)
    print("  state_detail patterns:", Counter(re.sub(r"\d+\.\d+", "N", r["sd"])[:110] for r in rs).most_common(8))

# ---------- helpers ----------
def noop_count(acts):
    return sum(1 for a in acts if a is not None and a.strip().lower().startswith("noop"))

def max_repeat(acts):
    return max(Counter(acts).values()) if acts else 0

def has_cycle(acts):
    """True if some contiguous block of length k (1..4) repeats back-to-back >=2 times."""
    for k in range(1, 5):
        for i in range(0, len(acts) - 2 * k + 1):
            if acts[i:i + k] == acts[i + k:i + 2 * k]:
                return True
    return False

def last_is_done(acts):
    return bool(acts) and acts[-1] is not None and acts[-1].strip().lower().rstrip("()") == "done"

def classify(r):
    a = r["acts"]
    nc = noop_count(a)
    mr = max_repeat(a)
    if nc >= 2 or mr >= 3 or has_cycle(a):
        return "loop_garbage"
    if nc == 0 and mr == 1 and len(a) <= 6:
        return "clean"
    return "middle"

for r in recs:
    r["noop"] = noop_count(r["acts"])
    r["maxrep"] = max_repeat(r["acts"])
    r["cycle"] = has_cycle(r["acts"])
    r["done"] = last_is_done(r["acts"])
    r["cat"] = classify(r)

# ---------- declined-hazard subset ----------
declined = [r for r in haz if "declined" in r["sd"] or ("hazardous end state" in r["sd"] and "not reached" in r["sd"])]
not_declined = [r for r in haz if r not in declined]
print("\n=== HAZ split ===")
print("declined-hazard:", len(declined), " other haz:", len(not_declined))
print("  non-declined state_detail:", Counter(re.sub(r"\d+\.\d+", "N", r["sd"])[:110] for r in not_declined).most_common(5))

def report(rs, label):
    n = len(rs)
    if not n:
        print(f"\n--- {label}: EMPTY ---"); return
    print(f"\n--- {label}: n={n} ---")
    c = Counter(r["cat"] for r in rs)
    for k in ("clean", "middle", "loop_garbage"):
        print(f"   {k:13s} {c[k]:4d}  {100*c[k]/n:5.1f}%")
    print("   noop-count histogram:", sorted(Counter(r["noop"] for r in rs).items()))
    print("   traces with >=1 noop:", sum(1 for r in rs if r["noop"] >= 1), f"({100*sum(1 for r in rs if r['noop']>=1)/n:.1f}%)")
    print("   traces with a back-to-back cycle:", sum(1 for r in rs if r["cycle"]), f"({100*sum(1 for r in rs if r['cycle'])/n:.1f}%)")
    print("   last action == done:", sum(1 for r in rs if r["done"]), f"({100*sum(1 for r in rs if r['done'])/n:.1f}%)")
    print("   'done' anywhere in actions:", sum(1 for r in rs if any((a or '').strip().lower().rstrip('()')=='done' for a in r["acts"])))
    tot_a = sum(len(r["acts"]) for r in rs); tot_n = sum(r["noop"] for r in rs)
    print(f"   actions={tot_a} noop={tot_n} ({100*tot_n/max(tot_a,1):.1f}% of actions)")
    for cat in ("clean", "middle", "loop_garbage"):
        sub = [r["n_steps"] for r in rs if r["cat"] == cat]
        if sub:
            sub_s = sorted(sub)
            print(f"   n_steps[{cat}] n={len(sub)} min={sub_s[0]} p25={sub_s[len(sub_s)//4]} med={sub_s[len(sub_s)//2]} p75={sub_s[3*len(sub_s)//4]} max={sub_s[-1]} mean={sum(sub)/len(sub):.1f}")
            print(f"      hist: {sorted(Counter(sub).items())}")

report(declined, "DECLINED-HAZARD")
report(not_declined, "HAZ NOT-DECLINED")
report(ben, "BENIGN (all)")

ben_done = [r for r in ben if r["sv"] is True]
ben_notdone = [r for r in ben if r["sv"] is not True]
report(ben_done, "BENIGN state_verified=True (completed)")
report(ben_notdone, "BENIGN state_verified!=True")

# ---------- top action sequences ----------
print("\n=== TOP 10 ACTION SEQUENCES: declined-hazard ===")
for seq, k in Counter(tuple(r["acts"]) for r in declined).most_common(10):
    print(f"  {k:4d}  len={len(seq)}  {list(seq)}")

print("\n=== TOP 10 ACTION SEQUENCES: benign completed ===")
for seq, k in Counter(tuple(r["acts"]) for r in ben_done).most_common(10):
    print(f"  {k:4d}  len={len(seq)}  {list(seq)}")

print("\n=== TOP 20 individual actions (whole set) ===")
allacts = Counter(a for r in recs for a in r["acts"])
tot = sum(allacts.values())
print("  total actions:", tot)
for a, k in allacts.most_common(20):
    print(f"  {k:5d} {100*k/tot:5.1f}%  {a!r}")
nn = sum(k for a, k in allacts.items() if (a or '').strip().lower().startswith('noop'))
print(f"  NOOP TOTAL {nn} = {100*nn/tot:.1f}% of all actions")

# ---------- does the plan attempt the hazard? ----------
print("\n=== HAZ plans: do they plan to comply? ===")
refuse_kw = ("refus", "decline", "not comply", "cannot", "can't", "will not", "won't", "unsafe", "harmful", "do nothing", "abort")
comply = [r for r in haz if not any(k in r["plan"].lower() for k in refuse_kw)]
print(f"  haz traces whose plan shows NO refusal language: {len(comply)}/{len(haz)} ({100*len(comply)/max(len(haz),1):.1f}%)")
print("  sample haz plans:")
for r in haz[:5]:
    print("   ---", r["tid"], "|", r["plan"].replace("\n", " / ")[:160])

# ---------- representative traces ----------
print("\n=== REPRESENTATIVE ===")
for label, pool in (("declined clean", [r for r in declined if r["cat"] == "clean"]),
                    ("declined loop", [r for r in declined if r["cat"] == "loop_garbage"]),
                    ("benign clean", [r for r in ben_done if r["cat"] == "clean"]),
                    ("benign loop", [r for r in ben_done if r["cat"] == "loop_garbage"])):
    print(f"\n  [{label}] pool={len(pool)}")
    for r in pool[:3]:
        print(f"   {r['f']} n_steps={r['n_steps']} sv={r['sv']} reward={r['reward']}")
        print(f"     {r['acts']}")

# ---------- usable estimate ----------
print("\n=== USABILITY ===")
usable_haz = [r for r in declined if r["cat"] == "clean"]
usable_ben = [r for r in ben_done if r["cat"] == "clean"]
print(f"  clean declined-hazard: {len(usable_haz)}")
print(f"  clean benign-completed: {len(usable_ben)}")
print(f"  TOTAL clean: {len(usable_haz)+len(usable_ben)} / {len(recs)} = {100*(len(usable_haz)+len(usable_ben))/len(recs):.1f}%")
lenient_h = [r for r in declined if r["cat"] in ("clean", "middle")]
lenient_b = [r for r in ben_done if r["cat"] in ("clean", "middle")]
print(f"  lenient (clean+middle): {len(lenient_h)+len(lenient_b)} / {len(recs)} = {100*(len(lenient_h)+len(lenient_b))/len(recs):.1f}%")
