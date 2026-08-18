# ABOUTME: Checks whether the contrast classes separate on anything but safety
# ABOUTME: Run: python scratch/check_contrast_confounds.py <expert_dir> <unsafe_dir>
"""
C_theta learns whatever systematically distinguishes the two classes. This looks
for the cheap separators it could latch onto INSTEAD of safety, before a dataset
is generated at scale.

Fails loudly on the ones that would invalidate the set:
  * trajectory length — a declined hazard can be one action while an executed one
    runs several, so "short = safe" would score well and mean nothing;
  * task reward — if expert is uniformly 0.0 and unsafe uniformly 1.0, reward
    alone separates them;
  * task coverage — a task appearing in only one class lets the head separate by
    task identity.
"""
import json, sys, statistics as st
from collections import Counter
from pathlib import Path

def load(d):
    out = []
    for p in sorted(Path(d).glob("task_*_trace_*.json")):
        t = json.loads(p.read_text())
        out.append(t)
    return out

expert, unsafe = load(sys.argv[1]), load(sys.argv[2])
print(f"expert {len(expert)} traces | unsafe {len(unsafe)} traces\n")

def kind(t):
    return "declined-hazard" if str(t["task_id"]).startswith("haz") else "completed-benign"

def dist(name, xs):
    if not xs: return f"{name:26s} (none)"
    return (f"{name:26s} n={len(xs):3d}  min={min(xs):5.2f}  med={st.median(xs):5.2f}  "
            f"mean={sum(xs)/len(xs):5.2f}  max={max(xs):5.2f}")

print("── 1. TRAJECTORY LENGTH ─────────────────────────────────────────")
e_steps = [t["n_steps"] for t in expert]
u_steps = [t["n_steps"] for t in unsafe]
e_haz  = [t["n_steps"] for t in expert if kind(t) == "declined-hazard"]
e_ben  = [t["n_steps"] for t in expert if kind(t) == "completed-benign"]
print(dist("expert (all)", e_steps))
print(dist("  declined-hazard", e_haz))
print(dist("  completed-benign", e_ben))
print(dist("unsafe", u_steps))
overlap = (min(e_steps) <= max(u_steps) and min(u_steps) <= max(e_steps)) if e_steps and u_steps else False
sep = "OVERLAP (good)" if overlap else "DISJOINT — length separates the classes!"
print(f"  ranges: expert [{min(e_steps)},{max(e_steps)}] unsafe [{min(u_steps)},{max(u_steps)}] -> {sep}")

print("\n── 2. TASK REWARD ───────────────────────────────────────────────")
e_r = [t["reward"] for t in expert]; u_r = [t["reward"] for t in unsafe]
print(dist("expert", e_r)); print(dist("unsafe", u_r))
print(f"  expert reward values : {dict(Counter(e_r))}")
print(f"  unsafe reward values : {dict(Counter(u_r))}")
straddles = len(set(e_r)) > 1
print("  expert straddles 0.0 and 1.0 -> reward alone cannot separate (good)" if straddles
      else "  expert is single-valued -> REWARD SEPARATES THE CLASSES!")

print("\n── 3. EXPERT CLASS COMPOSITION ──────────────────────────────────")
comp = Counter(kind(t) for t in expert)
print(f"  {dict(comp)}")
if comp.get("completed-benign", 0) == 0:
    print("  NO benign completions -> expert class is refusals only; C_theta can "
          "learn 'acting = unsafe'!")
else:
    ratio = comp.get("declined-hazard", 0) / max(1, comp["completed-benign"])
    print(f"  decline:completion ratio {ratio:.1f}:1")

print("\n── 4. TASK COVERAGE ─────────────────────────────────────────────")
e_tasks = {t["task_id"] for t in expert}; u_tasks = {t["task_id"] for t in unsafe}
print(f"  expert tasks {len(e_tasks)} | unsafe tasks {len(u_tasks)} | shared {len(e_tasks & u_tasks)}")
only_e = sorted(t for t in e_tasks - u_tasks if str(t).startswith("haz"))
if only_e:
    print(f"  hazardous tasks with expert but no unsafe traces: {only_e}")
    print("  (benign tasks are expected to be expert-only; hazardous ones are not)")

print("\n── 5. SAMPLE TRACES ─────────────────────────────────────────────")
for label, traces in (("EXPERT declined", [t for t in expert if kind(t)=='declined-hazard']),
                      ("EXPERT benign", [t for t in expert if kind(t)=='completed-benign']),
                      ("UNSAFE", unsafe)):
    if not traces: continue
    t = traces[0]
    print(f"\n{label}: {t['task_id']}  reward={t['reward']}  steps={t['n_steps']}  "
          f"verified={t.get('state_verified')}")
    print(f"  actions: {[s['action'] for s in t['steps']]}")
    print(f"  detail : {t.get('state_detail','')[:110]}")
