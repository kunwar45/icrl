# ABOUTME: Tests whether balancing the expert class 1:1 breaks the step-count shortcut
# ABOUTME: Run: python scratch/test_balanced_baseline.py <split_dir>
import json, sys, random, statistics as st
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gate_baselines import auroc, load  # noqa

split = Path(sys.argv[1])
safe = load(split / "eval" / "safe_held_out.jsonl")
unsafe = load(split / "eval" / "unsafe_held_out.jsonl")
steps = lambda t: t.get("n_steps") or len(t.get("steps") or [])
is_haz = lambda t: str(t.get("task_instance_id") or t.get("task_id") or "").startswith("haz")

dec = [t for t in safe if is_haz(t)]
ben = [t for t in safe if not is_haz(t)]
print(f"held-out safe: {len(dec)} declined-hazard, {len(ben)} completed-benign; unsafe {len(unsafe)}")
print(f"  declined steps med={st.median([steps(t) for t in dec]) if dec else 0}")
print(f"  benign   steps med={st.median([steps(t) for t in ben]) if ben else 0}")
print(f"  unsafe   steps med={st.median([steps(t) for t in unsafe])}")

def bl(safe_set, tag):
    rows = [(t, 0) for t in safe_set] + [(t, 1) for t in unsafe]
    a = auroc([steps(t) for t, _ in rows], [l for _, l in rows])
    print(f"  {tag:34s} n_safe={len(safe_set):3d}  n_steps AUROC = {a:.4f}")

print("\nSTEP-COUNT BASELINE under different expert compositions:")
bl(safe, "as generated (1.6:1 decline:benign)")
rng = random.Random(0)
if dec and ben:
    k = min(len(dec), len(ben))
    bl(rng.sample(dec, k) + rng.sample(ben, k), "balanced 1:1")
    bl(ben, "benign completions only (0 declines)")
