# ABOUTME: Part 3 of the shortcut audit - replicates the real encoder text and grouped-vs-random split inflation.
# ABOUTME: Run on the cluster: /project/aip-s2ganapa/kunwar/venvs/icrl_v4/bin/python shortcut_audit_safeagentbench_part3.py
import json, os, collections
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import GroupKFold, StratifiedKFold

ROOT = "/scratch/kunwar/trajectories/safeagentbench"
SETS = {"expert_synthetic": 0, "unsafe_synthetic": 1}


def traj_json_to_text(traj):
    """Verbatim copy of scripts/embed_trajectories.py::traj_json_to_text."""
    parts = []
    goal = traj.get("goal", "")
    if not goal and traj.get("steps"):
        goal = f"Task {traj.get('task_id', '')}"
    if goal:
        parts.append(f"[GOAL] {goal}")
    policies = traj.get("policies", [])
    if policies:
        ps = [f"({p.get('source','')}) {p.get('description','')}" for p in policies if p.get("description")]
        if ps:
            parts.append("[POLICIES] " + " | ".join(ps))
    for step in traj.get("steps", []):
        parts.append(f"[STEP {step.get('step_idx','?')}] [OBS] {step.get('observation','').strip()} "
                     f"[ACTION] {step.get('action','').strip()}")
    return "\n".join(parts)


rows = []
for sname, label in SETS.items():
    d = os.path.join(ROOT, sname)
    for fn in sorted(os.listdir(d)):
        if not fn.startswith("task_") or not fn.endswith(".json"):
            continue
        t = json.load(open(os.path.join(d, fn)))
        rows.append(dict(y=label, task_id=t["task_id"], enc_text=traj_json_to_text(t),
                         n_steps=t["n_steps"], reward=t["reward"]))

y = np.array([r["y"] for r in rows]); groups = np.array([r["task_id"] for r in rows])
texts = [r["enc_text"] for r in rows]
print(f"N={len(rows)} expert={int((y==0).sum())} unsafe={int((y==1).sum())} majority={max((y==0).mean(),(y==1).mean()):.4f}")
print("\nENCODER TEXT SAMPLE (first 300 chars):")
print(texts[0][:300].replace("\n", " | "))
print("!! note the literal task id in the [GOAL] line !!\n")


def run(texts, name, splitter, split_args, vec_kw=None):
    vec_kw = vec_kw or dict(lowercase=True, token_pattern=r"[^\s,]+", ngram_range=(1, 1))
    oof = np.zeros(len(y)); oh = np.zeros(len(y), int)
    for tr, te in splitter.split(*split_args):
        v = TfidfVectorizer(**vec_kw)
        Xtr = v.fit_transform([texts[i] for i in tr]); Xte = v.transform([texts[i] for i in te])
        m = LogisticRegression(max_iter=5000, C=5.0).fit(Xtr, y[tr])
        oof[te] = m.predict_proba(Xte)[:, 1]; oh[te] = m.predict(Xte)
    print(f"  {name:<56} acc={accuracy_score(y,oh):.4f} AUROC={roc_auc_score(y,oof):.4f}")


gkf = GroupKFold(n_splits=5)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
print("="*76)
print("F. FULL ENCODER TEXT (what the frozen-backbone encoder actually sees)")
print("="*76)
run(texts, "encoder text  | RANDOM split (ungrouped, the naive default)", skf, (np.zeros(len(y)), y))
run(texts, "encoder text  | GROUPED by task_id (honest)", gkf, (np.zeros(len(y)), y, groups))

stripped = [t.split("\n", 1)[1] if t.startswith("[GOAL]") and "\n" in t else t for t in texts]
run(stripped, "encoder text MINUS [GOAL] task-id line | RANDOM split", skf, (np.zeros(len(y)), y))
run(stripped, "encoder text MINUS [GOAL] task-id line | GROUPED", gkf, (np.zeros(len(y)), y, groups))

goal_only = ["[GOAL] Task " + r["task_id"] for r in rows]
run(goal_only, "TASK-ID LINE ALONE (no trajectory at all) | RANDOM split", skf, (np.zeros(len(y)), y))
run(goal_only, "TASK-ID LINE ALONE | GROUPED (should collapse)", gkf, (np.zeros(len(y)), y, groups))

print()
print("="*76)
print("G. THE THREE-FEATURE STRAWMAN (no text at all)")
print("="*76)
for nm, X in [("n_steps only", np.array([[r["n_steps"]] for r in rows], float)),
              ("reward only", np.array([[r["reward"]] for r in rows], float)),
              ("n_steps + reward", np.array([[r["n_steps"], r["reward"]] for r in rows], float))]:
    for sname, sp, ar in [("RANDOM", skf, (np.zeros(len(y)), y)), ("GROUPED", gkf, (np.zeros(len(y)), y, groups))]:
        oof = np.zeros(len(y)); oh = np.zeros(len(y), int)
        for tr, te in sp.split(*ar):
            m = LogisticRegression(max_iter=5000).fit(X[tr], y[tr])
            oof[te] = m.predict_proba(X[te])[:, 1]; oh[te] = m.predict(X[te])
        print(f"  {nm:<20} {sname:<8} acc={accuracy_score(y,oh):.4f} AUROC={roc_auc_score(y,oof):.4f}")

print()
print("="*76)
print("H. HAZ-ONLY SUBSET, ENCODER TEXT, GROUPED (the fairest possible test)")
print("="*76)
idx = [i for i, r in enumerate(rows) if r["task_id"].startswith("haz")]
yy = y[idx]; gg = groups[idx]; tt = [stripped[i] for i in idx]
g2 = GroupKFold(n_splits=5)
oof = np.zeros(len(idx)); oh = np.zeros(len(idx), int)
for tr, te in g2.split(np.zeros(len(idx)), yy, gg):
    v = TfidfVectorizer(lowercase=True, token_pattern=r"[^\s,]+")
    Xtr = v.fit_transform([tt[i] for i in tr]); Xte = v.transform([tt[i] for i in te])
    m = LogisticRegression(max_iter=5000, C=5.0).fit(Xtr, yy[tr])
    oof[te] = m.predict_proba(Xte)[:, 1]; oh[te] = m.predict(Xte)
print(f"  haz-only, no [GOAL] line, grouped: n={len(idx)} majority={max((yy==0).mean(),(yy==1).mean()):.4f} "
      f"acc={accuracy_score(yy,oh):.4f} AUROC={roc_auc_score(yy,oof):.4f}")

print()
print("="*76)
print("I. CROSS-CLASS TASK LEAKAGE UNDER make_demo_splits.py")
print("="*76)
e = set(r["task_id"] for r in rows if r["y"] == 0); u = set(r["task_id"] for r in rows if r["y"] == 1)
both = e & u
print(f"  make_demo_splits.py shuffles/splits the 'safe' and 'unsafe' label lists INDEPENDENTLY.")
print(f"  {len(both)} task ids appear in both classes ({sum(1 for r in rows if r['task_id'] in both)} traces,"
      f" {sum(1 for r in rows if r['task_id'] in both)/len(rows):.1%} of the data).")
print(f"  With a 0.3 held-out fraction and independent shuffles, ~{0.3*0.7+0.7*0.3:.0%} of those tasks land on"
      f" OPPOSITE sides of the two splits -> the held-out set is contaminated by same-task traces.")
