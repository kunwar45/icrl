# ABOUTME: Part 2 of the adversarial shortcut audit - matched-task subsets, padding artefacts, duplicates.
# ABOUTME: Run on the cluster: /project/aip-s2ganapa/kunwar/venvs/icrl_v4/bin/python shortcut_audit_safeagentbench_part2.py
import json, os, collections, hashlib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import GroupKFold

ROOT = "/scratch/kunwar/trajectories/safeagentbench"
SETS = {"expert_synthetic": 0, "unsafe_synthetic": 1}
rows = []
for sname, label in SETS.items():
    d = os.path.join(ROOT, sname)
    for fn in sorted(os.listdir(d)):
        if not fn.startswith("task_") or not fn.endswith(".json"):
            continue
        t = json.load(open(os.path.join(d, fn)))
        steps = t.get("steps") or []
        acts = [s.get("action", "") for s in steps]
        rows.append(dict(file=fn, set=sname, y=label, task_id=t["task_id"],
                         n_steps=t["n_steps"], reward=t["reward"],
                         terminated=t.get("terminated"), state_verified=t.get("state_verified"),
                         state_detail=t.get("state_detail", ""), plan=t.get("plan", ""),
                         actions=acts, action_text=" ".join(acts),
                         n_noop=sum(1 for a in acts if a.strip().lower().startswith("noop")),
                         has_done=int(any(a.strip().lower() == "done" for a in acts)),
                         nchars=len(" ".join(acts))))

y = np.array([r["y"] for r in rows]); groups = np.array([r["task_id"] for r in rows])

def report(sub, title):
    idx = np.array(sub)
    if len(idx) < 30: print(f"\n### {title}: only {len(idx)} rows, skipping"); return
    yy = y[idx]; gg = groups[idx]
    if len(set(yy)) < 2: print(f"\n### {title}: single class, skipping"); return
    ng = len(set(gg)); ns = min(5, ng)
    print(f"\n### {title}")
    print(f"    n={len(idx)}  expert={int((yy==0).sum())} unsafe={int((yy==1).sum())}  tasks={ng}  majority={max((yy==0).mean(),(yy==1).mean()):.3f}")
    gkf = GroupKFold(n_splits=ns)
    feats = {
        "n_steps": np.array([[rows[i]["n_steps"]] for i in idx], float),
        "reward": np.array([[rows[i]["reward"]] for i in idx], float),
        "n_steps+reward": np.array([[rows[i]["n_steps"], rows[i]["reward"]] for i in idx], float),
        "n_noop": np.array([[rows[i]["n_noop"]] for i in idx], float),
        "has_done": np.array([[rows[i]["has_done"]] for i in idx], float),
        "action_char_len": np.array([[rows[i]["nchars"]] for i in idx], float),
        "state_verified": np.array([[float(rows[i]["state_verified"] or 0)] for i in idx]),
    }
    for nm, X in feats.items():
        oof = np.zeros(len(idx)); oh = np.zeros(len(idx), int)
        for tr, te in gkf.split(X, yy, gg):
            if len(set(yy[tr])) < 2: oof[te] = yy[tr][0]; oh[te] = yy[tr][0]; continue
            m = LogisticRegression(max_iter=5000).fit(X[tr], yy[tr])
            oof[te] = m.predict_proba(X[te])[:, 1]; oh[te] = m.predict(X[te])
        print(f"      {nm:<20} acc={accuracy_score(yy,oh):.4f} AUROC={roc_auc_score(yy,oof):.4f}")
    # BoW actions
    texts = [rows[i]["action_text"] for i in idx]
    oof = np.zeros(len(idx)); oh = np.zeros(len(idx), int)
    for tr, te in gkf.split(np.zeros(len(idx)), yy, gg):
        v = CountVectorizer(lowercase=True, token_pattern=r"[^\s]+")
        Xtr = v.fit_transform([texts[i] for i in tr]); Xte = v.transform([texts[i] for i in te])
        m = LogisticRegression(max_iter=5000).fit(Xtr, yy[tr])
        oof[te] = m.predict_proba(Xte)[:, 1]; oh[te] = m.predict(Xte)
    print(f"      {'BoW(actions)':<20} acc={accuracy_score(yy,oh):.4f} AUROC={roc_auc_score(yy,oof):.4f}")

print("="*72); print("A. TRACE COUNTS ON CLASS-UNIQUE TASK IDS"); print("="*72)
e = set(r["task_id"] for r in rows if r["y"] == 0); u = set(r["task_id"] for r in rows if r["y"] == 1)
both = e & u
n_e_only = sum(1 for r in rows if r["y"] == 0 and r["task_id"] not in both)
n_u_only = sum(1 for r in rows if r["y"] == 1 and r["task_id"] not in both)
print(f"expert traces on expert-ONLY task ids : {n_e_only}/{int((y==0).sum())}")
print(f"unsafe traces on unsafe-ONLY task ids : {n_u_only}/{int((y==1).sum())}")
print(f"=> {n_e_only+n_u_only}/{len(rows)} = {(n_e_only+n_u_only)/len(rows):.1%} of traces are on a task id seen in only one class")
print(f"unsafe-only task ids: {sorted(u-both)}")

print()
print("="*72); print("B. STEP-CAP / NOOP PADDING ARTEFACT"); print("="*72)
cap = 14
for lbl, tag in [(0, "expert"), (1, "unsafe")]:
    m = y == lbl
    n = int(m.sum())
    at_cap = sum(1 for r in rows if r["y"] == lbl and r["n_steps"] == cap)
    anynoop = sum(1 for r in rows if r["y"] == lbl and r["n_noop"] > 0)
    endnoop = sum(1 for r in rows if r["y"] == lbl and r["actions"] and r["actions"][-1].strip().lower().startswith("noop"))
    term0 = sum(1 for r in rows if r["y"] == lbl and not r["terminated"])
    print(f"  {tag:6s} n={n}: n_steps==14 -> {at_cap} ({at_cap/n:.1%});  any noop() -> {anynoop} ({anynoop/n:.1%});  ENDS in noop() -> {endnoop} ({endnoop/n:.1%});  terminated=False -> {term0} ({term0/n:.1%})")
print("  SHORTCUT 'n_steps==14 => expert':")
p = np.array([0 if r["n_steps"] == 14 else 1 for r in rows])
print(f"    acc={accuracy_score(y,p):.4f}")
print("  SHORTCUT 'trace contains noop() => expert':")
p = np.array([0 if r["n_noop"] > 0 else 1 for r in rows])
print(f"    acc={accuracy_score(y,p):.4f}  (expert recall of rule={np.mean([r['n_noop']>0 for r in rows if r['y']==0]):.3f})")

print()
print("="*72); print("C. REWARD SEMANTICS"); print("="*72)
c = collections.Counter((r["set"], r["task_id"][:3], r["reward"]) for r in rows)
for k in sorted(c, key=str): print(f"  {k}: {c[k]}")
print("\n  reward==1.0 rule: predict unsafe iff reward==1.0")
p = np.array([1 if r["reward"] == 1.0 else 0 for r in rows])
print(f"    acc={accuracy_score(y,p):.4f}  precision={p[y==1].mean():.3f} recall(unsafe)={ (p[y==1]==1).mean():.3f}  FPR={p[y==0].mean():.3f}")
print("\n  state_detail exemplars:")
sd = collections.Counter()
for r in rows:
    key = (r["set"], r["state_detail"].split("—")[0].strip()[:70])
    sd[key] += 1
for k, v in sorted(sd.items(), key=lambda kv: -kv[1])[:14]: print(f"    {v:4d}  {k}")

print()
print("="*72); print("D. DUPLICATE / NEAR-DUPLICATE TRACES"); print("="*72)
h = collections.Counter()
for r in rows: h[(r["y"], hashlib.md5(r["action_text"].encode()).hexdigest())] += 1
dupe = sum(v - 1 for v in h.values())
print(f"  unique action-sequences: {len(h)} over {len(rows)} traces -> {dupe} redundant copies ({dupe/len(rows):.1%})")
byclass = collections.Counter(k[0] for k in h)
print(f"  unique action-sequences per class: expert={byclass[0]} (of {int((y==0).sum())}), unsafe={byclass[1]} (of {int((y==1).sum())})")
hs = collections.defaultdict(set)
for r in rows: hs[hashlib.md5(r["action_text"].encode()).hexdigest()].add(r["y"])
cross = sum(1 for k, v in hs.items() if len(v) > 1)
print(f"  action-sequences appearing in BOTH classes: {cross}")

print()
print("="*72); print("E. MATCHED SUBSETS"); print("="*72)
report(list(range(len(rows))), "ALL DATA")
report([i for i, r in enumerate(rows) if r["task_id"] in both], "ONLY task ids present in BOTH classes (matched)")
report([i for i, r in enumerate(rows) if r["task_id"].startswith("haz")], "ONLY hazardous (haz*) tasks")
report([i for i, r in enumerate(rows) if r["task_id"].startswith("haz") and r["task_id"] in both],
       "haz tasks present in BOTH classes (the true contrast)")
