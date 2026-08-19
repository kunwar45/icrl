# ABOUTME: Adversarial shortcut audit of the SafeAgentBench expert/unsafe contrast dataset.
# ABOUTME: Run on the cluster: /project/aip-s2ganapa/kunwar/venvs/icrl_v4/bin/python shortcut_audit_safeagentbench.py
import json, os, re, sys, collections
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import GroupKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier

ROOT = "/scratch/kunwar/trajectories/safeagentbench"
SETS = {"expert_synthetic": 0, "unsafe_synthetic": 1}

rows = []
for sname, label in SETS.items():
    d = os.path.join(ROOT, sname)
    for fn in sorted(os.listdir(d)):
        if not fn.startswith("task_") or not fn.endswith(".json"):
            continue
        with open(os.path.join(d, fn)) as f:
            t = json.load(f)
        steps = t.get("steps") or []
        acts = [s.get("action", "") for s in steps]
        obs = [s.get("observation", "") for s in steps]
        rows.append(dict(
            file=fn, set=sname, y=label,
            task_id=t.get("task_id"),
            n_steps=t.get("n_steps"),
            reward=t.get("reward"),
            cup=t.get("cup"),
            terminated=t.get("terminated"),
            finished_deliberately=t.get("finished_deliberately"),
            ended_on_goal_confirmed=t.get("ended_on_goal_confirmed"),
            state_verified=t.get("state_verified"),
            plan_revisions=t.get("plan_revisions"),
            n_safety=len(t.get("safety_report") or []),
            n_scraped=len(t.get("page_scraped_violations") or []),
            plan=t.get("plan") or "",
            actions=acts,
            action_text=" ".join(acts),
            obs_text=" ".join(obs),
            len_steps=len(steps),
        ))

y = np.array([r["y"] for r in rows])
groups = np.array([r["task_id"] for r in rows])
print(f"N={len(rows)}  expert={int((y==0).sum())}  unsafe={int((y==1).sum())}")
print(f"majority-class baseline acc = {max((y==0).mean(),(y==1).mean()):.4f}")
print()

# ---------------- 4. Task-id overlap ----------------
print("="*70)
print("[4] TASK-ID OVERLAP")
print("="*70)
tid_by_set = {s: set(r["task_id"] for r in rows if r["set"] == s) for s in SETS}
e, u = tid_by_set["expert_synthetic"], tid_by_set["unsafe_synthetic"]
both = e & u
print(f"unique task ids: expert={len(e)}  unsafe={len(u)}  in BOTH={len(both)}")
print(f"expert-only={len(e-u)}  unsafe-only={len(u-e)}")
n_in_both = sum(1 for r in rows if r["task_id"] in both)
print(f"traces whose task id appears in BOTH classes: {n_in_both}/{len(rows)} = {n_in_both/len(rows):.1%}")
print(f"traces whose task id is UNIQUE to its class : {len(rows)-n_in_both}/{len(rows)} = {1-n_in_both/len(rows):.1%}")
pref = collections.Counter((r["set"], r["task_id"][:3]) for r in rows)
print("\ntraces by (set, task-id prefix):")
for k, v in sorted(pref.items()):
    print(f"  {k}: {v}")
tpref = collections.Counter((r["task_id"][:3], r["y"]) for r in rows)
print("\nPREFIX SHORTCUT: predict unsafe iff task_id startswith 'haz'")
pred = np.array([1 if r["task_id"].startswith("haz") else 0 for r in rows])
print(f"  accuracy = {accuracy_score(y,pred):.4f}   (AUROC = {roc_auc_score(y,pred):.4f})")
print()

# ---------------- grouped CV harness ----------------
uniq_groups = sorted(set(groups))
n_splits = min(5, len(uniq_groups))
gkf = GroupKFold(n_splits=n_splits)

def grouped_eval(X, name, model=None, verbose_tokens=None):
    """Group-by-task CV. Returns pooled out-of-fold AUROC + accuracy."""
    oof_p = np.zeros(len(y)); oof_h = np.zeros(len(y), dtype=int)
    for tr, te in gkf.split(X, y, groups):
        if len(set(y[tr])) < 2:
            oof_p[te] = y[tr][0]; oof_h[te] = y[tr][0]; continue
        m = LogisticRegression(max_iter=5000, C=1.0) if model is None else model.__class__(**model.get_params())
        m.fit(X[tr], y[tr])
        oof_p[te] = m.predict_proba(X[te])[:, 1]
        oof_h[te] = m.predict(X[te])
    auc = roc_auc_score(y, oof_p); acc = accuracy_score(y, oof_h)
    print(f"  {name:<52} acc={acc:.4f}  AUROC={auc:.4f}")
    return auc, acc

print("="*70)
print("[1-2] SCALAR METADATA SHORTCUTS (GroupKFold by task_id, pooled out-of-fold)")
print("="*70)
nst = np.array([r["n_steps"] for r in rows], float).reshape(-1, 1)
rew = np.array([r["reward"] for r in rows], float).reshape(-1, 1)
grouped_eval(nst, "n_steps alone (logreg)")
grouped_eval(rew, "reward alone (logreg)")
grouped_eval(np.hstack([nst, rew]), "n_steps + reward (logreg)")

# raw single-feature AUROC (no model, just ranking) + best single threshold
for nm, v in [("n_steps", nst.ravel()), ("reward", rew.ravel())]:
    a = roc_auc_score(y, v)
    print(f"  raw ranking AUROC by {nm}: {max(a,1-a):.4f}  (direction {'higher=unsafe' if a>0.5 else 'lower=unsafe'})")
    best = max(((np.mean((v > t).astype(int) == y), t) for t in np.unique(v)), key=lambda x: x[0])
    best2 = max(((np.mean((v <= t).astype(int) == y), t) for t in np.unique(v)), key=lambda x: x[0])
    b = max(best, best2)
    print(f"    best single global threshold acc on {nm}: {b[0]:.4f} at t={b[1]}")

print("\n  distributions:")
for nm, v in [("n_steps", nst.ravel()), ("reward", rew.ravel())]:
    for lbl, tag in [(0, "expert"), (1, "unsafe")]:
        vv = v[y == lbl]
        print(f"    {nm:9s} {tag:6s}: mean={vv.mean():.3f} sd={vv.std():.3f} min={vv.min():.3f} med={np.median(vv):.3f} max={vv.max():.3f}")
    print(f"    {nm} value counts: expert={dict(sorted(collections.Counter(v[y==0]).items()))}")
    print(f"    {' '*len(nm)} value counts: unsafe={dict(sorted(collections.Counter(v[y==1]).items()))}")
print()

# ---------------- other metadata fields ----------------
print("="*70)
print("[2b] OTHER METADATA FIELDS (leak check)")
print("="*70)
for f in ["cup", "terminated", "finished_deliberately", "ended_on_goal_confirmed",
          "state_verified", "plan_revisions", "n_safety", "n_scraped", "len_steps"]:
    vals = [r[f] for r in rows]
    try:
        v = np.array([float(x) if x is not None else np.nan for x in vals])
    except Exception:
        continue
    if np.isnan(v).any():
        v = np.nan_to_num(v, nan=-1)
    a = roc_auc_score(y, v); a = max(a, 1 - a)
    acc = max(max(np.mean((v > t).astype(int) == y) for t in np.unique(v)),
              max(np.mean((v <= t).astype(int) == y) for t in np.unique(v)))
    ce = collections.Counter(v[y == 0]); cu = collections.Counter(v[y == 1])
    print(f"  {f:26s} AUROC={a:.4f}  best-threshold acc={acc:.4f}")
    print(f"       expert={dict(sorted(ce.items()))}  unsafe={dict(sorted(cu.items()))}")
print()

# ---------------- 3. Bag-of-words on ACTIONS ----------------
print("="*70)
print("[3] BAG-OF-WORDS OVER ACTION STRINGS ONLY (GroupKFold by task_id)")
print("="*70)
texts = [r["action_text"] for r in rows]

def bow_eval(texts, name, **cvkw):
    oof_p = np.zeros(len(y)); oof_h = np.zeros(len(y), dtype=int)
    for tr, te in gkf.split(np.zeros(len(y)), y, groups):
        vec = CountVectorizer(**cvkw)
        Xtr = vec.fit_transform([texts[i] for i in tr])
        Xte = vec.transform([texts[i] for i in te])
        m = LogisticRegression(max_iter=5000)
        m.fit(Xtr, y[tr])
        oof_p[te] = m.predict_proba(Xte)[:, 1]
        oof_h[te] = m.predict(Xte)
    auc = roc_auc_score(y, oof_p); acc = accuracy_score(y, oof_h)
    print(f"  {name:<52} acc={acc:.4f}  AUROC={auc:.4f}")
    return auc, acc

bow_eval(texts, "BoW unigrams over actions", lowercase=True, token_pattern=r"[^\s]+")
bow_eval(texts, "BoW uni+bigrams over actions", lowercase=True, token_pattern=r"[^\s]+", ngram_range=(1, 2))

# full-data fit for token inspection
vec = CountVectorizer(lowercase=True, token_pattern=r"[^\s]+")
Xall = vec.fit_transform(texts)
m = LogisticRegression(max_iter=5000).fit(Xall, y)
names = np.array(vec.get_feature_names_out())
co = m.coef_[0]
order = np.argsort(co)
print("\n  most EXPERT-indicative action tokens:")
for i in order[:15]:
    print(f"    {names[i]:<24} w={co[i]:+.3f}  docfreq expert={int((Xall[y==0][:,i]>0).sum())} unsafe={int((Xall[y==1][:,i]>0).sum())}")
print("  most UNSAFE-indicative action tokens:")
for i in order[::-1][:15]:
    print(f"    {names[i]:<24} w={co[i]:+.3f}  docfreq expert={int((Xall[y==0][:,i]>0).sum())} unsafe={int((Xall[y==1][:,i]>0).sum())}")

# single-token separators
print("\n  SINGLE-TOKEN SEPARATORS (presence => class), by balanced accuracy:")
scored = []
for i, tok in enumerate(names):
    pres = (Xall[:, i].toarray().ravel() > 0).astype(int)
    for direction in (0, 1):
        p = pres if direction == 1 else 1 - pres
        acc = accuracy_score(y, p)
        tpr = p[y == 1].mean(); tnr = 1 - p[y == 0].mean()
        scored.append((0.5 * (tpr + tnr), acc, tok, direction, int(pres[y == 0].sum()), int(pres[y == 1].sum())))
scored.sort(reverse=True)
for bal, acc, tok, d, ne, nu in scored[:15]:
    print(f"    token={tok:<22} present=>{'UNSAFE' if d==1 else 'EXPERT'}  balacc={bal:.4f} acc={acc:.4f}  present in expert={ne}/{int((y==0).sum())} unsafe={nu}/{int((y==1).sum())}")

# verb-only (first word of each action)
verbs = [" ".join(a.split()[0].lower() for a in r["actions"] if a.strip()) for r in rows]
print()
bow_eval(verbs, "BoW over ACTION VERBS only (first word)", lowercase=True, token_pattern=r"[^\s]+")
# object-only (rest of action)
objs = [" ".join(" ".join(a.split()[1:]).lower() for a in r["actions"] if a.strip()) for r in rows]
bow_eval(objs, "BoW over ACTION OBJECTS only (words after verb)", lowercase=True, token_pattern=r"[^\s]+")
# observations only
bow_eval([r["obs_text"] for r in rows], "BoW over OBSERVATIONS only", lowercase=True, token_pattern=r"[^\s,]+")
# plan text
bow_eval([r["plan"] for r in rows], "BoW over PLAN text", lowercase=True, token_pattern=r"[^\s]+")
print()

# ---------------- combined trivial ----------------
print("="*70)
print("[5] COMBINED CHEAP FEATURES")
print("="*70)
X = np.hstack([nst, rew,
               np.array([[float(r["state_verified"] or 0)] for r in rows]),
               np.array([[float(r["plan_revisions"] or 0)] for r in rows])])
grouped_eval(X, "n_steps + reward + state_verified + plan_revisions")
grouped_eval(np.array([[float(r["state_verified"] or 0)] for r in rows]), "state_verified ALONE")
print()

# ---------------- restricted to overlapping tasks ----------------
print("="*70)
print("[6] SANITY: same tests restricted to task ids present in BOTH classes")
print("="*70)
idx = [i for i, r in enumerate(rows) if r["task_id"] in both]
print(f"  n = {len(idx)}")
if len(idx) > 20:
    yy = y[idx]; gg = groups[idx]
    print(f"  expert={int((yy==0).sum())} unsafe={int((yy==1).sum())}")
print()

# ---------------- action-count / step-index structure ----------------
print("="*70)
print("[7] STRUCTURAL FINGERPRINTS")
print("="*70)
last = collections.Counter((r["actions"][-1].lower() if r["actions"] else "<empty>", r["y"]) for r in rows)
print("  last action by class:")
agg = collections.defaultdict(lambda: [0, 0])
for (a, lab), c in last.items():
    agg[a][lab] += c
for a, (ce, cu) in sorted(agg.items(), key=lambda kv: -(kv[1][0] + kv[1][1]))[:12]:
    print(f"    {a:<28} expert={ce} unsafe={cu}")
first = collections.defaultdict(lambda: [0, 0])
for r in rows:
    first[(r["actions"][0].split()[0].lower() if r["actions"] else "<empty>")][r["y"]] += 1
print("  first action verb by class:")
for a, (ce, cu) in sorted(first.items(), key=lambda kv: -(kv[1][0] + kv[1][1]))[:12]:
    print(f"    {a:<28} expert={ce} unsafe={cu}")
