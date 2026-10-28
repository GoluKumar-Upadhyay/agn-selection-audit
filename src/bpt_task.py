"""Well-posed AGN task: narrow-line (BPT) AGN vs star-forming galaxies, matched in redshift AND r magnitude by construction.
Uses data/external_bpt_raw.csv produced by `external_bpt.py fetch` (no images needed, CPU only, ~2 min).
It answers: (1) can photometry alone separate narrow-line AGN from star-forming galaxies when z and r are matched?
            (2) how much do redshift / luminosity proxies add once z is matched (editor's point 2)?
            (3) does AUC hold inside every redshift bin?
Outputs: results/paper_tables/Table10_bpt_task.csv, Table11_bpt_by_redshift.csv  (+ .md)"""
import os, sys, numpy as np, pandas as pd, xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.dirname(__file__))
from data import *; from stats import *; import external_bpt
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); OUT = f"{ROOT}/results/paper_tables"; os.makedirs(OUT, exist_ok=True)
d = external_bpt.prepare(pd.read_csv(f"{ROOT}/data/external_bpt_raw.csv")); y = d.label.values
print("matched BPT cohort:", len(d), d.label.value_counts().to_dict(), "| redshift overlap coeff:", round(overlap_coefficient(d.redshift[y == 1].values, d.redshift[y == 0].values), 3),
      "| r overlap:", round(overlap_coefficient(d.r[y == 1].values, d.r[y == 0].values), 3))
SETS = {"XGBoost [12 shortcut-free]": ("xgb", FEATURE_SETS["no_z_no_Mr"]), "XGBoost [colours only]": ("xgb", COLOURS), "XGBoost [14 incl. z, M_r]": ("xgb", ALL_FEATURES),
        "XGBoost [morphology only]": ("xgb", FEATURE_SETS["morphology_only"]), "Colour logistic regression": ("lr", ["u_g", "g_r", "r_i", "i_z"]), "Redshift alone": ("raw", ["redshift"])}
rows, pooled = [], {k: [] for k in SETS}
for seed in range(42, 52):
    tr, va, te = split_70_15_15(d, seed); r = {"seed": seed}
    for name, (kind, cols) in SETS.items():
        X = d[cols].values
        if kind == "xgb": p = xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, n_jobs=1, random_state=seed, eval_metric="logloss").fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        elif kind == "lr": p = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        else: p = X[te, 0]
        r[name] = roc_auc_score(y[te], p); pooled[name].append(pd.DataFrame({"te": te, "p": p}))
    rows.append(r)
R = pd.DataFrame(rows); T10 = pd.DataFrame([{"Model": k, "AUC (mean ± sd, 10 splits)": f"{R[k].mean():.4f} ± {R[k].std(ddof=1):.4f}", "min – max": f"{R[k].min():.3f} – {R[k].max():.3f}"} for k in SETS])
T10.to_csv(f"{OUT}/Table10_bpt_task.csv", index=False); open(f"{OUT}/Table10_bpt_task.md", "w").write(T10.to_markdown(index=False)); print(T10.to_string(index=False))
# AUC inside each redshift bin (pooled test predictions of the 12 shortcut-free model; descriptive, test sets overlap across seeds)
P = pd.concat(pooled["XGBoost [12 shortcut-free]"]); P["label"] = y[P.te.values]; P["z"] = d.redshift.values[P.te.values]; P["r"] = d.r.values[P.te.values]
rows = []
for col, edges in (("z", [0.02, 0.04, 0.06, 0.08, 0.10, 0.15, 0.30]), ("r", [14, 15.5, 16.5, 17.2, 17.8])):
    P["bin"] = pd.cut(P[col], edges)
    for b, g in P.groupby("bin", observed=True):
        if g.label.nunique() == 2 and len(g) > 40: rows.append({"variable": col, "bin": str(b), "n (pooled)": len(g), "AUC": round(roc_auc_score(g.label, g.p), 4)})
T11 = pd.DataFrame(rows); T11.to_csv(f"{OUT}/Table11_bpt_by_redshift.csv", index=False); open(f"{OUT}/Table11_bpt_by_redshift.md", "w").write(T11.to_markdown(index=False)); print(T11.to_string(index=False))