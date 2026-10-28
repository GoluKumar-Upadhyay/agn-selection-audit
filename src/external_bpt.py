"""External validation on SPECTROSCOPICALLY SELECTED narrow-line AGN (MPA-JHU BPT class) vs star-forming galaxies,
matched in redshift and r magnitude, excluding every object that is in the training table.   CPU only.

  python src/external_bpt.py fetch    # needs internet (SkyServer SQL).  NOT tested from my sandbox.
  python src/external_bpt.py score    # tested offline on synthetic data

IMPORTANT interpretation: the training label is 'spectroscopic class QSO vs GALAXY'.  Narrow-line (type-2) AGN are
labelled GALAXY in SDSS, and some of them sit in the training negatives.  This test therefore asks whether the
QSO-vs-galaxy score also ranks narrow-line AGN above star-forming galaxies.  A low sensitivity here is a legitimate
finding (the model is a broad-line-QSO finder) and must be reported, not hidden.
Before the first run check in the SkyServer schema browser: table galSpecExtra, column bptclass
(expected codes: 1 = star forming, 4 = AGN; verify), and that your training query used modelMag (not extinction-corrected).
"""
import io, os, sys, time, json, argparse, numpy as np, pandas as pd, requests
sys.path.insert(0, os.path.dirname(__file__))
from data import *; from stats import *
from external_validation import evaluate
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); RAW = f"{ROOT}/data/external_bpt_raw.csv"
URL = "https://skyserver.sdss.org/dr17/SkyServerWS/SearchTools/SqlSearch"
Z_BINS = [0.02, 0.04, 0.06, 0.08, 0.10, 0.15, 0.20, 0.30]; R_BINS = [14.0, 15.5, 16.5, 17.2, 17.8]
SQL = """SELECT TOP {n} p.objID AS objid, p.ra, p.dec, p.modelMag_u AS u, p.modelMag_g AS g, p.modelMag_r AS r, p.modelMag_i AS i, p.modelMag_z AS z,
 p.petroR50_r, p.petroR90_r, s.z AS redshift, s.specObjID AS specobjid, x.bptclass
FROM SpecObj AS s JOIN PhotoObj AS p ON p.objID = s.bestObjID JOIN galSpecExtra AS x ON x.specObjID = s.specObjID
WHERE s.class='GALAXY' AND s.zWarning=0 AND s.sciencePrimary=1 AND p.mode=1 AND x.bptclass={bpt}
 AND s.z BETWEEN {z0} AND {z1} AND p.modelMag_r BETWEEN {r0} AND {r1} AND (s.specObjID % 10007) < {cut}
ORDER BY (s.specObjID % 9973)"""

def run_sql(q, tries=3):
    for k in range(tries):
        try:
            r = requests.get(URL, params={"cmd": q, "format": "csv"}, timeout=180)
            if r.status_code == 200 and "objid" in r.text[:3000].lower(): return pd.read_csv(io.StringIO(r.text), comment="#")
            print("  bad response:", r.text[:200].replace("\n", " "))
        except Exception as e: print("  retry", k, e)
        time.sleep(5)
    return pd.DataFrame()

def fetch(per_cell, cut):
    parts = []
    for bpt, lab in ((4, 1), (1, 0)):
        for zi in range(len(Z_BINS) - 1):
            for ri in range(len(R_BINS) - 1):
                d = run_sql(SQL.format(n=per_cell * 2, bpt=bpt, z0=Z_BINS[zi], z1=Z_BINS[zi + 1], r0=R_BINS[ri], r1=R_BINS[ri + 1], cut=cut))
                if len(d): d["label"], d["zbin"], d["rbin"] = lab, zi, ri; parts.append(d)
                print(f"bpt={bpt} z{zi} r{ri}: {len(d)}", flush=True)
    pd.concat(parts, ignore_index=True).to_csv(RAW, index=False); print("saved", RAW)

def prepare(raw):
    """clean, drop training overlap, add features, match positives/negatives per (z,r) cell"""
    d = raw.drop_duplicates("objid").copy()
    ok = np.ones(len(d), bool)
    for c in MAGS: ok &= ((d[c] > 10) & (d[c] < 25)).values
    d = d[ok].dropna(subset=MAGS + ["petroR50_r", "petroR90_r", "redshift"])
    train_ids = set(pd.read_csv(f"{ROOT}/data/sdss_agn_dataset.csv", usecols=["objid"]).objid)
    n0 = len(d); d = d[~d.objid.isin(train_ids)]; print(f"removed {n0-len(d)} objects that are in the training table")
    d = add_features(d); rng = np.random.default_rng(0); keep = []
    for _, g in d.groupby(["zbin", "rbin"]):
        p, n = g.index[g.label == 1], g.index[g.label == 0]; k = min(len(p), len(n))
        if k >= 5: keep += list(rng.choice(p, k, replace=False)) + list(rng.choice(n, k, replace=False))
    d = d.loc[keep].reset_index(drop=True); d["subtype"] = np.where(d.label == 1, "BPT-AGN", "BPT-star-forming"); return d

def score(raw_path, n_models=10):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    import xgboost as xgb
    ext = prepare(pd.read_csv(raw_path)); print("external cohort:", ext.label.value_counts().to_dict(), "| z range", ext.redshift.min().round(3), ext.redshift.max().round(3))
    df = pd.read_csv(f"{ROOT}/data/image_cohort.csv"); df["log_r50"] = np.log10(df.petroR50_r.clip(lower=0.05))
    rng = np.random.default_rng(0); key = pd.Series("", index=df.index)                                  # same matched TRAIN cohort as the paper
    for c, w in (("r", 0.5), ("log_r50", 0.1)): key = key + "_" + np.floor(df[c] / w).astype(int).astype(str)
    keep = []
    for _, g in df.groupby(key):
        q, gl = g.index[g.label == 1], g.index[g.label == 0]; n = min(len(q), len(gl))
        if n: keep += list(rng.choice(q, n, replace=False)) + list(rng.choice(gl, n, replace=False))
    tr_all = df.loc[np.sort(keep)].reset_index(drop=True); cols = FEATURE_SETS["no_z_no_Mr"]; ytr = tr_all.label.values
    px = np.zeros(len(ext)); 
    for seed in range(n_models):                                                                         # 10 bootstrap-bagged XGBoost models
        b = np.random.default_rng(seed).integers(0, len(tr_all), len(tr_all))
        m = xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, n_jobs=1, random_state=seed,
                              scale_pos_weight=float((ytr[b] == 0).sum() / max(1, (ytr[b] == 1).sum())), eval_metric="logloss").fit(tr_all[cols].values[b], ytr[b])
        px += m.predict_proba(ext[cols].values)[:, 1] / n_models
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")).fit(tr_all[["u_g", "g_r", "r_i", "i_z"]].values, ytr)
    pl = lr.predict_proba(ext[["u_g", "g_r", "r_i", "i_z"]].values)[:, 1]
    os.makedirs(f"{ROOT}/results", exist_ok=True); out = {}
    for name, s in (("XGBoost [12 shortcut-free], pre-specified threshold 0.5", px), ("Colour logistic regression", pl)):
        e = ext[["objid", "label", "subtype", "redshift"]].assign(score=s); r = evaluate(e, thr=0.5); out[name] = r; print("\n==", name); print(json.dumps(r, indent=1, default=float))
        if name.startswith("XGB"): e.to_csv(f"{ROOT}/results/external_scores.csv", index=False)
    json.dump(out, open(f"{ROOT}/results/external_table6.json", "w"), indent=1, default=float)
    import external_figures; os.makedirs(f"{ROOT}/figures/paper", exist_ok=True)
    external_figures.make(pd.read_csv(f"{ROOT}/results/external_scores.csv"), f"{ROOT}/figures/paper/fig8_external.png", thr=0.5); print("saved results/external_table6.json, fig8_external.png")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["fetch", "score"]); ap.add_argument("--per-cell", type=int, default=60)
    ap.add_argument("--hash-cut", type=int, default=3000); ap.add_argument("--raw", default=RAW); a = ap.parse_args()
    fetch(a.per_cell, a.hash_cut) if a.cmd == "fetch" else score(a.raw)
