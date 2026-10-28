"""CPU-only: (1) fairly TUNED tabular baselines on the same matched splits as the fusion models,
(2) paired tests tuned-baseline vs fusion, and gated-attention vs concatenation.
Run AFTER run_paper_experiments.py (needs results/paper/predictions.pkl).
Env: QUICK=1 (3 seeds, tiny search) | NITER_XGB / NITER_MLP | SEEDS=42,43,...
Outputs: results/paper_tables/Table8_tuned_baselines.csv, Table9_paired_contrasts.csv (+ .md)
"""
import os, sys, pickle, time, warnings, numpy as np, pandas as pd
from scipy import stats as sps
from scipy.stats import loguniform, randint, uniform
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import xgboost as xgb
sys.path.insert(0, os.path.dirname(__file__))
from data import *; from stats import *
warnings.filterwarnings("ignore")

# --- Windows fix: make console output and file writes UTF-8 (needed for the "−" and "±" characters) ---
for _s in (sys.stdout, sys.stderr):
    try: _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass

def to_md(d):
    """markdown table; falls back to a plain implementation if 'tabulate' is not installed"""
    try:
        return d.to_markdown(index=False)
    except ImportError:
        cols = list(d.columns)
        lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
        lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in d.itertuples(index=False)]
        return "\n".join(lines)

def write_text(path, text):
    with open(path, "w", encoding="utf-8") as f: f.write(text)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); IN = f"{ROOT}/results/paper"; OUT = f"{ROOT}/results/paper_tables"; os.makedirs(OUT, exist_ok=True)
QUICK = os.environ.get("QUICK") == "1"
NX = int(os.environ.get("NITER_XGB", 4 if QUICK else 40)); NM = int(os.environ.get("NITER_MLP", 3 if QUICK else 15))
MAIN = "Joint r + size matched (MAIN)"
PRED = pickle.load(open(f"{IN}/predictions.pkl", "rb"))
seeds = sorted({s for (c, s, m) in PRED if c == MAIN})
if os.environ.get("SEEDS"): seeds = [int(x) for x in os.environ["SEEDS"].split(",")]
elif QUICK: seeds = seeds[:3]
df = pd.read_csv(f"{ROOT}/data/image_cohort.csv"); df["log_r50"] = np.log10(df.petroR50_r.clip(lower=0.05)); y = df.label.values
def cem(d, keys, seed=0):                      # identical to run_paper_experiments.py
    rng = np.random.default_rng(seed); key = pd.Series("", index=d.index)
    for c, w in keys: key = key + "_" + np.floor(d[c] / w).astype(int).astype(str)
    keep = []
    for _, g in d.groupby(key):
        q, gl = g.index[g.label == 1], g.index[g.label == 0]; n = min(len(q), len(gl))
        if n: keep += list(rng.choice(q, n, replace=False)) + list(rng.choice(gl, n, replace=False))
    return np.sort(np.array(keep))
cidx = cem(df, [("r", 0.5), ("log_r50", 0.1)]); sub = df.iloc[cidx].reset_index(drop=True)
SETS = {"12 shortcut-free": FEATURE_SETS["no_z_no_Mr"], "photometry + colours": MAGS + COLOURS}
t0 = time.time(); log = lambda *a: print(f"[{time.time()-t0:7.1f}s]", *a, flush=True)

XDIST = dict(n_estimators=randint(100, 700), max_depth=randint(2, 9), learning_rate=loguniform(0.01, 0.3), subsample=uniform(0.6, 0.4),
             colsample_bytree=uniform(0.5, 0.5), min_child_weight=randint(1, 10), reg_lambda=loguniform(0.1, 20), gamma=uniform(0, 2))
MDIST = dict(mlpclassifier__hidden_layer_sizes=[(64,), (128, 64), (256, 128, 64), (128, 128), (64, 32)], mlpclassifier__alpha=loguniform(1e-5, 1e-1),
             mlpclassifier__learning_rate_init=loguniform(1e-4, 1e-2))
T = {}                                         # (seed, name) -> (te, p)
rows = []
for seed in seeds:
    tr_p, va_p, te_p = split_70_15_15(sub, seed); tr, te = cidx[tr_p], cidx[te_p]
    assert (te == PRED[(MAIN, seed, "XGBoost [12 shortcut-free]")]["te"]).all(), "split mismatch - run on the same cohort/seeds as run_paper_experiments.py"
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    for sname, cols in SETS.items():
        Xtr, Xte = df[cols].values[tr], df[cols].values[te]
        spw = float((y[tr] == 0).sum() / (y[tr] == 1).sum())
        s1 = RandomizedSearchCV(xgb.XGBClassifier(n_jobs=1, random_state=seed, scale_pos_weight=spw, eval_metric="logloss"), XDIST, n_iter=NX, cv=cv, scoring="roc_auc", n_jobs=-1, random_state=seed).fit(Xtr, y[tr])
        T[(seed, f"Tuned XGBoost [{sname}]")] = (te, s1.predict_proba(Xte)[:, 1])
        s2 = RandomizedSearchCV(make_pipeline(StandardScaler(), MLPClassifier(early_stopping=True, max_iter=400, random_state=seed)), MDIST, n_iter=NM, cv=cv, scoring="roc_auc", n_jobs=-1, random_state=seed).fit(Xtr, y[tr])
        T[(seed, f"Tuned MLP [{sname}]")] = (te, s2.predict_proba(Xte)[:, 1])
        rows.append(dict(seed=seed, sset=sname, xgb_cv_auc=s1.best_score_, mlp_cv_auc=s2.best_score_, **{f"xgb_{k}": v for k, v in s1.best_params_.items() if k in ("max_depth", "n_estimators", "learning_rate")}))
    log("seed", seed, "tuned")
pd.DataFrame(rows).to_csv(f"{OUT}/tuned_search_log.csv", index=False, encoding="utf-8")

# cache tuned predictions so a crash after tuning never forces a re-run
with open(f"{OUT}/tuned_predictions.pkl", "wb") as f: pickle.dump(T, f)

def get(name, seed):
    if (seed, name) in T: return T[(seed, name)]
    P = PRED[(MAIN, seed, name)]; return P["te"], P["p"]
ms = lambda v: f"{np.mean(v):.4f} ± {np.std(v, ddof=1) if len(v) > 1 else 0:.4f}"
def paired(a, b):
    d, ps = [], []
    for s in seeds:
        ta, pa = get(a, s); tb, pb = get(b, s); assert (ta == tb).all()
        r = delong_roc_test(y[ta], pa, pb); d.append(r[2]); ps.append(r[4])
    d = np.array(d)
    try: w = sps.wilcoxon(d).pvalue
    except Exception: w = float("nan")
    return dict(mean_diff=f"{d.mean():+.4f}", seeds_a_better=f"{(d > 0).sum()}/{len(d)}", delong_sig=f"{(np.array(ps) < .05).sum()}/{len(d)}", wilcoxon_p=f"{w:.3g}")

# Table 8: default vs tuned baselines
t8 = []
for sname, dflt in (("12 shortcut-free", ("XGBoost [12 shortcut-free]", "MLP [12 shortcut-free]")), ("photometry + colours", ("XGB | Photometry + colours only", "MLP | Photometry + colours only"))):
    for kind, dn in zip(("XGBoost", "MLP"), dflt):
        auc_t = [roc_auc_score(y[get(f"Tuned {kind} [{sname}]", s)[0]], get(f"Tuned {kind} [{sname}]", s)[1]) for s in seeds]
        auc_d = [roc_auc_score(y[get(dn, s)[0]], get(dn, s)[1]) for s in seeds]
        t8.append({"Features": sname, "Model": kind, "Default AUC": ms(auc_d), "Tuned AUC": ms(auc_t), "Tuned − default": f"{np.mean(auc_t)-np.mean(auc_d):+.4f}"})
T8 = pd.DataFrame(t8); T8.to_csv(f"{OUT}/Table8_tuned_baselines.csv", index=False, encoding="utf-8"); write_text(f"{OUT}/Table8_tuned_baselines.md", to_md(T8))
# Table 9: paired contrasts
G12, C12, PCF = "Gated attention [12 shortcut-free]", "Concatenation [12 shortcut-free]", "Fusion | Photometry + colours only"
pairs = [(G12, "Tuned XGBoost [12 shortcut-free]"), (C12, "Tuned XGBoost [12 shortcut-free]"), (G12, "Tuned MLP [12 shortcut-free]"), (C12, "Tuned MLP [12 shortcut-free]"),
         (PCF, "Tuned XGBoost [photometry + colours]"), (PCF, "Tuned MLP [photometry + colours]"), (G12, C12), ("Cross-attention, no gate [12 shortcut-free]", C12)]
T9 = pd.DataFrame([{"a": a, "b": b, **paired(a, b)} for a, b in pairs]); T9.to_csv(f"{OUT}/Table9_paired_contrasts.csv", index=False, encoding="utf-8"); write_text(f"{OUT}/Table9_paired_contrasts.md", to_md(T9))
print(T8.to_string(index=False)); print(T9.to_string(index=False)); log("DONE")