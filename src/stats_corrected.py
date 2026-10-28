"""Corrected inference for repeated random train/test splits (answers 'ten overlapping splits are not ten independent experiments').
For every paired comparison (same test objects within a split) it reports
  - mean AUC difference and Cohen's d_z (effect size = mean / SD of per-split differences)
  - 95% CI and p-value from the Nadeau-Bengio corrected resampled t-test: variance inflated by (1/k + n_test/n_train)
  - Holm-Bonferroni adjusted p-value within each cohort's family of comparisons
  - the naive Wilcoxon p (descriptive only)
and, for each model, mean AUC with a corrected 95% CI.
  python src/stats_corrected.py            (needs results/paper/predictions.pkl ; and bpt_predictions.pkl for the BPT task)"""
import os, sys, pickle, numpy as np, pandas as pd
from scipy import stats as sps
from sklearn.metrics import roc_auc_score
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); OUT = f"{ROOT}/results/paper_tables"; os.makedirs(OUT, exist_ok=True)
TEST_OVER_TRAIN = 0.15 / 0.70            # 70/15/15 splits
def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
    for r, i in enumerate(o): run = max(run, (m - r) * p[i]); adj[i] = min(1, run)
    return adj
def analyse(label, aucs, comps, y_of):
    """aucs: {model: array per split}; comps: list of (a, b)"""
    rows, k = [], None
    for a, b in comps:
        d = aucs[a] - aucs[b]; k = len(d); var = d.var(ddof=1); se = np.sqrt((1 / k + TEST_OVER_TRAIN) * var); t = d.mean() / se; p = 2 * sps.t.sf(abs(t), k - 1); h = sps.t.ppf(.975, k - 1) * se
        try: w = sps.wilcoxon(d).pvalue
        except Exception: w = np.nan
        rows.append(dict(cohort=label, a=a, b=b, mean_diff=d.mean(), ci95_lo=d.mean() - h, ci95_hi=d.mean() + h, cohens_dz=d.mean() / d.std(ddof=1), p_corrected_t=p, wilcoxon_p_descriptive=w, splits_a_better=f"{(d > 0).sum()}/{k}"))
    T = pd.DataFrame(rows); T["p_holm"] = holm(T.p_corrected_t.values); T["significant_after_holm"] = T.p_holm < 0.05
    M = pd.DataFrame([dict(cohort=label, model=m, mean_auc=v.mean(), sd=v.std(ddof=1), ci95_lo=v.mean() - sps.t.ppf(.975, len(v) - 1) * np.sqrt((1 / len(v) + TEST_OVER_TRAIN) * v.var(ddof=1)),
                           ci95_hi=min(1.0, v.mean() + sps.t.ppf(.975, len(v) - 1) * np.sqrt((1 / len(v) + TEST_OVER_TRAIN) * v.var(ddof=1)))) for m, v in aucs.items()])
    return T, M
allT, allM = [], []
p1 = f"{ROOT}/results/paper/predictions.pkl"
if os.path.exists(p1):
    y = pd.read_csv(f"{ROOT}/data/image_cohort.csv").label.values; PR = pickle.load(open(p1, "rb")); MAIN = "Joint r + size matched (MAIN)"; aucs = {}
    for (c, s, m), v in PR.items():
        if c == MAIN: aucs.setdefault(m, {})[s] = roc_auc_score(y[v["te"]], v["p"])
    aucs = {m: np.array([d[s] for s in sorted(d)]) for m, d in aucs.items()}
    E, G, C, N = "EfficientNet-B0 (frozen, image only)", "Gated attention [12 shortcut-free]", "Concatenation [12 shortcut-free]", "Cross-attention, no gate [12 shortcut-free]"
    comps = [(G, E), (C, E), (G, "XGBoost [12 shortcut-free]"), (C, "XGBoost [12 shortcut-free]"), (G, C), (N, C), ("Gated attention [14]", "XGBoost [14]"), ("Fusion | Photometry + colours only", "XGB | Photometry + colours only")]
    T, M = analyse("QSO vs galaxy (matched)", aucs, comps, y); allT.append(T); allM.append(M)
p2 = f"{ROOT}/results/paper/bpt_predictions.pkl"
if os.path.exists(p2):
    y = pd.read_csv(f"{ROOT}/data/bpt_cohort_matched.csv").label.values; PR = pickle.load(open(p2, "rb")); aucs = {}
    for (s, m), (te, p) in PR.items(): aucs.setdefault(m, {})[s] = roc_auc_score(y[te], p)
    aucs = {m: np.array([d[s] for s in sorted(d)]) for m, d in aucs.items()}
    E = "EfficientNet-B0 (frozen, image only)"
    comps = [("Gated attention [12]", E), ("Concatenation [12]", E), ("Gated attention [12]", "XGBoost [12]"), ("Concatenation [12]", "XGBoost [12]"), ("Gated attention [12]", "MLP [12]"),
             ("Gated attention [12]", "Concatenation [12]"), ("Cross-attention, no gate [12]", "Concatenation [12]"), ("Gated attention [14]", "XGBoost [14]"), ("Gated attention [14]", "Gated attention [12]")]
    T, M = analyse("BPT AGN vs star-forming (matched)", aucs, comps, y); allT.append(T); allM.append(M)
T = pd.concat(allT, ignore_index=True); M = pd.concat(allM, ignore_index=True)
f = lambda x: f"{x:.4f}" if isinstance(x, float) else x
T.to_csv(f"{OUT}/Table14_corrected_paired_stats.csv", index=False); M.to_csv(f"{OUT}/Table15_model_auc_corrected_ci.csv", index=False)
open(f"{OUT}/Table14_corrected_paired_stats.md", "w").write(T.to_markdown(index=False, floatfmt=".4f")); open(f"{OUT}/Table15_model_auc_corrected_ci.md", "w").write(M.to_markdown(index=False, floatfmt=".4f"))
pd.set_option("display.width", 250, "display.max_columns", 20, "display.max_colwidth", 38); print(T.round(4).to_string(index=False)); print(); print(M.round(4).to_string(index=False))