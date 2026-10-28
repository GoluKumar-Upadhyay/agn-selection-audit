"""Aggregate fine-tuned runs (results/ft_predictions_<task>_<mode>_<tab>_<seed>.npy) and compare them with the frozen-feature models
on the SAME test objects, using the corrected resampled t-test + Holm adjustment (as in stats_corrected.py).
  python src/analyze_finetune.py            # QSO task (needs results/paper/predictions.pkl)
Needs >= 3 seeds per configuration; use all ten (42-51) for the paper."""
import os, sys, glob, pickle, re, numpy as np, pandas as pd
from scipy import stats as sps
from sklearn.metrics import roc_auc_score
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); OUT = f"{ROOT}/results/paper_tables"; os.makedirs(OUT, exist_ok=True)
TOT = 0.15 / 0.70; MAIN = "Joint r + size matched (MAIN)"
y = pd.read_csv(f"{ROOT}/data/image_cohort.csv").label.values; PR = pickle.load(open(f"{ROOT}/results/paper/predictions.pkl", "rb"))
frozen = {}
for (c, s, m), v in PR.items():
    if c == MAIN: frozen.setdefault(m, {})[s] = (np.asarray(v["te"]), np.asarray(v["p"]))
ft = {}
for f in glob.glob(f"{ROOT}/results/ft_predictions_qso_*.npy"):
    m = re.match(r"ft_predictions_qso_(.+)_(photo12|full14)_(\d+)\.npy", os.path.basename(f))
    if not m: continue
    te, p = np.load(f); ft.setdefault(f"Fine-tuned {m.group(1)} [{m.group(2)}]", {})[int(m.group(3))] = (te.astype(int), p)
ft = {k: v for k, v in ft.items() if len(v) >= 3}
if not ft: sys.exit("no fine-tuned configuration has >= 3 seeds yet")
def auc_vec(d, seeds): return np.array([roc_auc_score(y[d[s][0]], d[s][1]) for s in seeds])
def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
    for r, i in enumerate(o): run = max(run, (m - r) * p[i]); adj[i] = min(1, run)
    return adj
allm = {**frozen, **ft}; rows = []
for k, d in ft.items():
    seeds = sorted(d); a = auc_vec(d, seeds); h = sps.t.ppf(.975, len(a) - 1) * np.sqrt((1 / len(a) + TOT) * a.var(ddof=1))
    rows.append(dict(model=k, n_seeds=len(a), mean_auc=a.mean(), sd=a.std(ddof=1), ci95_lo=a.mean() - h, ci95_hi=min(1, a.mean() + h)))
M = pd.DataFrame(rows)
E, X, C, G = "EfficientNet-B0 (frozen, image only)", "XGBoost [12 shortcut-free]", "Concatenation [12 shortcut-free]", "Gated attention [12 shortcut-free]"
comps = []
for k in ft:
    comps += [(k, E), (k, X), (k, G)]
names = list(ft)
ig = [n for n in names if "img_gap" in n]; ag = [n for n in names if "attn_gate" in n]; cc = [n for n in names if "concat" in n]
if ig and ag: comps.append((ag[0], ig[0]))
if ig and cc: comps.append((cc[0], ig[0]))
if cc and ag: comps.append((ag[0], cc[0]))
out = []
for a_, b_ in comps:
    seeds = sorted(set(allm[a_]) & set(allm[b_]))
    for s in seeds: assert (allm[a_][s][0] == allm[b_][s][0]).all(), f"test objects differ for seed {s}: {a_} vs {b_}"
    d = auc_vec(allm[a_], seeds) - auc_vec(allm[b_], seeds); k = len(d); se = np.sqrt((1 / k + TOT) * d.var(ddof=1)); t = d.mean() / se; h = sps.t.ppf(.975, k - 1) * se
    out.append(dict(a=a_, b=b_, n_seeds=k, mean_diff=d.mean(), ci95_lo=d.mean() - h, ci95_hi=d.mean() + h, cohens_dz=d.mean() / d.std(ddof=1), p_corrected=2 * sps.t.sf(abs(t), k - 1), a_better=f"{(d > 0).sum()}/{k}"))
P = pd.DataFrame(out); P["p_holm"] = holm(P.p_corrected.values)
M.to_csv(f"{OUT}/Table16_finetuned_models.csv", index=False); P.to_csv(f"{OUT}/Table17_finetuned_paired.csv", index=False)
open(f"{OUT}/Table16_finetuned_models.md", "w").write(M.to_markdown(index=False, floatfmt=".4f")); open(f"{OUT}/Table17_finetuned_paired.md", "w").write(P.to_markdown(index=False, floatfmt=".4f"))
pd.set_option("display.width", 220, "display.max_columns", 20, "display.max_colwidth", 42); print(M.round(4).to_string(index=False)); print(); print(P.round(4).to_string(index=False))