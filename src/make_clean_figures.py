"""Publication-quality ROC / PR / confusion-matrix / reliability figures, averaged over all splits (no smoothing).
  python src/make_clean_figures.py qso     # needs results/paper/predictions.pkl  (from run_paper_experiments.py)
  python src/make_clean_figures.py bpt     # needs results/paper/bpt_predictions.pkl (rerun run_bpt_fusion.py once; it now saves it)
ROC/PR: each split's curve is interpolated on a common grid; line = mean over splits, band = +-1 SD over splits.
Confusion matrices: mean count per split at the pre-specified threshold 0.5.  Reliability: equal-count bins per split, mean +- SD over splits."""
import os, sys, pickle, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score, confusion_matrix
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); which = (sys.argv[1] if len(sys.argv) > 1 else "qso").lower()
OUT = f"{ROOT}/figures/paper_clean"; os.makedirs(OUT, exist_ok=True)
if which == "qso":
    y = pd.read_csv(f"{ROOT}/data/image_cohort.csv").label.values; PR = pickle.load(open(f"{ROOT}/results/paper/predictions.pkl", "rb")); MAIN = "Joint r + size matched (MAIN)"
    runs = {}
    for (c, s, m), v in PR.items():
        if c == MAIN: runs.setdefault(m, {})[s] = (np.asarray(v["te"]), np.asarray(v["p"]))
    NAMES = [("EfficientNet-B0 (frozen, image only)", "Image only (EfficientNet-B0)", "#7f7f7f", "-"), ("XGBoost [12 shortcut-free]", "XGBoost, 12 features", "#1f77b4", "-"),
             ("Concatenation [12 shortcut-free]", "Concatenation, 12 features", "#2ca02c", "-"), ("Gated attention [12 shortcut-free]", "Gated attention, 12 features", "#d62728", "-"),
             ("XGBoost [14]", "XGBoost, 14 features (incl. z, M_r)", "#1f77b4", ":")]
    GATED, IMG, TAB, TITLE = NAMES[3][0], NAMES[0][0], NAMES[1][0], "QSO vs galaxy, matched cohort"
else:
    y = pd.read_csv(f"{ROOT}/data/bpt_cohort_matched.csv").label.values; PR = pickle.load(open(f"{ROOT}/results/paper/bpt_predictions.pkl", "rb")); runs = {}
    for (s, m), (te, p) in PR.items(): runs.setdefault(m, {})[s] = (np.asarray(te), np.asarray(p))
    NAMES = [("EfficientNet-B0 (frozen, image only)", "Image only (EfficientNet-B0)", "#7f7f7f", "-"), ("XGBoost [12]", "XGBoost, 12 features", "#1f77b4", "-"),
             ("Concatenation [12]", "Concatenation, 12 features", "#2ca02c", "-"), ("Gated attention [12]", "Gated attention, 12 features", "#d62728", "-"), ("XGBoost [14]", "XGBoost, 14 features (incl. z, M_r)", "#1f77b4", ":")]
    GATED, IMG, TAB, TITLE = NAMES[3][0], NAMES[0][0], NAMES[1][0], "BPT AGN vs star-forming, matched cohort"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": .25, "legend.frameon": False, "savefig.dpi": 300})
fg = np.linspace(0, 1, 501); n_splits = len(next(iter(runs.values())))
def curves(m):
    R, Pc, au, ap = [], [], [], []
    for s, (te, p) in sorted(runs[m].items()):
        yt = y[te]; f, t, _ = roc_curve(yt, p); ti = np.interp(fg, f, t); ti[0] = 0; R.append(ti)
        pr, rc, _ = precision_recall_curve(yt, p); Pc.append(np.interp(fg, rc[::-1], pr[::-1])); au.append(roc_auc_score(yt, p)); ap.append(average_precision_score(yt, p))
    return np.array(R), np.array(Pc), np.array(au), np.array(ap)
def band(ax, x, M, col, ls, lab):
    mu, sd = M.mean(0), M.std(0, ddof=1); ax.plot(x, mu, color=col, ls=ls, lw=1.8, label=lab); ax.fill_between(x, np.clip(mu - sd, 0, 1), np.clip(mu + sd, 0, 1), color=col, alpha=.12, lw=0)
# ---------------- main 2x2 figure
fig, ax = plt.subplots(2, 2, figsize=(11, 9.5)); axr, axp, axc, axl = ax.ravel()
ins = axr.inset_axes([0.42, 0.30, 0.52, 0.38]); ins.grid(alpha=.25)
for key, lab, col, ls in NAMES:
    if key not in runs: continue
    R, Pc, au, ap = curves(key); band(axr, fg, R, col, ls, f"{lab}  (AUC {au.mean():.3f} ± {au.std(ddof=1):.3f})"); band(ins, fg, R, col, ls, ""); band(axp, fg, Pc, col, ls, f"{lab}  (AP {ap.mean():.3f})")
axr.plot([0, 1], [0, 1], "k:", lw=.8); axr.set(xlabel="False positive rate", ylabel="True positive rate", title=f"(a) ROC: mean ± 1 SD over {n_splits} splits", xlim=(0, 1), ylim=(0, 1.01))
zx, zy = ((0, .15), (.8, 1.005)) if which == "qso" else ((0, .30), (.5, 1.005)); ins.set(xlim=zx, ylim=zy); ins.tick_params(labelsize=8); ins.set_title(f"zoom: FPR {zx[0]}-{zx[1]}", fontsize=8)
axr.legend(loc="lower right", fontsize=7.5, bbox_to_anchor=(1.0, 0.0))
axp.set(xlabel="Recall", ylabel="Precision", title=f"(b) Precision-recall: mean ± 1 SD", xlim=(0, 1), ylim=(.5, 1.01)); axp.legend(loc="lower left", fontsize=7.5)
# (c) confusion matrix of the gated model, mean counts per split at 0.5
cm = np.mean([confusion_matrix(y[te], (p >= .5).astype(int), labels=[0, 1]) for s, (te, p) in runs[GATED].items()], axis=0); pct = cm / cm.sum(1, keepdims=True) * 100
axc.imshow(pct, cmap="Blues", vmin=0, vmax=100); axc.grid(False)
for i in range(2):
    for j in range(2): axc.text(j, i, f"{cm[i, j]:.0f}\n({pct[i, j]:.1f}%)", ha="center", va="center", fontsize=13, color="white" if pct[i, j] > 55 else "black")
cls = ("Galaxy", "QSO") if which == "qso" else ("Star-forming", "AGN")
axc.set_xticks([0, 1]); axc.set_xticklabels(cls); axc.set_yticks([0, 1]); axc.set_yticklabels(cls); axc.set(xlabel="Predicted", ylabel="True", title=f"(c) Gated attention, threshold 0.5\n(mean count per split; % of true class)")
# (d) reliability: equal-count bins per split, mean ± SD over splits
for key, lab, col, ls in (NAMES[0], NAMES[3]):
    xs, ys = [], []
    for s, (te, p) in sorted(runs[key].items()):
        o = np.argsort(p); b = np.array_split(o, 8); xs.append([p[i].mean() for i in b]); ys.append([y[te][i].mean() for i in b])
    xs, ys = np.array(xs), np.array(ys); ece = np.mean([np.sum([len(i) / len(p) * abs(y[te][i].mean() - p[i].mean()) for i in np.array_split(np.argsort(p), 10)]) for s, (te, p) in runs[key].items()])
    axl.errorbar(xs.mean(0), ys.mean(0), yerr=ys.std(0, ddof=1), color=col, marker="o", ms=5, capsize=2.5, lw=1.5, label=f"{lab}  (ECE {ece:.3f})")
axl.plot([0, 1], [0, 1], "k:", lw=.8); axl.set(xlabel="Predicted probability (equal-count bins)", ylabel="Observed fraction of positives", title="(d) Reliability: mean ± SD over splits", xlim=(-.02, 1.02), ylim=(-.02, 1.02)); axl.legend(loc="upper left", fontsize=8)
fig.suptitle(TITLE, fontsize=12, y=.995); plt.tight_layout(); plt.savefig(f"{OUT}/fig_main_{which}.png"); plt.close()
# ---------------- confusion matrices of three models side by side
fig, ax = plt.subplots(1, 3, figsize=(12, 4.2))
for a, (key, lab) in zip(ax, ((IMG, "Image only"), (TAB, "XGBoost, 12 features"), (GATED, "Gated attention, 12 features"))):
    cm = np.mean([confusion_matrix(y[te], (p >= .5).astype(int), labels=[0, 1]) for s, (te, p) in runs[key].items()], axis=0); pct = cm / cm.sum(1, keepdims=True) * 100
    a.imshow(pct, cmap="Blues", vmin=0, vmax=100); a.grid(False)
    for i in range(2):
        for j in range(2): a.text(j, i, f"{cm[i, j]:.0f}\n({pct[i, j]:.1f}%)", ha="center", va="center", fontsize=12, color="white" if pct[i, j] > 55 else "black")
    a.set_xticks([0, 1]); a.set_xticklabels(cls); a.set_yticks([0, 1]); a.set_yticklabels(cls); a.set(xlabel="Predicted", ylabel="True", title=lab)
fig.suptitle(f"{TITLE}: confusion matrices at threshold 0.5 (mean count per split)", fontsize=11); plt.tight_layout(); plt.savefig(f"{OUT}/fig_confusion_{which}.png"); plt.close(); print("saved to", OUT)