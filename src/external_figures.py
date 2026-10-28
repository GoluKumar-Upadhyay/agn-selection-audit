"""Figure 8 (external validation panel): ROC+CI band, PR, confusion matrix at a PRE-SPECIFIED threshold, calibration.
Input CSV: label, score.  Selftest: python external_figures.py --selftest"""
import sys, os, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, confusion_matrix
def make(df, out, thr=0.5):
    y, s = df.label.values, df.score.values; rng = np.random.default_rng(0); g = np.linspace(0, 1, 101); b = []
    for _ in range(300):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max(): f, t, _ = roc_curve(y[i], s[i]); b.append(np.interp(g, f, t))
    fig, ax = plt.subplots(2, 2, figsize=(9, 8)); f, t, _ = roc_curve(y, s)
    ax[0, 0].plot(f, t, label=f"AUC {roc_auc_score(y, s):.3f}"); ax[0, 0].fill_between(g, *np.percentile(b, [2.5, 97.5], 0), alpha=.2); ax[0, 0].legend(); ax[0, 0].set_title("(a) ROC")
    p, r, _ = precision_recall_curve(y, s); ax[0, 1].plot(r, p); ax[0, 1].set_title("(b) Precision-recall")
    cm = confusion_matrix(y, (s >= thr).astype(int)); ax[1, 0].imshow(cm, cmap="Blues"); ax[1, 0].set_title(f"(c) Confusion @ {thr}")
    for i in range(2):
        for j in range(2): ax[1, 0].text(j, i, cm[i, j], ha="center", va="center")
    bins = np.linspace(0, 1, 11); k = np.digitize(s, bins[1:-1]); xs = [s[k == q].mean() for q in range(10) if (k == q).sum() > 3]; ys = [y[k == q].mean() for q in range(10) if (k == q).sum() > 3]
    ax[1, 1].plot(xs, ys, "o-"); ax[1, 1].plot([0, 1], [0, 1], "k:"); ax[1, 1].set_title("(d) Calibration"); plt.tight_layout(); plt.savefig(out, dpi=200)
if __name__ == "__main__":
    if "--selftest" in sys.argv:
        r = np.random.default_rng(1); d = pd.DataFrame({"label": np.r_[np.ones(100), np.zeros(100)].astype(int), "score": np.clip(np.r_[r.normal(.75, .2, 100), r.normal(.3, .2, 100)], 0, 1)}); make(d, "/tmp/fig8_selftest.png"); print("ok")
    else: make(pd.read_csv(sys.argv[1]), "../figures/paper/fig8_external.png")
