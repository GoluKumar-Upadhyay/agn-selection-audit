"""Statistical utilities: DeLong, bootstrap CIs, paired bootstrap, calibration, conformal."""
from __future__ import annotations
import numpy as np
from scipy import stats as sps
from sklearn.metrics import roc_auc_score, brier_score_loss


# ---------------------------------------------------------------- DeLong ----
def _midrank(x):
    j = np.argsort(x)
    z = x[j]
    n = len(x)
    t = np.zeros(n)
    i = 0
    while i < n:
        k = i
        while k < n and z[k] == z[i]:
            k += 1
        t[i:k] = 0.5 * (i + k - 1) + 1
        i = k
    out = np.empty(n)
    out[j] = t
    return out


def _fast_delong(preds_sorted, m):
    """preds_sorted: (k, n) with the m positives first (Sun & Xu 2014)."""
    k, n = preds_sorted.shape
    n_neg = n - m
    pos, neg = preds_sorted[:, :m], preds_sorted[:, m:]
    tx = np.array([_midrank(p) for p in pos])
    ty = np.array([_midrank(p) for p in neg])
    tz = np.array([_midrank(p) for p in preds_sorted])
    aucs = tz[:, :m].sum(axis=1) / m / n_neg - (m + 1.0) / 2.0 / n_neg
    v01 = (tz[:, :m] - tx) / n_neg
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx, sy = np.atleast_2d(np.cov(v01)), np.atleast_2d(np.cov(v10))
    return aucs, sx / m + sy / n_neg


def delong_roc_test(y, p1, p2):
    """Two-sided paired DeLong test, AUC(p1) vs AUC(p2) on the SAME samples.
    Returns (auc1, auc2, diff, z, p)."""
    y = np.asarray(y).astype(int)
    order = np.argsort(-y, kind="stable")
    m = int(y.sum())
    preds = np.vstack([np.asarray(p1)[order], np.asarray(p2)[order]])
    aucs, cov = _fast_delong(preds, m)
    diff = aucs[0] - aucs[1]
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    if var <= 0:
        return aucs[0], aucs[1], diff, np.nan, 1.0
    z = diff / np.sqrt(var)
    return aucs[0], aucs[1], diff, z, 2 * sps.norm.sf(abs(z))


def delong_ci(y, p, alpha=0.05):
    y = np.asarray(y).astype(int)
    order = np.argsort(-y, kind="stable")
    m = int(y.sum())
    aucs, cov = _fast_delong(np.asarray(p)[order][None, :], m)
    se = np.sqrt(cov[0, 0])
    z = sps.norm.ppf(1 - alpha / 2)
    return aucs[0], max(0.0, aucs[0] - z * se), min(1.0, aucs[0] + z * se)


# ------------------------------------------------------------- bootstrap ----
def bootstrap_auc(y, p, n_boot=1000, seed=0, alpha=0.05):
    y, p = np.asarray(y), np.asarray(p)
    rng = np.random.default_rng(seed)
    vals, n = [], len(y)
    for _ in range(n_boot):
        b = rng.integers(0, n, n)
        if y[b].min() == y[b].max():
            continue
        vals.append(roc_auc_score(y[b], p[b]))
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return roc_auc_score(y, p), lo, hi


def paired_bootstrap_diff(y, p1, p2, n_boot=1000, seed=0):
    """Bootstrap mean/CI/p for AUC(p1)-AUC(p2) on the same objects."""
    y, p1, p2 = map(np.asarray, (y, p1, p2))
    rng = np.random.default_rng(seed)
    n, d = len(y), []
    for _ in range(n_boot):
        b = rng.integers(0, n, n)
        if y[b].min() == y[b].max():
            continue
        d.append(roc_auc_score(y[b], p1[b]) - roc_auc_score(y[b], p2[b]))
    d = np.array(d)
    p = 2 * min((d <= 0).mean(), (d >= 0).mean())
    return d.mean(), np.percentile(d, 2.5), np.percentile(d, 97.5), min(1.0, p)


# ----------------------------------------------------------- calibration ----
def expected_calibration_error(y, p, n_bins=10):
    y, p = np.asarray(y), np.asarray(p)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (p >= lo) & ((p < hi) if hi < 1 else (p <= hi))
        if m.sum():
            ece += m.mean() * abs(y[m].mean() - p[m].mean())
    return ece


def calibration_summary(y, p):
    return dict(brier=brier_score_loss(y, p), ece=expected_calibration_error(y, p))


# ------------------------------------------------------------- conformal ----
def split_conformal_threshold(p_cal, y_cal, alpha=0.1):
    """Class-conditional split conformal. Score = 1 - P(true class). Returns {0:q0, 1:q1}."""
    q = {}
    for c in (0, 1):
        pc = p_cal[y_cal == c]
        s = 1 - (pc if c == 1 else 1 - pc)
        n = len(s)
        level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
        q[c] = np.quantile(s, level, method="higher")
    return q


def conformal_sets(p, q):
    """(n,2) boolean: is class c in the prediction set?"""
    return np.stack([p <= q[0], (1 - p) <= q[1]], axis=1)


def conformal_coverage(sets, y):
    return sets[np.arange(len(y)), y.astype(int)].mean()


def overlap_coefficient(a, b, bins=60):
    lo, hi = min(a.min(), b.min()), max(a.max(), b.max())
    ha, _ = np.histogram(a, bins=bins, range=(lo, hi))
    hb, _ = np.histogram(b, bins=bins, range=(lo, hi))
    return np.minimum(ha / ha.sum(), hb / hb.sum()).sum()
