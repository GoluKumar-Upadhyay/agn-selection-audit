"""External validation on an independently selected X-ray (or other) cohort WITH negative controls.

Input : a CSV with columns  objid, label (1 = X-ray/independently confirmed AGN, 0 = control),
        subtype (e.g. Sy1, Sy2, LINER, SF, quiescent), redshift, score  (model P(AGN) for that object)
Output: external AUC with DeLong CI, sensitivity at fixed specificity (and vice versa), subtype-stratified recall,
        calibration (Brier/ECE) and a bootstrap CI for every number.

Positive-only lists (like the original 10 'Seyferts') CANNOT estimate specificity or AUC; this script refuses to
report AUC unless both classes are present and each has >= 20 objects.

Tested on synthetic data via `python external_validation.py --selftest`.
How to build the cohort (not done here): cross-match SDSS spectroscopic objects of ANY spec class with 4XMM-DR13 /
Chandra CSC 2.0 / Swift-BAT; positives: L_X(2-10 keV) > 1e42 erg/s; negatives: SDSS galaxies inside the same X-ray
footprint with an X-ray upper limit below that luminosity at their redshift (no detection alone is NOT enough -
shallow exposure would label real AGN as negatives).  Make sure none of them is in the training table (merge on objid).
"""
import argparse, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from stats import delong_ci, bootstrap_auc, calibration_summary
from sklearn.metrics import roc_curve


def sens_at_spec(y, s, spec=0.95):
    f, t, _ = roc_curve(y, s)
    return float(t[np.searchsorted(f, 1 - spec, side="right") - 1])


def spec_at_sens(y, s, sens=0.95):
    f, t, _ = roc_curve(y, s)
    return float(1 - f[np.searchsorted(t, sens, side="left")])


def boot(fn, y, s, n=1000, seed=0):
    rng = np.random.default_rng(seed); v = []
    for _ in range(n):
        b = rng.integers(0, len(y), len(y))
        if y[b].min() != y[b].max(): v.append(fn(y[b], s[b]))
    return np.percentile(v, [2.5, 97.5])


def evaluate(df, thr=0.5):
    y, s = df.label.values.astype(int), df.score.values
    if min((y == 1).sum(), (y == 0).sum()) < 20:
        raise ValueError("Need >=20 positives AND >=20 negatives; a positive-only list cannot give AUC/specificity.")
    auc, lo, hi = delong_ci(y, s)
    out = {"n_pos": int((y == 1).sum()), "n_neg": int((y == 0).sum()), "AUC": auc, "AUC_CI": (lo, hi),
           "sens@spec95": sens_at_spec(y, s, .95), "sens@spec95_CI": tuple(boot(lambda a, b: sens_at_spec(a, b, .95), y, s)),
           "spec@sens95": spec_at_sens(y, s, .95), "spec@sens95_CI": tuple(boot(lambda a, b: spec_at_sens(a, b, .95), y, s)),
           **calibration_summary(y, s)}
    rec = {}
    for st, g in df[df.label == 1].groupby("subtype"):
        rec[st] = dict(n=len(g), recall=float((g.score > thr).mean()))
    out["recall_by_subtype"] = rec
    out["specificity_by_control_type"] = {st: dict(n=len(g), specificity=float((g.score <= thr).mean()))
                                          for st, g in df[df.label == 0].groupby("subtype")}
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--cohort"); ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        rng = np.random.default_rng(0); n = 120
        d = pd.DataFrame({"label": np.r_[np.ones(n), np.zeros(n)].astype(int),
                          "subtype": np.r_[rng.choice(["Sy1", "Sy2", "LINER"], n), rng.choice(["SF", "quiescent"], n)],
                          "score": np.clip(np.r_[rng.normal(.8, .2, n), rng.normal(.3, .25, n)], 0, 1)})
        print(evaluate(d)); sys.exit()
    print(evaluate(pd.read_csv(a.cohort)))
