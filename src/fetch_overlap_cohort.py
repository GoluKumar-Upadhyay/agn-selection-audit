"""Build a redshift- and magnitude-OVERLAPPING SDSS DR17 cohort (positivity by construction).

STATUS: written for you to run - NOT executed in my sandbox (no route to skyserver.sdss.org).
Check table/column names in the SkyServer schema browser before the first run.

Why this replaces 01_fetch_sdss_data.py
  * `SELECT TOP N` without ORDER BY returns an arbitrary database slice, not a random sample.
    In the original table the QSO redshifts sit in a few discrete bands and sky position alone
    gives AUC=0.64 (see results/tabular_audit.json).
  * The original galaxies have z<=0.030 while QSOs reach z=0.484: only 24/4958 QSOs lie inside the
    galaxy redshift support, so no redshift-matched analysis is possible.
Design here
  * Strata = (spectroscopic class group) x (redshift bin) x (r-mag bin).  The SAME number of objects is
    requested per class in every cell -> z and r distributions are identical across classes by design.
  * Pseudo-random, reproducible sampling with a hash on specObjID (not on database order).
  * Three positive/negative groups:  QSO (broad-line, spec class QSO)  |  NLAGN (class GALAXY with
    MPA-JHU BPT class AGN, i.e. narrow-line / obscured AGN)  |  SF/quiescent control (class GALAXY, BPT star-forming).
    Labels for the main task: positive = QSO or NLAGN, negative = SF.  Sub-type kept for stratified reporting.
"""
import argparse, io, time
import numpy as np, pandas as pd, requests

URL = "https://skyserver.sdss.org/dr17/SkyServerWS/SearchTools/SqlSearch"
Z_BINS = [0.03, 0.06, 0.10, 0.15, 0.20, 0.30, 0.40, 0.55]
R_BINS = [15.0, 16.5, 17.5, 18.5, 19.5, 20.5, 21.5]
BASE = """
SELECT TOP {n} p.objID AS objid, p.ra, p.dec,
       p.modelMag_u AS u, p.modelMag_g AS g, p.modelMag_r AS r, p.modelMag_i AS i, p.modelMag_z AS z,
       p.extinction_u, p.extinction_g, p.extinction_r, p.extinction_i, p.extinction_z,
       p.petroR50_r, p.petroR90_r, p.psfMag_r, p.deVAB_r, p.expAB_r, p.fracDeV_r,
       s.z AS redshift, s.class AS spec_class, s.subclass, s.snMedian, s.specObjID AS specobjid
       {extra_select}
FROM SpecObj AS s
JOIN PhotoObj AS p ON p.objID = s.bestObjID
{extra_join}
WHERE s.sciencePrimary = 1 AND s.zWarning = 0 AND p.mode = 1
  AND s.z BETWEEN {z0} AND {z1} AND p.modelMag_r BETWEEN {r0} AND {r1}
  AND {class_cond}
  AND (s.specObjID % 10007) < {hash_cut}
"""
GROUPS = {
    "QSO":   dict(class_cond="s.class = 'QSO'", extra_select="", extra_join=""),
    # MPA-JHU emission-line classification: verify the numeric BPT codes in the schema browser
    "NLAGN": dict(class_cond="s.class = 'GALAXY' AND x.bptclass = 4",
                  extra_select=", x.bptclass", extra_join="JOIN galSpecExtra AS x ON x.specObjID = s.specObjID"),
    "SF":    dict(class_cond="s.class = 'GALAXY' AND x.bptclass = 1",
                  extra_select=", x.bptclass", extra_join="JOIN galSpecExtra AS x ON x.specObjID = s.specObjID"),
}


def run_sql(q, retries=3):
    for k in range(retries):
        try:
            r = requests.get(URL, params={"cmd": q, "format": "csv"}, timeout=180)
            if r.status_code == 200 and "objid" in r.text[:2000].lower():
                return pd.read_csv(io.StringIO(r.text), comment="#")
        except Exception as e:
            print("retry", k, e)
        time.sleep(5)
    return pd.DataFrame()


def main(per_cell, hash_cut, out):
    parts = []
    for gi, (gname, spec) in enumerate(GROUPS.items()):
        for zi in range(len(Z_BINS) - 1):
            for ri in range(len(R_BINS) - 1):
                q = BASE.format(n=per_cell * 3, z0=Z_BINS[zi], z1=Z_BINS[zi + 1], r0=R_BINS[ri], r1=R_BINS[ri + 1],
                                hash_cut=hash_cut, **spec)
                d = run_sql(q)
                if len(d):
                    d["group"], d["zbin"], d["rbin"] = gname, zi, ri
                    parts.append(d)
                print(gname, zi, ri, len(d), flush=True)
    df = pd.concat(parts, ignore_index=True).drop_duplicates("objid")
    # --- enforce matching: in every (zbin,rbin) cell keep the same number of positives and negatives
    rng = np.random.default_rng(0)
    df["label"] = df.group.isin(["QSO", "NLAGN"]).astype(int)
    keep = []
    for (zb, rb), c in df.groupby(["zbin", "rbin"]):
        pos, neg = c.index[c.label == 1], c.index[c.label == 0]
        n = min(len(pos), len(neg))
        if n >= 5:
            keep += list(rng.choice(pos, n, replace=False)) + list(rng.choice(neg, n, replace=False))
    m = df.loc[keep].reset_index(drop=True)
    m.to_csv(out, index=False)
    print(m.groupby(["group"]).size(), "\nsaved", out, m.shape)
    # --- positivity diagnostics
    from stats import overlap_coefficient
    for c in ["redshift", "r"]:
        print(c, "overlap coeff:", overlap_coefficient(m[c][m.label == 1].values, m[c][m.label == 0].values))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-cell", type=int, default=60)
    ap.add_argument("--hash-cut", type=int, default=2500, help="keep rows with specObjID %% 10007 < cut (~25%% sample)")
    ap.add_argument("--out", default="../data/overlap_cohort.csv")
    a = ap.parse_args()
    main(a.per_cell, a.hash_cut, a.out)
