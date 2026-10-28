"""Selection-shortcut audit on the tabular SDSS table (runs on CPU in a few minutes).

Produces results/*.csv|json and figures/*.png.  Every number in the revised manuscript's
tabular sections comes from this script.
"""
from __future__ import annotations
import json, os, sys, time, warnings
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, r2_score
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.isotonic import IsotonicRegression
from scipy.stats import spearmanr
sys.path.insert(0, os.path.dirname(__file__))
from data import *
from stats import *
warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES, FIG = os.path.join(ROOT, "results"), os.path.join(ROOT, "figures")
SEEDS = [42, 43, 44, 45, 46]
t0 = time.time()
log = lambda *a: print(f"[{time.time()-t0:6.1f}s]", *a, flush=True)


# --------------------------------------------------------------- helpers ----
def fit_xgb(Xtr, ytr, Xva, yva, depth=6, lr=0.05, n=500, seed=42, spw=None):
    m = xgb.XGBClassifier(n_estimators=n, max_depth=depth, learning_rate=lr, subsample=0.8,
                          colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
                          eval_metric="logloss", early_stopping_rounds=50, random_state=seed,
                          n_jobs=1, scale_pos_weight=spw if spw else (ytr == 0).sum() / (ytr == 1).sum())
    m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    return m


def fit_lr(Xtr, ytr):
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")).fit(Xtr, ytr)


def best_threshold_youden(score, y):
    from sklearn.metrics import roc_curve
    f, t, th = roc_curve(y, score)
    return th[np.argmax(t - f)]


def run_models(df, tr, va, te, seed, feats_dict, which):
    """Return dict name -> test probability/score (np array)."""
    y = df.label.values
    out = {}
    for name in which:
        cols = feats_dict[name]
        X = df[cols].values
        m = fit_xgb(X[tr], y[tr], X[va], y[va], seed=seed)
        out[f"XGB[{name}]"] = m.predict_proba(X[te])[:, 1]
    return out


# ------------------------------------------------------------------ main ----
df = load_clean(os.path.join(ROOT, "data", "sdss_agn_dataset.csv"))
y = df.label.values
log(f"clean table {df.shape}  QSO={y.sum()}  GAL={(y==0).sum()}  dropped={df.attrs['n_dropped']}")
R = {"n_total": int(len(df)), "n_qso": int(y.sum()), "n_gal": int((y == 0).sum())}

# ------------------------------------------------ 1. data-design diagnostics --
zq, zg = df.redshift[y == 1].values, df.redshift[y == 0].values
R["z_gal_max"], R["z_qso_max"] = float(zg.max()), float(zq.max())
R["z_gal_q"] = np.quantile(zg, [0, .5, .99, 1]).tolist()
R["z_qso_q"] = np.quantile(zq, [0, .1, .5, .9, 1]).tolist()
R["z_overlap_coeff"] = float(overlap_coefficient(zq, zg))
R["frac_qso_in_gal_z_support"] = float((zq <= zg.max()).mean())
R["n_qso_in_gal_z_support"] = int((zq <= zg.max()).sum())
R["frac_obj_z_gt_0.03_is_qso"] = float(df[df.redshift > 0.03].label.mean())
R["auc_redshift_alone"] = float(roc_auc_score(y, df.redshift))
R["sky_slice"] = {"ra_range_gal": [float(df[y==0].ra.min()), float(df[y==0].ra.max())],
                  "ra_range_qso": [float(df[y==1].ra.min()), float(df[y==1].ra.max())]}
ov = {}
for c in ["r", "u_g", "g_r", "concentration", "petroR50_r"]:
    ov[c] = float(overlap_coefficient(df[c][y == 1].values, df[c][y == 0].values))
R["overlap_coefficients"] = ov
log("overlap coefficients", ov, "z:", R["z_overlap_coeff"])

# ------------------------------- 2. classical + tabular baselines, 5 seeds ----
rows = []
detail = {}
for seed in SEEDS:
    tr, va, te = split_70_15_15(df, seed)
    yt = y[te]
    sc = {}
    # classical: UV-excess colour cut / morphology / colour LR / shallow tree on colours
    sc["Classical: UV-excess (u-g)"] = -df.u_g.values[te]
    sc["Classical: concentration (R50/R90)"] = -df.concentration.values[te]
    sc["Classical: Petrosian R50 (angular size)"] = -df.petroR50_r.values[te]
    lr = fit_lr(df[["u_g", "g_r", "r_i", "i_z"]].values[tr], y[tr])
    sc["Classical: colour-colour LR (ugriz colours)"] = lr.predict_proba(df[["u_g", "g_r", "r_i", "i_z"]].values[te])[:, 1]
    lr2 = fit_lr(df[["u_g", "g_r", "r_i", "i_z", "concentration"]].values[tr], y[tr])
    sc["Classical: colours + concentration LR"] = lr2.predict_proba(df[["u_g", "g_r", "r_i", "i_z", "concentration"]].values[te])[:, 1]
    dt = DecisionTreeClassifier(max_depth=3, class_weight="balanced", random_state=seed).fit(df[COLOURS + MORPH].values[tr], y[tr])
    sc["Classical: depth-3 tree (colours+conc.)"] = dt.predict_proba(df[COLOURS + MORPH].values[te])[:, 1]
    sc["Shortcut: redshift threshold"] = df.redshift.values[te]
    lr3 = fit_lr(df[["redshift"]].values[tr], y[tr])
    sc["Shortcut: redshift LR"] = lr3.predict_proba(df[["redshift"]].values[te])[:, 1]
    sc.update(run_models(df, tr, va, te, seed, FEATURE_SETS,
                         ["full_14", "no_redshift", "no_z_no_Mr", "colours_only", "morphology_only", "shortcut_only"]))
    for k, s in sc.items():
        rows.append(dict(seed=seed, model=k, auc=roc_auc_score(yt, s)))
    if seed == 42:
        detail = {"tr": tr, "va": va, "te": te, "scores": sc}
        # paper-config XGB with SMOTE (what the original code did)  -> reproduce 0.9998?
        from imblearn.over_sampling import SMOTE
        Xf = df[ALL_FEATURES].values
        sm = StandardScaler().fit(Xf[tr])
        Xs, ys = SMOTE(random_state=seed).fit_resample(sm.transform(Xf[tr]), y[tr])
        m = fit_xgb(Xs, ys, sm.transform(Xf[va]), y[va], seed=seed, spw=1.0)
        p = m.predict_proba(sm.transform(Xf[te]))[:, 1]
        rows.append(dict(seed=seed, model="XGB[full_14] + SMOTE (original recipe)", auc=roc_auc_score(yt, p)))
        detail["scores"]["XGB[full_14] + SMOTE (original recipe)"] = p
        # MLP proxy (sklearn) full vs no_z_no_Mr
        for nm in ["full_14", "no_z_no_Mr"]:
            Xn = StandardScaler().fit(df[FEATURE_SETS[nm]].values[tr])
            mlp = MLPClassifier((256, 128, 64), early_stopping=True, random_state=seed, max_iter=200)
            mlp.fit(Xn.transform(df[FEATURE_SETS[nm]].values[tr]), y[tr])
            p = mlp.predict_proba(Xn.transform(df[FEATURE_SETS[nm]].values[te]))[:, 1]
            rows.append(dict(seed=seed, model=f"MLP[{nm}]", auc=roc_auc_score(yt, p)))
            detail["scores"][f"MLP[{nm}]"] = p
    log("seed", seed, "done")

A = pd.DataFrame(rows)
summ = A.groupby("model").auc.agg(["mean", "std", "min", "max", "count"]).sort_values("mean", ascending=False)
summ.to_csv(os.path.join(RES, "tab_baselines_multiseed.csv"))
log("\n" + summ.round(4).to_string())

# seed-42 detailed table with DeLong CIs + accuracy/F1-type at Youden threshold (val-free: train thr)
te = detail["te"]; yt = y[te]; tbl = []
for k, s in detail["scores"].items():
    a, lo, hi = delong_ci(yt, s)
    cal = calibration_summary(yt, s) if (s.min() >= 0 and s.max() <= 1) else dict(brier=np.nan, ece=np.nan)
    tbl.append(dict(model=k, auc=a, ci_lo=lo, ci_hi=hi, **cal))
T42 = pd.DataFrame(tbl).sort_values("auc", ascending=False)
T42.to_csv(os.path.join(RES, "tab_seed42_delong_ci.csv"), index=False)
R["n_test_seed42"] = int(len(te))

# paired DeLong between key models
pairs = [("XGB[full_14]", "XGB[no_z_no_Mr]"), ("XGB[full_14]", "Shortcut: redshift threshold"),
         ("XGB[no_z_no_Mr]", "Classical: colours + concentration LR"),
         ("XGB[no_z_no_Mr]", "Classical: UV-excess (u-g)"), ("XGB[full_14]", "XGB[shortcut_only]")]
P = []
for a, b in pairs:
    a1, a2, d, z, p = delong_roc_test(yt, detail["scores"][a], detail["scores"][b])
    P.append(dict(model_a=a, model_b=b, auc_a=a1, auc_b=a2, diff=d, z=z, p=p))
pd.DataFrame(P).to_csv(os.path.join(RES, "tab_paired_delong.csv"), index=False)
log("paired DeLong:\n" + pd.DataFrame(P).round(4).to_string())

# ------------------------------------- 3. proxy leakage: can z be rebuilt? ----
nz_cols = FEATURE_SETS["no_z_no_Mr"] + ["petroR50_r", "petroR90_r"]
zhat = cross_val_predict(xgb.XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05, n_jobs=1),
                         df[nz_cols].values, df.redshift.values, cv=5)
R["proxy_z_from_nonz_R2"] = float(r2_score(df.redshift, zhat))
R["proxy_z_from_nonz_spearman"] = float(spearmanr(df.redshift, zhat)[0])
# class AUC achieved by the *reconstructed* z alone
R["auc_class_from_reconstructed_z"] = float(roc_auc_score(y, zhat))
log("z reconstructed from non-z features: R2=%.3f rho=%.3f AUC(z_hat)=%.4f" %
    (R["proxy_z_from_nonz_R2"], R["proxy_z_from_nonz_spearman"], R["auc_class_from_reconstructed_z"]))

# ------------------------------------------- 4. shortcut-peeling cohorts ----
def coarsened_match(df, keys_bins, seed=0):
    """Coarsened exact matching: in each stratum keep min(n_qso, n_gal) of each class."""
    rng = np.random.default_rng(seed)
    key = pd.Series(0, index=df.index)
    for col, w in keys_bins:
        key = key.astype(str) + "_" + np.floor(df[col] / w).astype(int).astype(str)
    keep = []
    for k, g in df.groupby(key):
        q, gl = g.index[g.label == 1], g.index[g.label == 0]
        n = min(len(q), len(gl))
        if n:
            keep += list(rng.choice(q, n, replace=False)) + list(rng.choice(gl, n, replace=False))
    return df.loc[keep].reset_index(drop=True)

df["log_r50"] = np.log10(df.petroR50_r.clip(lower=0.05))
cohorts = {
    "Original (unmatched)": df,
    "Matched on r (0.25 mag)": coarsened_match(df, [("r", 0.25)]),
    "Matched on r + angular size": coarsened_match(df, [("r", 0.5), ("log_r50", 0.1)]),
    "Matched on r + size + u-g": coarsened_match(df, [("r", 0.5), ("log_r50", 0.1), ("u_g", 0.2)]),
}
peel = []
for cname, cdf in cohorts.items():
    ycd = cdf.label.values
    res = {"cohort": cname, "n": len(cdf), "n_qso": int(ycd.sum())}
    for fs in ["full_14", "no_z_no_Mr", "colours_only"]:
        aucs = []
        for seed in SEEDS:
            tr_, va_, te_ = split_70_15_15(cdf, seed)
            X = cdf[FEATURE_SETS[fs]].values
            m = fit_xgb(X[tr_], ycd[tr_], X[va_], ycd[va_], seed=seed)
            aucs.append(roc_auc_score(ycd[te_], m.predict_proba(X[te_])[:, 1]))
        res[f"auc_{fs}"], res[f"sd_{fs}"] = float(np.mean(aucs)), float(np.std(aucs))
    res["auc_r_alone"] = float(roc_auc_score(ycd, cdf.r))
    res["auc_size_alone"] = float(roc_auc_score(ycd, -cdf.petroR50_r))
    res["auc_z_alone"] = float(roc_auc_score(ycd, cdf.redshift))
    peel.append(res)
    log("cohort", cname, {k: round(v, 4) if isinstance(v, float) else v for k, v in res.items()})
pd.DataFrame(peel).to_csv(os.path.join(RES, "tab_peeling_cohorts.csv"), index=False)

# ---------------------------------------------- 5. negative controls ----------
tr, va, te = detail["tr"], detail["va"], detail["te"]
nc = {}
Xf = df[ALL_FEATURES].values
rng = np.random.default_rng(0)
yperm = rng.permutation(y[tr])
m = fit_xgb(Xf[tr], yperm, Xf[va], rng.permutation(y[va]), seed=0)
nc["label_permutation_AUC"] = float(roc_auc_score(y[te], m.predict_proba(Xf[te])[:, 1]))
# galaxy-vs-galaxy pseudo task (random halves of the galaxy class): should be 0.5
g = df[df.label == 0].reset_index(drop=True)
gy = (rng.random(len(g)) < 0.33).astype(int)
gtr, gva, gte = split_70_15_15(g.assign(label=gy), 0)
Xg = g[ALL_FEATURES].values
m = fit_xgb(Xg[gtr], gy[gtr], Xg[gva], gy[gva], seed=0)
nc["galaxy_vs_galaxy_pseudo_AUC"] = float(roc_auc_score(gy[gte], m.predict_proba(Xg[gte])[:, 1]))
# sky position only (detects database-slice artefacts of un-ordered TOP-N queries)
Xp = df[["ra", "dec"]].values
m = fit_xgb(Xp[tr], y[tr], Xp[va], y[va], seed=0)
nc["sky_position_only_AUC"] = float(roc_auc_score(y[te], m.predict_proba(Xp[te])[:, 1]))
Xo = df[["objid"]].values.astype(float)
m = fit_xgb(Xo[tr], y[tr], Xo[va], y[va], seed=0)
nc["objid_only_AUC"] = float(roc_auc_score(y[te], m.predict_proba(Xo[te])[:, 1]))
R["negative_controls"] = nc
log("negative controls", nc)

# ------------------------------------------------- 6. SHAP attribution -------
import shap
mf = fit_xgb(Xf[tr], y[tr], Xf[va], y[va], seed=42)
sv = shap.TreeExplainer(mf).shap_values(Xf[te][:1000])
imp = np.abs(sv).mean(0); imp = pd.Series(imp, index=ALL_FEATURES).sort_values(ascending=False)
share = (imp / imp.sum())
R["shap_share"] = share.round(4).to_dict()
R["shap_share_z_plus_Mr"] = float(share[["redshift", "abs_mag_r"]].sum())
Xn = df[FEATURE_SETS["no_z_no_Mr"]].values
mn = fit_xgb(Xn[tr], y[tr], Xn[va], y[va], seed=42)
svn = shap.TreeExplainer(mn).shap_values(Xn[te][:1000])
impn = pd.Series(np.abs(svn).mean(0), index=FEATURE_SETS["no_z_no_Mr"]).sort_values(ascending=False)
R["shap_share_no_z"] = (impn / impn.sum()).round(4).to_dict()
log("SHAP share (full):", share.head(5).round(3).to_dict(), "| z+Mr =", round(R["shap_share_z_plus_Mr"], 3))

# ----------------------- 7. Shortcut Reliance Index + subgroup AUCs ----------
auc_full = roc_auc_score(y[te], detail["scores"]["XGB[full_14]"])
auc_nz = roc_auc_score(y[te], detail["scores"]["XGB[no_z_no_Mr]"])
R["SRI_nz"] = float((auc_full - auc_nz) / (auc_full - 0.5))
pc = pd.DataFrame(peel).set_index("cohort")
R["SRI_by_cohort"] = {c: float((pc.loc[c, "auc_full_14"] - pc.loc[c, "auc_colours_only"]) / (pc.loc[c, "auc_full_14"] - 0.5))
                      for c in pc.index}
sub = []
dte = df.iloc[te].assign(p_full=detail["scores"]["XGB[full_14]"], p_nz=detail["scores"]["XGB[no_z_no_Mr]"])
for col, edges in [("r", [14, 16, 17, 18, 19, 20, 22]), ("redshift", [0, 0.03, 0.1, 0.2, 0.3, 0.5])]:
    dte["bin"] = pd.cut(dte[col], edges)
    for b, g_ in dte.groupby("bin", observed=True):
        if g_.label.nunique() == 2 and g_.label.sum() >= 5 and (g_.label == 0).sum() >= 5:
            sub.append(dict(var=col, bin=str(b), n=len(g_), n_qso=int(g_.label.sum()), n_gal=int((g_.label == 0).sum()),
                            auc_full=roc_auc_score(g_.label, g_.p_full), auc_no_z=roc_auc_score(g_.label, g_.p_nz)))
        else:
            sub.append(dict(var=col, bin=str(b), n=len(g_), n_qso=int(g_.label.sum()), n_gal=int((g_.label == 0).sum()),
                            auc_full=np.nan, auc_no_z=np.nan))
pd.DataFrame(sub).to_csv(os.path.join(RES, "tab_subgroup_auc.csv"), index=False)
log("subgroup AUC:\n" + pd.DataFrame(sub).round(4).to_string())

# ------------------- 8. calibration + conformal coverage under shift ---------
p_cal = mn.predict_proba(Xn[va])[:, 1]
# isotonic recalibration on the validation split
iso = IsotonicRegression(out_of_bounds="clip").fit(p_cal, y[va])
p_te_raw = detail["scores"]["XGB[no_z_no_Mr]"]
p_te_iso = iso.predict(p_te_raw)
R["calibration_no_z"] = {"raw": calibration_summary(y[te], p_te_raw), "isotonic": calibration_summary(y[te], p_te_iso)}
q = split_conformal_threshold(iso.predict(p_cal), y[va], alpha=0.10)
sets = conformal_sets(p_te_iso, q)
R["conformal"] = {"alpha": 0.10, "marginal_coverage": float(conformal_coverage(sets, y[te])),
                  "frac_singleton": float((sets.sum(1) == 1).mean()), "frac_both": float((sets.sum(1) == 2).mean()),
                  "frac_empty": float((sets.sum(1) == 0).mean())}
cov_rows = []
for col, edges in [("r", [14, 16, 17, 18, 19, 22]), ("petroR50_r", [0, 1.5, 2.5, 4, 6, 100])]:
    b = pd.cut(df.iloc[te][col], edges)
    for k in b.cat.categories:
        mk = (b == k).values
        if mk.sum() >= 30:
            cov_rows.append(dict(var=col, bin=str(k), n=int(mk.sum()),
                                 coverage=float(conformal_coverage(sets[mk], y[te][mk])),
                                 frac_both=float((sets[mk].sum(1) == 2).mean())))
# covariate shift: evaluate on the r-matched cohort (calibrated on original distribution)
mc = cohorts["Matched on r (0.25 mag)"]
Xm = mc[FEATURE_SETS["no_z_no_Mr"]].values
pm = iso.predict(mn.predict_proba(Xm)[:, 1])
sm_ = conformal_sets(pm, q)
cov_rows.append(dict(var="cohort", bin="r-matched cohort (shifted)", n=len(mc),
                     coverage=float(conformal_coverage(sm_, mc.label.values)), frac_both=float((sm_.sum(1) == 2).mean())))
mc2 = cohorts["Matched on r + angular size"]
pm2 = iso.predict(mn.predict_proba(mc2[FEATURE_SETS["no_z_no_Mr"]].values)[:, 1])
sm2 = conformal_sets(pm2, q)
cov_rows.append(dict(var="cohort", bin="r+size-matched cohort (shifted)", n=len(mc2),
                     coverage=float(conformal_coverage(sm2, mc2.label.values)), frac_both=float((sm2.sum(1) == 2).mean())))
pd.DataFrame(cov_rows).to_csv(os.path.join(RES, "tab_conformal_coverage.csv"), index=False)
log("conformal:", R["conformal"]); log("coverage by bin:\n" + pd.DataFrame(cov_rows).round(3).to_string())

# ----------------------------------------------------------- figures --------
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
# F1: redshift distributions + shortcut statement
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
bins = np.linspace(0, 0.5, 80)
ax[0].hist(zg, bins, alpha=.7, label=f"GALAXY (max z={zg.max():.3f})", color="#3b6fb6")
ax[0].hist(zq, bins, alpha=.7, label=f"QSO (max z={zq.max():.3f})", color="#d9534f")
ax[0].axvline(0.03, ls="--", c="k", lw=1); ax[0].set_yscale("log")
ax[0].set_xlabel("spectroscopic redshift"); ax[0].set_ylabel("count (log)"); ax[0].legend(frameon=False)
ax[0].set_title("(a) Class-conditional redshift support")
for c, a_, lab in [("r", ax[1], "r (mag)"), ("petroR50_r", ax[2], "Petrosian R50 (arcsec)")]:
    b = np.linspace(df[c].quantile(.001), df[c].quantile(.999), 60)
    a_.hist(df[c][y == 0], b, alpha=.7, color="#3b6fb6", label="GALAXY"); a_.hist(df[c][y == 1], b, alpha=.7, color="#d9534f", label="QSO")
    a_.set_xlabel(lab); a_.set_title(f"({'bc'[c!='r']}) overlap coeff = {ov[c]:.2f}"); a_.legend(frameon=False)
plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig_A_selection_design.png"), dpi=200); plt.close()

# F2: shortcut peeling
fig, ax = plt.subplots(figsize=(8.5, 4.6))
labs = list(pc.index); xs = np.arange(len(labs)); w = 0.26
for j, (fs, col, nm) in enumerate([("full_14", "#444", "XGB, all 14 features"), ("no_z_no_Mr", "#d9534f", "XGB, no z / no M_r"),
                                   ("colours_only", "#3b6fb6", "XGB, colours only")]):
    ax.bar(xs + (j - 1) * w, pc[f"auc_{fs}"], w, yerr=pc[f"sd_{fs}"], color=col, label=nm, capsize=2)
ax.axhline(0.5, c="gray", ls=":"); ax.set_ylim(0.45, 1.02)
ax.set_xticks(xs); ax.set_xticklabels([l.replace(" (", "\n(").replace(" + ", "\n+ ") for l in labs], fontsize=8)
ax.set_ylabel("test AUC (mean ± sd, 5 splits)"); ax.legend(frameon=False, loc="lower left", fontsize=8)
ax.set_title("Shortcut peeling: AUC as selection proxies are removed")
plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig_B_shortcut_peeling.png"), dpi=200); plt.close()

# F3: model comparison with CIs
Tm = T42.sort_values("auc")
fig, ax = plt.subplots(figsize=(8.5, 6))
colr = ["#d9534f" if "Shortcut" in m or "full_14" in m or "shortcut_only" in m else ("#999" if "Classical" in m else "#3b6fb6") for m in Tm.model]
ax.barh(Tm.model, Tm.auc, xerr=[Tm.auc - Tm.ci_lo, Tm.ci_hi - Tm.auc], color=colr, capsize=2)
ax.set_xlim(0.5, 1.005); ax.set_xlabel(f"test AUC (95% DeLong CI), seed 42, n={len(te)}"); ax.tick_params(axis="y", labelsize=8)
plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig_C_model_comparison.png"), dpi=200); plt.close()

# F4: conformal coverage
cv = pd.DataFrame(cov_rows)
fig, ax = plt.subplots(figsize=(9, 4))
ax.bar(range(len(cv)), cv.coverage, color=["#3b6fb6" if v != "cohort" else "#d9534f" for v in cv["var"]])
ax.axhline(0.9, ls="--", c="k"); ax.set_ylim(0.5, 1.02)
ax.set_xticks(range(len(cv))); ax.set_xticklabels([f"{a}\n{b}" for a, b in zip(cv["var"], cv.bin)], fontsize=7, rotation=45, ha="right")
ax.set_ylabel("empirical coverage (target 0.90)"); ax.set_title("Conformal coverage: marginal vs. conditional and shifted cohorts")
plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig_D_conformal.png"), dpi=200); plt.close()

# F5: SHAP shares
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
share.sort_values().plot.barh(ax=ax[0], color=["#d9534f" if i in SHORTCUT else "#3b6fb6" for i in share.sort_values().index])
ax[0].set_title("TreeSHAP share, all 14 features"); ax[0].set_xlabel("share of mean |SHAP|")
(impn / impn.sum()).sort_values().plot.barh(ax=ax[1], color="#3b6fb6"); ax[1].set_title("TreeSHAP share, z & M_r removed")
plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig_E_shap.png"), dpi=200); plt.close()

json.dump(R, open(os.path.join(RES, "tabular_audit.json"), "w"), indent=2, default=float)
log("ALL DONE")
