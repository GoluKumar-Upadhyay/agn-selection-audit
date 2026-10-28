"""Build Tables 1-7 (CSV + markdown) and Figures 1-7 from results/paper/*.  Environment INDIR/OUTDIR override paths."""
import os, sys, json, pickle, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from PIL import Image
from scipy import stats as sps
sys.path.insert(0, os.path.dirname(__file__))
from data import *; from stats import *
from sklearn.metrics import roc_curve, precision_recall_curve, average_precision_score, roc_auc_score
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN = os.environ.get("INDIR", f"{ROOT}/results/paper"); OUT = os.environ.get("OUTDIR", f"{ROOT}/results/paper_tables"); FIG = os.environ.get("FIGDIR", f"{ROOT}/figures/paper")
os.makedirs(OUT, exist_ok=True); os.makedirs(FIG, exist_ok=True)
R = pd.read_csv(f"{IN}/all_runs.csv"); PRED = pickle.load(open(f"{IN}/predictions.pkl", "rb")); EXTRA = pickle.load(open(f"{IN}/extra.pkl", "rb"))
cinfo = json.load(open(f"{IN}/cohort_info.json")); MAIN = "Joint r + size matched (MAIN)"
cdf = pd.read_csv(f"{ROOT}/data/image_cohort.csv"); y_all = cdf.label.values
seeds = sorted(R.seed.unique()); ns = len(seeds)
ms = lambda v, d=4: f"{np.mean(v):.{d}f} ± {np.std(v, ddof=1) if len(v) > 1 else 0:.{d}f}"
def get(cohort, name): return R[(R.cohort == cohort) & (R.model == name)].sort_values("seed")
def md(df, name):
    df.to_csv(f"{OUT}/{name}.csv", index=False)
    open(f"{OUT}/{name}.md", "w").write(df.to_markdown(index=False))

def paired(cohort, a, b):
    """per-seed paired DeLong on identical test objects -> (mean diff, #seeds a>b, #DeLong p<.05, wilcoxon p across seeds)"""
    d, ps = [], []
    for s in seeds:
        A, B = PRED[(cohort, s, a)], PRED[(cohort, s, b)]; assert (A["te"] == B["te"]).all()
        r = delong_roc_test(y_all[A["te"]], A["p"], B["p"]); d.append(r[2]); ps.append(r[4])
    d = np.array(d)
    try: w = sps.wilcoxon(d).pvalue
    except Exception: w = np.nan
    return d.mean(), int((d > 0).sum()), int((np.array(ps) < .05).sum()), w
fmtp = lambda t: f"{t[0]:+.4f} ({t[1]}/{ns} seeds; DeLong sig {t[2]}/{ns}; Wilcoxon p={t[3]:.3g})"

# ------------------------------------------------------------------ Table 1
rows1 = [("Colour-cut selector (u-g)", "Optical colour u-g"), ("Morphology rule", "Concentration / size"),
 ("Logistic regression [14]", "14 tabular"), ("Random forest [14]", "14 tabular"), ("SVM (RBF) [14]", "14 tabular"), ("XGBoost [14]", "14 tabular"),
 ("MLP [14]", "14 tabular"), ("XGBoost [12 shortcut-free]", "12 tabular (no z, no M_r)"), ("MLP [12 shortcut-free]", "12 tabular (no z, no M_r)"),
 ("EfficientNet-B0 (frozen, image only)", "64x64 cutout upsampled to 224"),
 ("Concatenation [14]", "Image + 14 tabular"), ("Cross-attention, no gate [14]", "Image + 14 tabular"), ("Gated attention [14]", "Image + 14 tabular"),
 ("Concatenation [12 shortcut-free]", "Image + 12 tabular"), ("Cross-attention, no gate [12 shortcut-free]", "Image + 12 tabular"), ("Gated attention [12 shortcut-free]", "Image + 12 tabular")]
t1 = []
for key, mod in rows1:
    g = R[(R.cohort == MAIN) & (R.model.str.startswith(key.split(" (")[0] if key.startswith("Morphology") or key.startswith("Colour") else key))] if False else None
    g = R[(R.cohort == MAIN) & ((R.model == key) | (key.startswith("Morphology rule") & R.model.str.startswith("Morphology rule")))]
    if key == "Morphology rule": g = R[(R.cohort == MAIN) & R.model.str.startswith("Morphology rule")]
    if not len(g): continue
    t1.append({"Model": key, "Input": mod, "AUC": ms(g.auc), "Accuracy (%)": ms(g.acc, 2), "Precision": ms(g.prec, 3), "Recall": ms(g.rec, 3), "F1": ms(g.f1, 3),
               "95% AUC CI (bootstrap, mean over seeds)": f"[{g.ci_lo.mean():.4f}, {g.ci_hi.mean():.4f}]",
               "Params / fit time (s)": (f"{int(g.params.iloc[0]):,}" if g.params.notna().any() else "n/a") + f" / {g.secs.mean():.1f}"})
md(pd.DataFrame(t1), "Table1_main_results")

# ------------------------------------------------------------------ Table 2
eff = "EfficientNet-B0 (frozen, image only)"; eff_auc = get(MAIN, eff).auc.values
t2 = []
for cfg in ["Full tabular features", "Without redshift", "Without luminosity proxy", "Without redshift + luminosity proxy", "Photometry + colours only", "Morphology proxies only"]:
    f = get(MAIN, f"Fusion | {cfg}").auc.values
    t2.append({"Feature configuration": cfg, "XGBoost AUC": ms(get(MAIN, f"XGB | {cfg}").auc), "MLP AUC": ms(get(MAIN, f"MLP | {cfg}").auc),
               "EfficientNet AUC": ms(eff_auc), "Fusion AUC": ms(f),
               "Fusion vs image-only": fmtp(paired(MAIN, f"Fusion | {cfg}", eff)), "Fusion vs XGBoost (same features)": fmtp(paired(MAIN, f"Fusion | {cfg}", f"XGB | {cfg}"))})
t2.append({"Feature configuration": "Image only", "XGBoost AUC": "—", "MLP AUC": "—", "EfficientNet AUC": ms(eff_auc), "Fusion AUC": "—", "Fusion vs image-only": "—", "Fusion vs XGBoost (same features)": "—"})
md(pd.DataFrame(t2), "Table2_shortcut_removal")

# ------------------------------------------------------------------ Table 3
t3 = []
for c in ["Original image cohort", "Magnitude-matched (r)", MAIN]:
    ci = cinfo[c]; L = get(c, "Logistic regression [colours]").auc
    t3.append({"Evaluation cohort": c, "n_QSO": ci["n_qso"], "n_galaxy": ci["n_gal"], "Image-only AUC": ms(get(c, eff).auc), "Tabular-only AUC (XGB, 12 shortcut-free)": ms(get(c, "XGBoost [12 shortcut-free]").auc),
               "Fusion AUC (gated, 12)": ms(get(c, "Gated attention [12 shortcut-free]").auc), "Classical AUC (colour LR)": ms(L)})
z = cinfo["z-support check"]
t3.append({"Evaluation cohort": "Redshift-matched (z within galaxy support)", "n_QSO": z["qso_inside_galaxy_z_support"], "n_galaxy": "all", "Image-only AUC": "INFEASIBLE", "Tabular-only AUC (XGB, 12 shortcut-free)": "INFEASIBLE",
           "Fusion AUC (gated, 12)": "INFEASIBLE", "Classical AUC (colour LR)": "INFEASIBLE"})
sh = EXTRA.get("shift", {})
t3.append({"Evaluation cohort": f"Objects discarded by matching (shifted); AUC", "n_QSO": "", "n_galaxy": "", "Image-only AUC": "", "Tabular-only AUC (XGB, 12 shortcut-free)": "",
           "Fusion AUC (gated, 12)": f"{sh.get('auc_rest', float('nan')):.4f} (n={sh.get('n_rest')}, seed {seeds[0]})", "Classical AUC (colour LR)": ""})
t3.append({"Evaluation cohort": f"Out-of-support QSOs (z>{z['galaxy_zmax']:.3f}); no AUC (no negatives)", "n_QSO": sh.get("n_qso_out_of_support", ""), "n_galaxy": 0, "Image-only AUC": "",
           "Tabular-only AUC (XGB, 12 shortcut-free)": "", "Fusion AUC (gated, 12)": f"mean P(AGN)={sh.get('mean_p_out_of_support', float('nan')):.3f}; detected={sh.get('frac_detected_out_of_support', float('nan')):.3f}", "Classical AUC (colour LR)": ""})
md(pd.DataFrame(t3), "Table3_matched_cohorts")

# ------------------------------------------------------------------ Table 4
ladder = [eff, "Concatenation [12 shortcut-free]", "Cross-attention, no gate [12 shortcut-free]", "Global-token + gate [12 shortcut-free]", "Gated attention [12 shortcut-free]"]
t4 = []
for k, name in enumerate(ladder):
    a = get(MAIN, name).auc.values; h = sps.t.ppf(.975, len(a) - 1) * a.std(ddof=1) / np.sqrt(len(a)) if len(a) > 1 else 0
    t4.append({"Configuration": name, "Mean test AUC ± SD": ms(a), "95% CI of mean (t, seeds)": f"[{a.mean()-h:.4f}, {a.mean()+h:.4f}]",
               "ΔAUC vs image-only": "—" if k == 0 else fmtp(paired(MAIN, name, ladder[0])), "ΔAUC vs previous": "—" if k == 0 else fmtp(paired(MAIN, name, ladder[k - 1]))})
md(pd.DataFrame(t4), "Table4_architecture_ablation")

# ------------------------------------------------------------------ Table 5
def op_table(name):
    out = []
    for s in seeds:
        P = PRED[(MAIN, s, name)]; yt, p, yv, pv = y_all[P["te"]], P["p"], y_all[P["va"]], P["pv"]
        f, t, _ = roc_curve(yt, p); pr, rc, _ = precision_recall_curve(yt, p); th = np.unique(pv)
        f1s = [(2 * ((pv >= q) & (yv == 1)).sum() / max(1, (pv >= q).sum() + (yv == 1).sum()), q) for q in th]; bq = max(f1s)[1]
        pred = (p >= bq).astype(int); tp = ((pred == 1) & (yt == 1)).sum()
        out.append(dict(brier=calibration_summary(yt, p)["brier"], ece=calibration_summary(yt, p)["ece"], auc=roc_auc_score(yt, p), ap=average_precision_score(yt, p),
                        sens5=float(np.interp(0.05, f, t)), pur90=float(pr[np.where(rc >= 0.9)[0][-1]]), thr=bq, f1=2 * tp / max(1, pred.sum() + yt.sum())))
    return pd.DataFrame(out)
t5 = []
for name in ["XGBoost [12 shortcut-free]", "XGBoost [14]", eff, "Concatenation [12 shortcut-free]", "Gated attention [12 shortcut-free]", "Gated attention [14]"]:
    o = op_table(name); t5.append({"Model": name, "Brier": ms(o.brier, 4), "ECE": ms(o.ece, 4), "AUC": ms(o.auc), "AP": ms(o.ap), "Sensitivity @ FPR=5%": ms(o.sens5, 3),
                                   "Purity @ 90% completeness": ms(o.pur90, 3), "Best-F1 threshold (val) → test F1": f"{o.thr.mean():.2f} → {o.f1.mean():.3f}"})
md(pd.DataFrame(t5), "Table5_calibration_operating_points")

# ------------------------------------------------------------------ Table 7
# Heuristic, NON-EXCLUSIVE error flags (an object can meet several rules; rules are photometric proxies, not confirmed physics).
name = "Gated attention [12 shortcut-free]"
flags = {"FN": [("Reddened / dusty-looking (u-g > 1.0)", lambda d: d.u_g > 1.0, "Optical-QSO prior; reddening", "WISE mid-IR features"),
                ("Host-dominated / extended (R50 > 2.5 arcsec)", lambda d: d.petroR50_r > 2.5, "Nuclear light diluted by host", "Larger cutouts, PSF modelling"),
                ("High redshift (z > 0.2)", lambda d: d.redshift > 0.2, "Limited resolution / faintness", "Deeper or higher-resolution imaging"),
                ("None of the above", None, "Unclassified", "Inspect individually")],
         "FP": [("Compact + red (R50 < 2 arcsec, g-r > 0.6)", lambda d: (d.petroR50_r < 2.0) & (d.g_r > 0.6), "Compact elliptical mimics a point source", "Sersic index, ellipticity"),
                ("Point-like + blue (R50 < 1.5 arcsec, g-r <= 0.6)", lambda d: (d.petroR50_r < 1.5) & (d.g_r <= 0.6), "Possible stellar/point-source contamination (NOT confirmed)", "Gaia cross-match, neighbour masking"),
                ("None of the above", None, "Unclassified", "Inspect individually")]}
acc = {k: {r[0]: [] for r in v} for k, v in flags.items()}; totals = {"FN": [], "FP": []}; good = {"TP": [], "TN": []}
for s_ in seeds:
    P = PRED[(MAIN, s_, name)]; d = cdf.iloc[P["te"]].reset_index(drop=True); p = P["p"]; yy = d.label.values
    for kind, mask in (("FN", (yy == 1) & (p < .5)), ("FP", (yy == 0) & (p >= .5))):
        dd = d[mask]; totals[kind].append(len(dd)); anyf = np.zeros(len(dd), bool)
        for nm_, fn, _, _ in flags[kind]:
            if fn is None: acc[kind][nm_].append(int((~anyf).sum()))
            else: m_ = fn(dd).values; anyf |= m_; acc[kind][nm_].append(int(m_.sum()))
    good["TP"].append(int(((yy == 1) & (p > .9)).sum())); good["TN"].append(int(((yy == 0) & (p < .1)).sum()))
rows7 = []
for kind, lab in (("FN", "False negative"), ("FP", "False positive")):
    tot = np.mean(totals[kind]); rows7.append({"Error category": f"{lab}s - TOTAL", "Mean count / split": f"{tot:.1f}", "Share of this error type": "1.00", "Likely cause": "", "Mitigation": ""})
    for nm_, fn, cause, mit in flags[kind]:
        c_ = np.mean(acc[kind][nm_]); rows7.append({"Error category": f"  {lab}: {nm_}", "Mean count / split": f"{c_:.1f}", "Share of this error type": f"{c_/tot:.2f}" if tot else "-", "Likely cause": cause, "Mitigation": mit})
rows7.append({"Error category": "Correct high-confidence QSO (P > 0.9)", "Mean count / split": f"{np.mean(good['TP']):.1f}", "Share of this error type": "-", "Likely cause": "", "Mitigation": ""})
rows7.append({"Error category": "Correct galaxy rejection (P < 0.1)", "Mean count / split": f"{np.mean(good['TN']):.1f}", "Share of this error type": "-", "Likely cause": "", "Mitigation": ""})
md(pd.DataFrame(rows7), "Table7_error_taxonomy")
print("tables done")

# ============================================================== FIGURES ====
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
# Fig 1 workflow
fig, ax = plt.subplots(figsize=(13, 3.2)); ax.axis("off")
steps = ["SDSS DR17\nquery", "QC / cleaning\n(sentinels, dups)", "Original\ncohort", "Matched cohort\n(r, size)", "Model training\n10 seeds", "Shortcut\nablation", "Calibration\n+ conformal", "External X-ray\nvalidation\n(pending)", "Error\ntaxonomy"]
cols = ["#cfe2f3", "#cfe2f3", "#fce5cd", "#fce5cd", "#d9ead3", "#d9ead3", "#d9ead3", "#f4cccc", "#ead1dc"]
for i, (s, c) in enumerate(zip(steps, cols)):
    x = i * 1.42; ax.add_patch(FancyBboxPatch((x, .3), 1.2, .9, boxstyle="round,pad=0.03", fc=c, ec="k")); ax.text(x + .6, .75, s, ha="center", va="center", fontsize=8)
    if i < len(steps) - 1: ax.annotate("", (x + 1.4, .75), (x + 1.23, .75), arrowprops=dict(arrowstyle="->"))
ax.set_xlim(-.1, 13); ax.set_ylim(0, 1.5); plt.tight_layout(); plt.savefig(f"{FIG}/fig1_workflow.png", dpi=200); plt.close()
# Fig 2 diagnostics (full tabular table)
D = load_clean(f"{ROOT}/data/sdss_agn_dataset.csv"); fig, ax = plt.subplots(2, 2, figsize=(9, 7)); cl = {0: ("#3b6fb6", "Galaxy"), 1: ("#d9534f", "QSO")}
for a, (c, lab, bins) in zip(ax.flat[:3], [("redshift", "redshift", np.linspace(0, .5, 70)), ("r", "r (mag)", np.linspace(14, 22, 60)), ("concentration", "R50/R90", np.linspace(.15, .7, 60))]):
    for k, (col, nm) in cl.items(): a.hist(D[c][D.label == k], bins, density=True, alpha=.65, color=col, label=nm)
    a.set_xlabel(lab); a.set_ylabel("density"); a.legend(frameon=False)
ax[0, 0].set_yscale("log")
for k, (col, nm) in cl.items(): ax[1, 1].scatter(D.redshift[D.label == k], D.r[D.label == k], s=2, alpha=.3, c=col, label=nm)
ax[1, 1].set_xlabel("redshift"); ax[1, 1].set_ylabel("r (mag)"); ax[1, 1].legend(markerscale=4, frameon=False)
for a, t in zip(ax.flat, "abcd"): a.set_title(f"({t})", loc="left", fontweight="bold")
plt.tight_layout(); plt.savefig(f"{FIG}/fig2_selection_bias_panel.png", dpi=200); plt.close()
# Fig 3 ROC/PR with bootstrap bands (first seed, MAIN)
s0 = seeds[0]; meths = [("Colour-cut selector (u-g)", "Colour cut"), ("Morphology rule", "Morphology rule"), ("XGBoost [14]", "XGBoost [14]"), (eff, "EfficientNet (frozen)"), ("Concatenation [12 shortcut-free]", "Concat [12]"), ("Gated attention [12 shortcut-free]", "Gated attention [12]")]
fig, ax = plt.subplots(1, 2, figsize=(11, 4.6)); grid = np.linspace(0, 1, 101); rng = np.random.default_rng(0)
for (key, lab) in meths:
    kk = [k for k in {m for (c, s, m) in PRED if c == MAIN and s == s0} if k == key or (key == "Morphology rule" and k.startswith("Morphology rule"))][0]
    P = PRED[(MAIN, s0, kk)]; yt, p = y_all[P["te"]], P["p"]; rocs, prs = [], []
    for _ in range(200):
        b = rng.integers(0, len(yt), len(yt))
        if yt[b].min() == yt[b].max(): continue
        f, t, _ = roc_curve(yt[b], p[b]); rocs.append(np.interp(grid, f, t)); pr, rc, _ = precision_recall_curve(yt[b], p[b]); prs.append(np.interp(grid, rc[::-1], pr[::-1]))
    f, t, _ = roc_curve(yt, p); l, = ax[0].plot(f, t, label=f"{lab} ({roc_auc_score(yt, p):.3f})"); ax[0].fill_between(grid, *np.percentile(rocs, [2.5, 97.5], 0), color=l.get_color(), alpha=.12)
    pr, rc, _ = precision_recall_curve(yt, p); ax[1].plot(rc, pr, color=l.get_color(), label=lab); ax[1].fill_between(grid, *np.percentile(prs, [2.5, 97.5], 0), color=l.get_color(), alpha=.12)
    if key.startswith("Gated"):
        pr_ = (p >= .5); ax[0].scatter([((pr_) & (yt == 0)).sum() / (yt == 0).sum()], [((pr_) & (yt == 1)).sum() / (yt == 1).sum()], c="k", zorder=5, marker="*", s=80, label="operating point (P=0.5)")
ax[0].plot([0, 1], [0, 1], "k:", lw=.8); ax[0].set(xlabel="False positive rate", ylabel="True positive rate", title="(a) ROC (AUC)"); ax[0].legend(frameon=False, fontsize=7, loc="lower right")
ax[1].set(xlabel="Recall", ylabel="Precision", title="(b) Precision-recall", ylim=(.4, 1.02)); plt.tight_layout(); plt.savefig(f"{FIG}/fig3_roc_pr.png", dpi=200); plt.close()
# Fig 4 shortcut-removal bars
cfgs = ["Full tabular features", "Without redshift", "Without luminosity proxy", "Without redshift + luminosity proxy", "Photometry + colours only", "Morphology proxies only"]
fig, ax = plt.subplots(figsize=(11, 4.6)); w = .2
for j, (pre, col, nm) in enumerate([("XGB | ", "#444", "XGBoost"), ("MLP | ", "#999", "MLP"), ("Fusion | ", "#2a9d8f", "Gated fusion")]):
    v = [get(MAIN, pre + c).auc for c in cfgs]; ax.bar(np.arange(len(cfgs)) + (j - 1) * w, [x.mean() for x in v], w, yerr=[x.std(ddof=1) for x in v], color=col, label=nm, capsize=2)
ax.axhline(eff_auc.mean(), ls="--", c="#3b6fb6", label=f"EfficientNet image-only ({eff_auc.mean():.3f})"); ax.set_ylim(.5, 1.01)
ax.set_xticks(range(len(cfgs))); ax.set_xticklabels([c.replace(" + ", "\n+ ").replace("Without ", "Without\n") for c in cfgs], fontsize=7.5); ax.set_ylabel(f"AUC (mean ± sd, {ns} seeds)"); ax.legend(frameon=False, fontsize=8, loc="lower left")
plt.tight_layout(); plt.savefig(f"{FIG}/fig4_shortcut_removal.png", dpi=200); plt.close()
# Fig 5 calibration
def pooled(name): 
    return np.concatenate([y_all[PRED[(MAIN, s, name)]["te"]] for s in seeds]), np.concatenate([PRED[(MAIN, s, name)]["p"] for s in seeds])
fig, ax = plt.subplots(1, 3, figsize=(14, 4)); bins = np.linspace(0, 1, 11)
for name, lab in [(eff, "EfficientNet (frozen)"), ("Gated attention [12 shortcut-free]", "Gated attention [12]")]:
    yy, pp = pooled(name); idx = np.digitize(pp, bins[1:-1]); xs = [pp[idx == k].mean() for k in range(10) if (idx == k).sum() > 5]; ys = [yy[idx == k].mean() for k in range(10) if (idx == k).sum() > 5]
    ax[0].plot(xs, ys, "o-", label=f"{lab} (ECE {expected_calibration_error(yy, pp):.3f})")
ax[0].plot([0, 1], [0, 1], "k:"); ax[0].set(xlabel="predicted P(AGN)", ylabel="observed fraction", title="(a) Reliability (pooled seeds)"); ax[0].legend(frameon=False, fontsize=7)
yy, pp = pooled("Gated attention [12 shortcut-free]"); ax[1].hist(pp[yy == 0], 20, alpha=.7, color="#3b6fb6", label="Galaxy"); ax[1].hist(pp[yy == 1], 20, alpha=.7, color="#d9534f", label="QSO"); ax[1].set(xlabel="P(AGN)", title="(b) Score distributions"); ax[1].legend(frameon=False)
ext = f"{ROOT}/results/external_scores.csv"
if os.path.exists(ext):
    E = pd.read_csv(ext); nm1, nm0 = ("BPT AGN", "BPT star-forming") if "subtype" in E and E.subtype.astype(str).str.contains("BPT").any() else ("external AGN", "external controls")
    ax[2].hist(E.score[E.label == 0], 20, alpha=.7, label=nm0); ax[2].hist(E.score[E.label == 1], 20, alpha=.7, label=nm1); ax[2].legend(frameon=False)
else: ax[2].text(.5, .5, "External cohort not yet available\n(results/external_scores.csv)", ha="center", va="center"); ax[2].set_xticks([]); ax[2].set_yticks([])
ax[2].set_title("(c) External cohort (QSO-trained model)"); plt.tight_layout(); plt.savefig(f"{FIG}/fig5_calibration.png", dpi=200); plt.close()
# Fig 6 qualitative grid / Fig 7 attention + SHAP
P = PRED[(MAIN, s0, "Gated attention [12 shortcut-free]")]; te, p = P["te"], P["p"]; yt = y_all[te]
pick = {"True positive": np.where((yt == 1) & (p > .5))[0][np.argsort(-p[(yt == 1) & (p > .5)])], "True negative": np.where((yt == 0) & (p < .5))[0][np.argsort(p[(yt == 0) & (p < .5)])],
        "False negative": np.where((yt == 1) & (p < .5))[0], "False positive": np.where((yt == 0) & (p >= .5))[0]}
path = lambda i: f"{ROOT}/data/images/{'agn' if cdf.label[i] == 1 else 'galaxy'}/{cdf.objid[i]}.jpg"
fig, ax = plt.subplots(4, 4, figsize=(9, 9.5))
for r, (nm, ids) in enumerate(pick.items()):
    for c in range(4):
        a = ax[r, c]; a.axis("off")
        if c < len(ids): g = te[ids[c]]; a.imshow(Image.open(path(g))); a.set_title(f"P={p[ids[c]]:.2f} | true={'QSO' if cdf.label[g] else 'Gal'}\nz={cdf.redshift[g]:.3f} | {str(cdf.objid[g])[-7:]}", fontsize=7)
    ax[r, 0].text(-.08, .5, nm, transform=ax[r, 0].transAxes, rotation=90, va="center", ha="right", fontweight="bold")
plt.tight_layout(); plt.savefig(f"{FIG}/fig6_qualitative.png", dpi=200); plt.close()
if "att" in EXTRA:
    te_a, att = EXTRA["att"]; assert (te_a == te).all()
    import shap, xgboost as xgb
    allcols = FEATURE_SETS["no_z_no_Mr"]; Xall = cdf[allcols].values
    exec_idx = np.array([k for k in range(len(cdf))])
    xm = xgb.XGBClassifier(n_estimators=200, max_depth=4, learning_rate=.05, n_jobs=1).fit(Xall[np.setdiff1d(exec_idx, te)], y_all[np.setdiff1d(exec_idx, te)])
    ex = shap.TreeExplainer(xm); fig, ax = plt.subplots(4, 3, figsize=(10, 10))
    for r, (nm, ids) in enumerate(pick.items()):
        a0, a1, a2 = ax[r]; a0.axis("off"); a1.axis("off"); a2.axis("off")
        if not len(ids): continue
        g = te[ids[0]]; im = Image.open(path(g)).convert("RGB"); a0.imshow(im); a0.set_title(f"{nm}  P={p[ids[0]]:.2f}", fontsize=8)
        a1.imshow(im); a1.imshow(np.array(Image.fromarray((att[ids[0]].reshape(7, 7) / att[ids[0]].max() * 255).astype(np.uint8)).resize(im.size, Image.BICUBIC)), cmap="jet", alpha=.45); a1.set_title("tabular-query attention", fontsize=8)
        sv = ex.shap_values(Xall[g:g + 1])[0]; o = np.argsort(-np.abs(sv))[:4]; a2.text(0, .5, "Top SHAP (XGB, 12 feats)\n" + "\n".join(f"{allcols[k]}: {sv[k]:+.2f}" for k in o), fontsize=8, va="center")
    fig.suptitle("Attention and SHAP show sensitivity, not causal physical evidence. (Grad-CAM column: use train_finetune.py checkpoints.)", fontsize=8); plt.tight_layout(); plt.savefig(f"{FIG}/fig7_attention_shap.png", dpi=200); plt.close()
print("figures done")