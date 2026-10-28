"""Aggregate multi-seed fusion runs -> tables, paired DeLong tests, figures."""
import os, sys, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon, ttest_rel
sys.path.insert(0, os.path.dirname(__file__))
from stats import delong_roc_test, paired_bootstrap_diff
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); RES, FIG = f"{ROOT}/results", f"{ROOT}/figures"
R = pd.read_csv(f"{RES}/fusion_all_runs.csv")
df = pd.read_csv(f"{ROOT}/data/image_cohort.csv"); y = df.label.values
P = np.load(f"{RES}/fusion_predictions.npz")
seeds = sorted(R.seed.unique()); order = list(dict.fromkeys(R.model))

# ---- Table 1: mean +- sd over seeds
g = R.groupby("model")
T = pd.DataFrame({
    "AUC": g.auc.mean(), "AUC_sd": g.auc.std(), "AUC_min": g.auc.min(), "Acc": g.acc.mean(), "F1": g.f1.mean(),
    "Brier": g.brier.mean(), "ECE": g.ece.mean(), "rho_z|QSO": g.rho_score_z_within_QSO.mean(),
    "probeR2_logz": g.probe_R2_logz.mean()}).loc[order]
T.to_csv(f"{RES}/fusion_table_main.csv"); print(T.round(4).to_string())

# ---- Table 2: paired DeLong per seed (same test objects within seed) for key contrasts
contrasts = [("Spatial attn + gate [photo12]", "Image only (GAP head)"),
             ("Spatial attn + gate [photo12]", "Tabular only [photo12]"),
             ("Spatial attn + gate [photo12]", "Concat [photo12]"),
             ("Spatial attn + gate [photo12]", "Spatial attn, no gate [photo12]"),
             ("Spatial attn + gate [photo12]", "Global-token gated [photo12]"),
             ("Concat [photo12]", "Image only (GAP head)"),
             ("Tabular only [photo12]", "Image only (GAP head)"),
             ("Spatial attn + gate [full14]", "Tabular only [full14]"),
             ("Spatial attn + gate [full14]", "Spatial attn + gate [photo12]"),
             ("Spatial attn + gate + z-adversary [photo12]", "Spatial attn + gate [photo12]")]
rows = []
for a, b in contrasts:
    d, ps, zs = [], [], []
    for s in seeds:
        te, pa = P[f"{s}|{a}"]; te2, pb = P[f"{s}|{b}"]; te = te.astype(int)
        assert (te == te2.astype(int)).all()
        a1, a2, diff, z, p = delong_roc_test(y[te], pa, pb)
        d.append(diff); ps.append(p)
    d, ps = np.array(d), np.array(ps)
    try: wp = wilcoxon(d).pvalue
    except Exception: wp = np.nan
    rows.append(dict(model_a=a, model_b=b, mean_diff=d.mean(), sd_diff=d.std(ddof=1), min_diff=d.min(), max_diff=d.max(),
                     seeds_a_better=int((d > 0).sum()), seeds_total=len(d), seeds_delong_p_lt_005=int((ps < 0.05).sum()),
                     median_delong_p=float(np.median(ps)), wilcoxon_p_across_seeds=wp))
C = pd.DataFrame(rows); C.to_csv(f"{RES}/fusion_paired_contrasts.csv", index=False)
print(C.round(4).to_string())

# ---- modality perturbation + gate
M = R[R.auc_tab_shuffled.notna()].groupby("model")[["auc", "auc_tab_shuffled", "auc_tab_at_mean", "auc_image_blanked"]].mean()
M.to_csv(f"{RES}/fusion_modality_perturbation.csv"); print(M.round(4).to_string())
Gt = R[R.gate_mean_QSO.notna()].groupby("model")[["gate_mean_QSO", "gate_mean_GAL", "gate_std_across_objects", "attn_entropy"]].mean()
Gt.to_csv(f"{RES}/fusion_gate_stats.csv"); print(Gt.round(3).to_string())

# ---- figures
fig, ax = plt.subplots(figsize=(9, 5.5))
labs = [o for o in order]; data = [R[R.model == o].auc.values for o in labs]
colors = ["#3b6fb6" if "Image" in o else ("#d9534f" if "[full14]" in o else ("#777" if "code7" in o else "#2a9d8f")) for o in labs]
bp = ax.boxplot(data, vert=False, patch_artist=True, widths=.6)
for patch, c in zip(bp["boxes"], colors): patch.set_facecolor(c); patch.set_alpha(.6)
ax.set_yticklabels(labs, fontsize=8); ax.invert_yaxis(); ax.set_xlabel(f"test AUC over {len(seeds)} independent splits")
ax.set_xlim(R.auc.min() - .002, 1.0005)
plt.tight_layout(); plt.savefig(f"{FIG}/fig_F_fusion_multiseed.png", dpi=200); plt.close()

# mean attention maps
try:
    A = np.load(f"{RES}/fusion_attention_seed0.npz"); yy = A["y"]
    key = [k for k in A.files if k.startswith("attn::") and "Spatial attn + gate [photo12]" == k.split("::")[1]][0]
    a = A[key].reshape(-1, 7, 7)
    fig, ax = plt.subplots(1, 3, figsize=(9, 3))
    for i, (m, t) in enumerate([(yy == 1, "QSO (mean attention)"), (yy == 0, "GALAXY (mean attention)")]):
        im = ax[i].imshow(a[m].mean(0), cmap="magma"); ax[i].set_title(t, fontsize=9); ax[i].axis("off"); plt.colorbar(im, ax=ax[i], fraction=.046)
    im = ax[2].imshow(a[yy == 1].mean(0) - a[yy == 0].mean(0), cmap="coolwarm"); ax[2].set_title("QSO - GALAXY", fontsize=9); ax[2].axis("off"); plt.colorbar(im, ax=ax[2], fraction=.046)
    plt.tight_layout(); plt.savefig(f"{FIG}/fig_G_attention_maps.png", dpi=200); plt.close()
except Exception as e:
    print("attention figure skipped:", e)
print("analysis done")
