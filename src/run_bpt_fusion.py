"""Image / tabular / fusion comparison on the BPT task (narrow-line AGN vs star-forming, matched in z and r). 10 splits, CPU.
Needs data/bpt_tokens.npy + data/bpt_cohort_matched.csv.  Env: QUICK=1 (3 seeds, 6 epochs), SEEDS=...
Outputs: results/paper_tables/Table12_bpt_fusion.csv, Table13_bpt_paired.csv (+.md)"""
import os, sys, time, warnings, numpy as np, pandas as pd, torch, torch.nn as nn
from scipy import stats as sps
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
import xgboost as xgb
sys.path.insert(0, os.path.dirname(__file__))
from data import *; from stats import *; from models_fusion import FusionNet
warnings.filterwarnings("ignore"); torch.set_num_threads(os.cpu_count())
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); OUT = f"{ROOT}/results/paper_tables"; os.makedirs(OUT, exist_ok=True); os.makedirs(f"{ROOT}/results/paper", exist_ok=True)
QUICK = os.environ.get("QUICK") == "1"; MAX_EP, PATIENCE, BS, LR, WD, LS = (6 if QUICK else 70), 12, 64, 1e-3, 1e-2, 0.05
SEEDS = [int(s) for s in os.environ["SEEDS"].split(",")] if os.environ.get("SEEDS") else (list(range(42, 45)) if QUICK else list(range(42, 52)))
t0 = time.time(); log = lambda *a: print(f"[{time.time()-t0:7.1f}s]", *a, flush=True)
d = pd.read_csv(f"{ROOT}/data/bpt_cohort_matched.csv"); N = len(d); y = d.label.values.astype(np.float32)
tok = np.load(f"{ROOT}/data/bpt_tokens.npy", mmap_mode="r"); gap_all = np.asarray(tok, dtype=np.float32).mean((2, 3))
tok_flat = np.asarray(tok, dtype=np.float32).reshape(N, 1280, 49).transpose(0, 2, 1); log("cohort", N, d.label.value_counts().to_dict())
SF = FEATURE_SETS["no_z_no_Mr"]
def predict(m, g, t, x):
    m.eval(); o = []
    with torch.no_grad():
        for s in range(0, len(g), 256): o.append(torch.sigmoid(m(g[s:s+256], t[s:s+256], x[s:s+256])[0]))
    return torch.cat(o).numpy()
def train_t(mode, T, tr, va, seed):
    g, t, x, yt = T; torch.manual_seed(seed); np.random.seed(seed); m = FusionNet(mode, tab_dim=x.shape[1])
    opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=WD); sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_EP, eta_min=1e-5)
    pw = torch.tensor(float((y[tr] == 0).sum() / (y[tr] == 1).sum())); best, st, bad = -1, None, 0
    for ep in range(MAX_EP):
        m.train(); perm = torch.from_numpy(np.random.permutation(tr))
        for s in range(0, len(perm), BS):
            b = perm[s:s + BS]
            if len(b) < 4: continue
            loss = nn.functional.binary_cross_entropy_with_logits(m(g[b], t[b], x[b])[0], yt[b] * (1 - LS) + 0.5 * LS, pos_weight=pw)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        sch.step(); a = roc_auc_score(y[va], predict(m, g[va], t[va], x[va]))
        if a > best + 1e-5: best, bad, st = a, 0, {k: v.clone() for k, v in m.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE: break
    m.load_state_dict(st); return m
P, rows = {}, []
def rec(seed, name, te, p, thr=0.5):
    P[(seed, name)] = (te, p); pr = (p >= thr).astype(int)
    rows.append(dict(seed=seed, model=name, auc=roc_auc_score(y[te], p), acc=100 * accuracy_score(y[te], pr), f1=f1_score(y[te], pr), **calibration_summary(y[te], p)))
for seed in SEEDS:
    tr, va, te = split_70_15_15(d, seed); rng = np.random.default_rng(seed)
    pca = PCA(128, random_state=seed).fit(tok_flat[tr][:, rng.choice(49, 12, replace=False), :].reshape(-1, 1280))
    tokp = pca.transform(tok_flat.reshape(-1, 1280)).reshape(N, 49, 128).astype(np.float32); tokp /= tokp[tr].std()
    g_t = torch.from_numpy(StandardScaler().fit(gap_all[tr]).transform(gap_all).astype(np.float32)); t_t = torch.from_numpy(tokp); yt = torch.from_numpy(y)
    def run_t(name, mode, cols):
        sc = StandardScaler().fit(d[cols].values[tr]); X = torch.from_numpy(sc.transform(d[cols].values).astype(np.float32))
        m = train_t(mode, (g_t, t_t, X, yt), tr, va, seed); rec(seed, name, te, predict(m, g_t[te], t_t[te], X[te]))
    run_t("EfficientNet-B0 (frozen, image only)", "img_gap", SF)
    run_t("Concatenation [12]", "concat", SF); run_t("Cross-attention, no gate [12]", "attn_nogate", SF); run_t("Global-token + gate [12]", "gate_global", SF); run_t("Gated attention [12]", "attn_gate", SF)
    run_t("Concatenation [14]", "concat", ALL_FEATURES); run_t("Gated attention [14]", "attn_gate", ALL_FEATURES)
    for nm, cols in (("XGBoost [12]", SF), ("XGBoost [14]", ALL_FEATURES), ("XGBoost [colours]", COLOURS)):
        spw = float((y[tr] == 0).sum() / (y[tr] == 1).sum())
        rec(seed, nm, te, xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, n_jobs=1, random_state=seed, scale_pos_weight=spw, eval_metric="logloss").fit(d[cols].values[tr], y[tr]).predict_proba(d[cols].values[te])[:, 1])
    rec(seed, "MLP [12]", te, make_pipeline(StandardScaler(), MLPClassifier((128, 64), early_stopping=True, random_state=seed, max_iter=300)).fit(d[SF].values[tr], y[tr]).predict_proba(d[SF].values[te])[:, 1])
    rec(seed, "Colour logistic regression", te, make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(d[["u_g", "g_r", "r_i", "i_z"]].values[tr], y[tr]).predict_proba(d[["u_g", "g_r", "r_i", "i_z"]].values[te])[:, 1])
    log("seed", seed, "done")
R = pd.DataFrame(rows); ms = lambda v, k=4: f"{np.mean(v):.{k}f} ± {np.std(v, ddof=1) if len(v) > 1 else 0:.{k}f}"
order = list(dict.fromkeys(R.model)); g = R.groupby("model")
T12 = pd.DataFrame([{"Model": m, "AUC": ms(g.get_group(m).auc), "Accuracy (%)": ms(g.get_group(m).acc, 2), "F1": ms(g.get_group(m).f1, 3), "Brier": ms(g.get_group(m).brier), "ECE": ms(g.get_group(m).ece)} for m in order])
T12.to_csv(f"{OUT}/Table12_bpt_fusion.csv", index=False); open(f"{OUT}/Table12_bpt_fusion.md", "w").write(T12.to_markdown(index=False))
def paired(a, b):
    dd, ps = [], []
    for s in SEEDS:
        (ta, pa), (tb, pb) = P[(s, a)], P[(s, b)]; assert (ta == tb).all(); r = delong_roc_test(y[ta], pa, pb); dd.append(r[2]); ps.append(r[4])
    dd = np.array(dd)
    try: w = sps.wilcoxon(dd).pvalue
    except Exception: w = float("nan")
    return dict(a=a, b=b, mean_diff=f"{dd.mean():+.4f}", seeds_a_better=f"{(dd > 0).sum()}/{len(dd)}", delong_sig=f"{(np.array(ps) < .05).sum()}/{len(dd)}", wilcoxon_p=f"{w:.3g}")
E = "EfficientNet-B0 (frozen, image only)"
T13 = pd.DataFrame([paired(a, b) for a, b in [("Gated attention [12]", E), ("Concatenation [12]", E), ("Gated attention [12]", "XGBoost [12]"), ("Concatenation [12]", "XGBoost [12]"), ("Gated attention [12]", "MLP [12]"),
                                              ("Gated attention [12]", "Concatenation [12]"), ("Cross-attention, no gate [12]", "Concatenation [12]"), ("Gated attention [14]", "XGBoost [14]"), ("Gated attention [14]", "Gated attention [12]"), ("XGBoost [12]", E)]])
T13.to_csv(f"{OUT}/Table13_bpt_paired.csv", index=False); open(f"{OUT}/Table13_bpt_paired.md", "w").write(T13.to_markdown(index=False)); print(T12.to_string(index=False)); print(T13.to_string(index=False)); import pickle; pickle.dump(P, open(f"{ROOT}/results/paper/bpt_predictions.pkl", "wb")); log("DONE")
