"""Generates the data behind Tables 1-7 and Figures 3-7 of the revised paper.

Protocol: every model sees the same objects in every split.  The MAIN cohort is the image+tabular set matched
by coarsened exact matching on r (0.5 mag) and log10 R50 (0.1 dex) - redshift matching is impossible on the current
sample (only 4 of 1048 image-cohort QSOs lie inside the galaxy redshift range).  Image tokens come from a frozen
ImageNet EfficientNet-B0 (see extract_features.py); swap in fine-tuned predictions from train_finetune.py for the
final paper.  Environment: SEEDS=42,43,...  QUICK=1 for a smoke run.
"""
import os, sys, time, json, warnings, pickle
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (roc_auc_score, average_precision_score, roc_curve, precision_recall_curve,
                             accuracy_score, precision_score, recall_score, f1_score)
import xgboost as xgb
sys.path.insert(0, os.path.dirname(__file__))
from data import *
from stats import *
from models_fusion import FusionNet
warnings.filterwarnings("ignore"); torch.set_num_threads(os.cpu_count())
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); RES = f"{ROOT}/results"; os.makedirs(f"{RES}/paper", exist_ok=True)
SEEDS = [int(s) for s in os.environ.get("SEEDS", ",".join(map(str, range(42, 52)))).split(",")]
QUICK = os.environ.get("QUICK") == "1"
MAX_EP, PATIENCE, BS, LR, WD, LS = (8 if QUICK else 70), 12, 64, 1e-3, 1e-2, 0.05
t0 = time.time(); log = lambda *a: print(f"[{time.time()-t0:7.1f}s]", *a, flush=True)

# ----------------------------------------------------------------- data ----
df = pd.read_csv(f"{ROOT}/data/image_cohort.csv"); N = len(df); y_all = df.label.values.astype(np.float32)
df["log_r50"] = np.log10(df.petroR50_r.clip(lower=0.05))
tok = np.load(f"{ROOT}/data/effb0_tokens.npy", mmap_mode="r")
gap_all = np.asarray(tok, dtype=np.float32).mean((2, 3))
tok_flat = np.asarray(tok, dtype=np.float32).reshape(N, 1280, 49).transpose(0, 2, 1)
logz = np.log10(df.redshift.values); logz_s = ((logz - logz.mean()) / logz.std()).astype(np.float32)


def cem(d, keys, seed=0):
    rng = np.random.default_rng(seed); key = pd.Series("", index=d.index)
    for c, w in keys: key = key + "_" + np.floor(d[c] / w).astype(int).astype(str)
    keep = []
    for _, g in d.groupby(key):
        q, gl = g.index[g.label == 1], g.index[g.label == 0]; n = min(len(q), len(gl))
        if n: keep += list(rng.choice(q, n, replace=False)) + list(rng.choice(gl, n, replace=False))
    return np.sort(np.array(keep))

COHORTS = {"Original image cohort": np.arange(N),
           "Magnitude-matched (r)": cem(df, [("r", 0.5)]),
           "Joint r + size matched (MAIN)": cem(df, [("r", 0.5), ("log_r50", 0.1)])}
MAIN = "Joint r + size matched (MAIN)"
zmax_gal = df.redshift[df.label == 0].max()
cohort_info = {k: dict(n=int(len(v)), n_qso=int(y_all[v].sum()), n_gal=int((1 - y_all[v]).sum())) for k, v in COHORTS.items()}
cohort_info["z-support check"] = dict(qso_inside_galaxy_z_support=int(((df.label == 1) & (df.redshift <= zmax_gal)).sum()),
                                     n_qso=int((df.label == 1).sum()), n_obj_z_gt_0p5=int((df.redshift > 0.5).sum()), galaxy_zmax=float(zmax_gal))
json.dump(cohort_info, open(f"{RES}/paper/cohort_info.json", "w"), indent=1); log(cohort_info)

CFG = {  # feature configurations of Table 2
 "Full tabular features": ALL_FEATURES,
 "Without redshift": [c for c in ALL_FEATURES if c != "redshift"],
 "Without luminosity proxy": [c for c in ALL_FEATURES if c != "abs_mag_r"],
 "Without redshift + luminosity proxy": FEATURE_SETS["no_z_no_Mr"],
 "Photometry + colours only": MAGS + COLOURS,
 "Morphology proxies only": ["concentration", "petroR50_r", "petroR90_r"]}
SF = "Without redshift + luminosity proxy"          # the shortcut-free set ('photo12')


# ------------------------------------------------------------- torch part --
def predict(model, gap, tk, tab, bs=256):
    model.eval(); out, aux_a = [], []
    with torch.no_grad():
        for s in range(0, len(gap), bs):
            lg, _, h, aux = model(gap[s:s+bs], tk[s:s+bs], tab[s:s+bs]); out.append(torch.sigmoid(lg))
            if "attn" in aux: aux_a.append(aux["attn"])
    return torch.cat(out).numpy(), (torch.cat(aux_a).numpy() if aux_a else None)


def train_torch(mode, T, tr, va, seed):
    gap, tk, tab, yt = T
    torch.manual_seed(seed); np.random.seed(seed)
    model = FusionNet(mode, tab_dim=tab.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_EP, eta_min=1e-5)
    pw = torch.tensor(float((y_all[tr] == 0).sum() / (y_all[tr] == 1).sum())); best, st, bad = -1, None, 0
    for ep in range(MAX_EP):
        model.train(); perm = torch.from_numpy(np.random.permutation(tr))
        for s in range(0, len(perm), BS):
            b = perm[s:s+BS]
            if len(b) < 4: continue
            lg, _, _, _ = model(gap[b], tk[b], tab[b])
            loss = nn.functional.binary_cross_entropy_with_logits(lg, yt[b] * (1 - LS) + 0.5 * LS, pos_weight=pw)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        sched.step()
        pv, _ = predict(model, gap[va], tk[va], tab[va]); a = roc_auc_score(y_all[va], pv)
        if a > best + 1e-5: best, bad, st = a, 0, {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE: break
    model.load_state_dict(st); return model


def op_metrics(y, p, thr):
    pr = (p >= thr).astype(int)
    return dict(acc=100 * accuracy_score(y, pr), prec=precision_score(y, pr, zero_division=0), rec=recall_score(y, pr), f1=f1_score(y, pr))


def youden(score, y):
    f, t, th = roc_curve(y, score); return th[np.argmax(t - f)]


# --------------------------------------------------------------- main loop --
PRED, ROWS = {}, []          # PRED[(cohort,seed,model)] = dict(te,p,va,pv,thr)
EXTRA = {}
def record(cohort, seed, name, te, p, va, pv, thr, secs, params=np.nan, kind="tab"):
    yt = y_all[te]
    PRED[(cohort, seed, name)] = dict(te=te, p=p, va=va, pv=pv, thr=thr)
    a, lo, hi = bootstrap_auc(yt, p, n_boot=200 if QUICK else 500, seed=seed)
    ROWS.append(dict(cohort=cohort, seed=seed, model=name, auc=a, ci_lo=lo, ci_hi=hi, secs=secs, params=params, **op_metrics(yt, p, thr)))

for cname, cidx in COHORTS.items():
    full = cname == MAIN
    sub = df.iloc[cidx].reset_index(drop=True)
    for seed in SEEDS:
        tr_p, va_p, te_p = split_70_15_15(sub, seed); tr, va, te = cidx[tr_p], cidx[va_p], cidx[te_p]
        # image tensors (PCA + scaler fitted on training images of this split only)
        rng = np.random.default_rng(seed)
        samp = tok_flat[tr][:, rng.choice(49, 12, replace=False), :].reshape(-1, 1280)
        pca = PCA(128, random_state=seed).fit(samp)
        tokp = pca.transform(tok_flat.reshape(-1, 1280)).reshape(N, 49, 128).astype(np.float32); tokp /= tokp[tr].std()
        gap_t = torch.from_numpy(StandardScaler().fit(gap_all[tr]).transform(gap_all).astype(np.float32)); tok_t = torch.from_numpy(tokp)
        yt_t = torch.from_numpy(y_all)
        def tabT(cols):
            sc = StandardScaler().fit(df[cols].values[tr]); return torch.from_numpy(sc.transform(df[cols].values).astype(np.float32)), sc
        def run_t(name, mode, cols):
            t1 = time.time(); X, _ = tabT(cols); m = train_torch(mode, (gap_t, tok_t, X, yt_t), tr, va, seed)
            pv, _ = predict(m, gap_t[va], tok_t[va], X[va]); p, att = predict(m, gap_t[te], tok_t[te], X[te])
            record(cname, seed, name, te, p, va, pv, 0.5, time.time() - t1, sum(q.numel() for q in m.parameters()))
            if full and seed == SEEDS[0] and att is not None and name == "Gated attention [12 shortcut-free]": EXTRA["att"] = (te, att)
            return m
        # --- torch image / fusion models (all cohorts)
        run_t("EfficientNet-B0 (frozen, image only)", "img_gap", FEATURE_SETS["no_z_no_Mr"])
        run_t("Gated attention [12 shortcut-free]", "attn_gate", CFG[SF])
        # --- tabular XGBoost on 12 features and classical LR (all cohorts)
        def run_sk(name, mk, cols, score="proba"):
            t1 = time.time(); Xtr, Xva, Xte = [df[cols].values[i] for i in (tr, va, te)]
            m = mk().fit(Xtr, y_all[tr])
            f = (lambda X: m.predict_proba(X)[:, 1]) if score == "proba" else (lambda X: m.decision_function(X))
            ptr, pv, p = f(Xtr), f(Xva), f(Xte)
            if score != "proba": thr = 0.0
            else: thr = 0.5
            record(cname, seed, name, te, p, va, pv, thr, time.time() - t1)
        mk_xgb = lambda: xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                                           n_jobs=1, random_state=seed, scale_pos_weight=float((y_all[tr] == 0).sum() / (y_all[tr] == 1).sum()), eval_metric="logloss")
        run_sk("XGBoost [12 shortcut-free]", mk_xgb, CFG[SF])
        run_sk("Logistic regression [colours]", lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")), ["u_g", "g_r", "r_i", "i_z"])
        if full:
            # ---- Table 1 classical / tabular family on the 14 features
            t1 = time.time(); cs = -df.u_g.values; thr = youden(cs[tr], y_all[tr]); record(cname, seed, "Colour-cut selector (u-g)", te, cs[te], va, cs[va], thr, 0.0)
            mv = "petroR50_r" if roc_auc_score(y_all[tr], -df.petroR50_r.values[tr]) > roc_auc_score(y_all[tr], -df.concentration.values[tr]) else "concentration"
            ms = -df[mv].values; thr = youden(ms[tr], y_all[tr]); record(cname, seed, f"Morphology rule ({mv})", te, ms[te], va, ms[va], thr, 0.0)
            run_sk("Logistic regression [14]", lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")), ALL_FEATURES)
            run_sk("Random forest [14]", lambda: RandomForestClassifier(300, n_jobs=1, random_state=seed, class_weight="balanced"), ALL_FEATURES)
            run_sk("SVM (RBF) [14]", lambda: make_pipeline(StandardScaler(), SVC(class_weight="balanced")), ALL_FEATURES, score="dec")
            run_sk("XGBoost [14]", mk_xgb, ALL_FEATURES)
            run_sk("MLP [14]", lambda: make_pipeline(StandardScaler(), MLPClassifier((256, 128, 64), early_stopping=True, random_state=seed, max_iter=300)), ALL_FEATURES)
            run_sk("MLP [12 shortcut-free]", lambda: make_pipeline(StandardScaler(), MLPClassifier((256, 128, 64), early_stopping=True, random_state=seed, max_iter=300)), CFG[SF])
            # ---- Table 2: configuration sweep
            for cfg, cols in CFG.items():
                run_sk(f"XGB | {cfg}", mk_xgb, cols)
                run_sk(f"MLP | {cfg}", lambda: make_pipeline(StandardScaler(), MLPClassifier((128, 64), early_stopping=True, random_state=seed, max_iter=300)), cols)
                if cfg != SF: run_t(f"Fusion | {cfg}", "attn_gate", cols)
                else: PRED[(cname, seed, f"Fusion | {cfg}")] = PRED[(cname, seed, "Gated attention [12 shortcut-free]")]; ROWS.append({**[r for r in ROWS if r["cohort"] == cname and r["seed"] == seed and r["model"] == "Gated attention [12 shortcut-free]"][-1], "model": f"Fusion | {cfg}"})
            # ---- Table 4 ladder + Table 5 comparators
            run_t("Concatenation [12 shortcut-free]", "concat", CFG[SF])
            run_t("Cross-attention, no gate [12 shortcut-free]", "attn_nogate", CFG[SF])
            run_t("Global-token + gate [12 shortcut-free]", "gate_global", CFG[SF])
            run_t("Concatenation [14]", "concat", ALL_FEATURES)
            run_t("Cross-attention, no gate [14]", "attn_nogate", ALL_FEATURES)
            run_t("Gated attention [14]", "attn_gate", ALL_FEATURES)
            # ---- shift test: models trained on the matched cohort scored on the objects that matching DISCARDED
            if seed == SEEDS[0]:
                rest = np.setdiff1d(np.arange(N), cidx)
                if len(rest) > 20:
                    X12, _ = tabT(CFG[SF]); m = train_torch("attn_gate", (gap_t, tok_t, X12, yt_t), tr, va, seed)
                    pr_, _ = predict(m, gap_t[rest], tok_t[rest], X12[rest])
                    zq = (df.redshift.values[rest] > zmax_gal) & (y_all[rest] == 1)
                    EXTRA["shift"] = dict(n_rest=int(len(rest)), auc_rest=float(roc_auc_score(y_all[rest], pr_)) if len(set(y_all[rest])) > 1 else None,
                                          n_qso_out_of_support=int(zq.sum()), mean_p_out_of_support=float(pr_[zq].mean()), frac_detected_out_of_support=float((pr_[zq] > .5).mean()))
    log(f"{cname}: seed {seed} done")
    pd.DataFrame(ROWS).to_csv(f"{RES}/paper/all_runs.csv", index=False)

pickle.dump(PRED, open(f"{RES}/paper/predictions.pkl", "wb")); pickle.dump(EXTRA, open(f"{RES}/paper/extra.pkl", "wb"))
json.dump(EXTRA.get("shift", {}), open(f"{RES}/paper/shift_test.json", "w"), indent=1)
log("EXPERIMENTS DONE")
