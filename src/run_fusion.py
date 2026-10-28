"""Multi-seed, matched-cohort evaluation of image-only / tabular-only / fusion variants.

All models in one seed share (a) identical train/val/test objects, (b) identical frozen
EfficientNet-B0 tokens, (c) identical early-stopping rule -> differences are attributable to the
model component under test.
"""
import os, sys, time, json, warnings
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score, r2_score, accuracy_score, precision_score, recall_score, f1_score
from scipy.stats import spearmanr
sys.path.insert(0, os.path.dirname(__file__))
from data import *
from stats import *
from models_fusion import FusionNet
warnings.filterwarnings("ignore")
torch.set_num_threads(os.cpu_count())
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
SEEDS = [int(s) for s in os.environ.get("SEEDS", "42,43,44,45,46").split(",")]
MAX_EP, PATIENCE, BS, LR, WD, LS = 70, 12, 64, 1e-3, 1e-2, 0.05
GRL_MAX, ADV_W = 1.0, 0.5
t0 = time.time()
log = lambda *a: print(f"[{time.time()-t0:7.1f}s]", *a, flush=True)

df = pd.read_csv(os.path.join(ROOT, "data", "image_cohort.csv"))
tok_all = np.load(os.path.join(ROOT, "data", "effb0_tokens.npy"), mmap_mode="r")   # (N,1280,7,7) fp16
N = len(df); y = df.label.values.astype(np.float32)
log("cohort", N, "QSO frac %.3f" % y.mean())
gap_all = np.asarray(tok_all, dtype=np.float32).mean((2, 3))                      # (N,1280)
tok_flat = np.asarray(tok_all, dtype=np.float32).reshape(N, 1280, 49).transpose(0, 2, 1)   # (N,49,1280)
logz = np.log10(df.redshift.values); logz_s = ((logz - logz.mean()) / logz.std()).astype(np.float32)

TAB = {"photo12": FEATURE_SETS["no_z_no_Mr"],            # shortcut-free: mags, colours, concentration
       "full14": ALL_FEATURES,                           # as described in the original manuscript
       "code7": ["u", "g", "r", "i", "z", "petroR50_r", "petroR90_r"]}   # what the original 08 script really fed

# (name, mode, tab-set)
CONFIGS = [("Image only (GAP head)", "img_gap", "photo12"),
           ("Image only (spatial attn-pool)", "img_spatial", "photo12"),
           ("Tabular only [photo12]", "tab", "photo12"),
           ("Concat [photo12]", "concat", "photo12"),
           ("Global-token gated [photo12]", "gate_global", "photo12"),
           ("Spatial attn, no gate [photo12]", "attn_nogate", "photo12"),
           ("Spatial attn + gate [photo12]", "attn_gate", "photo12"),
           ("Spatial attn + gate + z-adversary [photo12]", "attn_gate_grl", "photo12"),
           ("Tabular only [full14]", "tab", "full14"),
           ("Concat [full14]", "concat", "full14"),
           ("Spatial attn + gate [full14]", "attn_gate", "full14"),
           ("Tabular only [code7]", "tab", "code7"),
           ("Spatial attn + gate [code7]", "attn_gate", "code7")]


def predict(model, gap, tok, tab, bs=256):
    model.eval(); out, hs, auxs = [], [], {"gate": [], "attn": []}
    with torch.no_grad():
        for s in range(0, len(gap), bs):
            lg, _, h, aux = model(gap[s:s+bs], tok[s:s+bs], tab[s:s+bs])
            out.append(torch.sigmoid(lg)); hs.append(h)
            for k in auxs:
                if k in aux: auxs[k].append(aux[k])
    aux = {k: torch.cat(v).numpy() for k, v in auxs.items() if v}
    return torch.cat(out).numpy(), torch.cat(hs).numpy(), aux


def train_one(mode, tab_cols, split, seed, tensors):
    tr, va, te = split
    gap, tok, tabX, yt, zt = tensors
    torch.manual_seed(seed); np.random.seed(seed)
    model = FusionNet(mode, tab_dim=tabX.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_EP, eta_min=1e-5)
    pw = torch.tensor(float((y[tr] == 0).sum() / (y[tr] == 1).sum()))
    best, best_state, bad = -1, None, 0
    for ep in range(MAX_EP):
        model.train(); perm = torch.from_numpy(np.random.permutation(tr))
        lam = GRL_MAX * (2 / (1 + np.exp(-10 * ep / MAX_EP)) - 1)
        for s in range(0, len(perm), BS):
            b = perm[s:s+BS]
            if len(b) < 4: continue
            lg, adv, _, _ = model(gap[b], tok[b], tabX[b], grl_lambda=lam)
            yb = yt[b]; ys = yb * (1 - LS) + 0.5 * LS
            loss = nn.functional.binary_cross_entropy_with_logits(lg, ys, pos_weight=pw)
            if adv is not None:
                loss = loss + ADV_W * nn.functional.mse_loss(adv, zt[b])
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        sched.step()
        pv, _, _ = predict(model, gap[va], tok[va], tabX[va])
        a = roc_auc_score(y[va], pv)
        if a > best + 1e-5:
            best, bad = a, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE: break
    model.load_state_dict(best_state)
    return model, best, ep + 1


all_rows, preds, extras = [], {}, {}
for seed in SEEDS:
    tr, va, te = split_70_15_15(df, seed)
    # unsupervised PCA of tokens (fit on TRAIN images only) -> 128-d tokens
    rng = np.random.default_rng(seed)
    samp = tok_flat[tr][:, rng.choice(49, 12, replace=False), :].reshape(-1, 1280)
    pca = PCA(128, random_state=seed).fit(samp)
    tokp = pca.transform(tok_flat.reshape(-1, 1280)).reshape(N, 49, 128).astype(np.float32)
    tokp /= tokp[tr].std()
    gap_s = StandardScaler().fit(gap_all[tr]); gap_t = torch.from_numpy(gap_s.transform(gap_all).astype(np.float32))
    tok_t = torch.from_numpy(tokp)
    yt, zt = torch.from_numpy(y), torch.from_numpy(logz_s)
    for name, mode, tset in CONFIGS:
        cols = TAB[tset]
        sc = StandardScaler().fit(df[cols].values[tr])
        tabX = torch.from_numpy(sc.transform(df[cols].values).astype(np.float32))
        model, vauc, nep = train_one(mode, cols, (tr, va, te), seed, (gap_t, tok_t, tabX, yt, zt))
        p, h, aux = predict(model, gap_t[te], tok_t[te], tabX[te])
        yte = y[te]
        pred = (p > 0.5).astype(int)
        row = dict(seed=seed, model=name, mode=mode, tab=tset, val_auc=vauc, epochs=nep, n_test=len(te),
                   auc=roc_auc_score(yte, p), acc=accuracy_score(yte, pred), prec=precision_score(yte, pred),
                   rec=recall_score(yte, pred), f1=f1_score(yte, pred), **calibration_summary(yte, p))
        # shortcut diagnostics: does the score order QSOs by redshift?
        q = yte == 1
        row["rho_score_z_within_QSO"] = spearmanr(p[q], df.redshift.values[te][q])[0]
        # linear probe: how much redshift is linearly decodable from the fused representation?
        _, htr, _ = predict(model, gap_t[tr], tok_t[tr], tabX[tr])
        row["probe_R2_logz"] = r2_score(logz_s[te], Ridge(alpha=1.0).fit(htr, logz_s[tr]).predict(h))
        # modality-perturbation tests (only for models using both modalities)
        if mode in ("concat", "gate_global", "attn_nogate", "attn_gate", "attn_gate_grl"):
            perm = np.random.default_rng(seed).permutation(len(te))
            pt, _, _ = predict(model, gap_t[te], tok_t[te], tabX[te][perm])           # break tab<->image pairing
            row["auc_tab_shuffled"] = roc_auc_score(yte, pt)
            mean_gap = gap_t[tr].mean(0, keepdim=True).expand(len(te), -1); mean_tok = tok_t[tr].mean(0, keepdim=True).expand(len(te), -1, -1)
            pi, _, _ = predict(model, mean_gap, mean_tok, tabX[te])                   # blank image
            row["auc_image_blanked"] = roc_auc_score(yte, pi)
            pz, _, _ = predict(model, gap_t[te], tok_t[te], torch.zeros_like(tabX[te]))   # tab at train mean
            row["auc_tab_at_mean"] = roc_auc_score(yte, pz)
        if "gate" in aux:
            row["gate_mean_QSO"], row["gate_mean_GAL"] = float(aux["gate"][q].mean()), float(aux["gate"][~q].mean())
            row["gate_std_across_objects"] = float(aux["gate"].mean(1).std())
        if "attn" in aux:
            a_ = aux["attn"]; row["attn_entropy"] = float(-(a_ * np.log(a_ + 1e-9)).sum(1).mean() / np.log(49))
            if seed == SEEDS[0]:
                extras[f"attn::{name}"] = dict(attn=a_, y=yte, idx=te)
        preds[(seed, name)] = (te, p)
        all_rows.append(row)
        log(f"seed {seed} | {name:46s} AUC {row['auc']:.4f} (val {vauc:.4f}, {nep} ep) rho_z|QSO {row['rho_score_z_within_QSO']:+.2f} probeR2 {row['probe_R2_logz']:.2f}")
    pd.DataFrame(all_rows).to_csv(os.path.join(RES, "fusion_all_runs.csv"), index=False)

R = pd.DataFrame(all_rows)
np.savez_compressed(os.path.join(RES, "fusion_predictions.npz"),
                    **{f"{s}|{n}": np.stack([te, p]) for (s, n), (te, p) in preds.items()})
np.savez_compressed(os.path.join(RES, "fusion_attention_seed0.npz"), **{k: v["attn"] for k, v in extras.items()},
                    y=list(extras.values())[0]["y"] if extras else np.array([]), idx=list(extras.values())[0]["idx"] if extras else np.array([]))
log("DONE")
