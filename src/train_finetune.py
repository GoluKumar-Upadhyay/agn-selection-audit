"""End-to-end fine-tuned EfficientNet-B0 + corrected gated cross-attention (run on a GPU).

Differences from the original 07/08 scripts
  * ONE split protocol shared by image-only, tabular-only and fusion models (same objects, same seeds),
    so paired DeLong tests are valid.  (Originally CNN: 70/15/15 of the image list; fusion: 80/20 then 85/15 of a
    different ordering => different test objects, and the fusion backbone had been fine-tuned on images that are in
    the fusion test split.)
  * Backbone BN layers of FROZEN blocks stay in eval() mode (originally their running stats kept updating).
  * Fusion head is trained from the *pretrained ImageNet* backbone, never from a checkpoint trained on other splits.
  * Weighted BCE + label smoothing as stated in the manuscript.
  * Saves test predictions as results/ft_predictions_<model>_<seed>.npy for analyze scripts.
Usage:  python train_finetune.py --task qso --mode img_gap --seed 42        # image-only EfficientNet-B0, fine-tuned
        python train_finetune.py --task qso --mode attn_gate --tab photo12 --seed 42
"""
import argparse, os, sys, time, numpy as np, pandas as pd, torch, torch.nn as nn
from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from efficientnet_pytorch import EfficientNet
sys.path.insert(0, os.path.dirname(__file__))
from data import split_70_15_15, FEATURE_SETS, ALL_FEATURES
from models_fusion import FusionNet

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAB = {"photo12": FEATURE_SETS["no_z_no_Mr"], "full14": ALL_FEATURES}
MEAN = np.array([0.485, 0.456, 0.406], np.float32).reshape(3, 1, 1); STD = np.array([0.229, 0.224, 0.225], np.float32).reshape(3, 1, 1)


class DS(torch.utils.data.Dataset):
    def __init__(self, df, idx, tab, size, train):
        self.df, self.idx, self.tab, self.size, self.train = df, idx, tab, size, train
    def __len__(self): return len(self.idx)
    def __getitem__(self, k):
        i = self.idx[k]; r = self.df.iloc[i]
        im = Image.open(r.img_path).convert("RGB").resize((self.size, self.size), Image.BICUBIC)
        x = (np.asarray(im, np.float32).transpose(2, 0, 1) / 255 - MEAN) / STD
        if self.train:   # SDSS cutouts have no preferred orientation -> flips + 90deg rotations are label-preserving
            if np.random.rand() < .5: x = x[:, :, ::-1]
            if np.random.rand() < .5: x = x[:, ::-1, :]
            x = np.rot90(x, np.random.randint(4), axes=(1, 2))
        return torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(self.tab[k]), torch.tensor(float(r.label))


class EndToEnd(nn.Module):
    def __init__(self, mode, tab_dim, weights, n_tokens=49):
        super().__init__()
        self.bb = EfficientNet.from_name("efficientnet-b0"); self.bb.load_state_dict(torch.load(weights, map_location="cpu"))
        self.fusion = FusionNet(mode, tab_dim, d_tok_in=1280, n_tokens=n_tokens)
    def forward(self, img, tab, lam=0.0):
        f = self.bb.extract_features(img)                          # (B,1280,h,w)
        tok = f.flatten(2).transpose(1, 2)                         # (B,h*w,1280)
        return self.fusion(tok.mean(1), tok, tab, lam)


def set_trainable(model, phase):
    for p in model.bb.parameters(): p.requires_grad = False
    if phase == 2:
        for blk in model.bb._blocks[-3:]:
            for p in blk.parameters(): p.requires_grad = True
        for p in model.bb._conv_head.parameters(): p.requires_grad = True
    for p in model.fusion.parameters(): p.requires_grad = True


def freeze_bn_in_frozen_parts(model):
    """keep BatchNorm of frozen backbone parts in eval mode (running stats must not drift)"""
    model.bb.eval()
    for blk in model.bb._blocks[-3:]:
        if any(p.requires_grad for p in blk.parameters()): blk.train()


def evaluate(model, loader, dev):
    model.eval(); ps, ys = [], []
    with torch.no_grad():
        for x, t, y in loader:
            lg, *_ = model(x.to(dev), t.to(dev)); ps.append(torch.sigmoid(lg).cpu()); ys.append(y)
    return torch.cat(ps).numpy(), torch.cat(ys).numpy()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="qso", choices=["qso", "bpt"], help="qso = matched QSO-vs-galaxy cohort (3,686 objects); bpt = BPT AGN vs star-forming")
    ap.add_argument("--mode", default="attn_gate", help="img_gap (image only) | concat | attn_nogate | gate_global | attn_gate")
    ap.add_argument("--tab", default="photo12"); ap.add_argument("--seed", type=int, default=42); ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--epochs", type=int, default=40); ap.add_argument("--phase1", type=int, default=10)
    ap.add_argument("--bs", type=int, default=8); ap.add_argument("--workers", type=int, default=0); ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    def cem(d, keys, seed=0):                 # identical to run_paper_experiments.py
        rng = np.random.default_rng(seed); key = pd.Series("", index=d.index)
        for c_, w in keys: key = key + "_" + np.floor(d[c_] / w).astype(int).astype(str)
        keep = []
        for _, g in d.groupby(key):
            q, gl = g.index[g.label == 1], g.index[g.label == 0]; n = min(len(q), len(gl))
            if n: keep += list(rng.choice(q, n, replace=False)) + list(rng.choice(gl, n, replace=False))
        return np.sort(np.array(keep))
    if a.task == "qso":
        df = pd.read_csv(f"{ROOT}/data/image_cohort.csv"); df["log_r50"] = np.log10(df.petroR50_r.clip(lower=0.05))
        cidx = cem(df, [("r", 0.5), ("log_r50", 0.1)]); folders = {1: "agn", 0: "galaxy"}; root_img = f"{ROOT}/data/images"
    else:
        df = pd.read_csv(f"{ROOT}/data/bpt_cohort_matched.csv"); cidx = np.arange(len(df)); folders = {1: "agn", 0: "sf"}; root_img = f"{ROOT}/data/images_bpt"
    df["img_path"] = [f"{root_img}/{folders[int(l)]}/{int(o)}.jpg" for o, l in zip(df.objid, df.label)]
    sub = df.iloc[cidx].reset_index(drop=True)
    tr_p, va_p, te_p = split_70_15_15(sub, a.seed); tr, va, te = cidx[tr_p], cidx[va_p], cidx[te_p]     # same splits as the frozen-feature experiments
    print(f"task={a.task} cohort={len(cidx)} train={len(tr)} val={len(va)} test={len(te)} device={dev}")
    if a.smoke: tr, va, te = tr[:64], va[:32], te[:32]; a.epochs, a.phase1 = 2, 1
    cols = TAB[a.tab]; sc = StandardScaler().fit(df[cols].values[tr]); X = sc.transform(df[cols].values).astype(np.float32)
    mk = lambda idx, train: torch.utils.data.DataLoader(DS(df, idx, X[idx], a.img_size, train), batch_size=a.bs, shuffle=train, num_workers=a.workers)
    ltr, lva, lte = mk(tr, True), mk(va, False), mk(te, False)
    model = EndToEnd(a.mode, X.shape[1], f"{ROOT}/data/effb0.pth", n_tokens=(a.img_size // 32) ** 2).to(dev)
    pw = torch.tensor(float((df.label.values[tr] == 0).sum() / (df.label.values[tr] == 1).sum())).to(dev)
    best, state, bad = -1, None, 0
    for ep in range(a.epochs):
        phase = 1 if ep < a.phase1 else 2
        if ep in (0, a.phase1):
            set_trainable(model, phase)
            opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=3e-4 if phase == 1 else 3e-5, weight_decay=1e-2)
        model.train(); freeze_bn_in_frozen_parts(model)
        for x, t, yb in ltr:
            x, t, yb = x.to(dev), t.to(dev), yb.to(dev)
            lg, *_ = model(x, t)
            loss = nn.functional.binary_cross_entropy_with_logits(lg, yb * 0.95 + 0.025, pos_weight=pw)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        pv, yv = evaluate(model, lva, dev); auc = roc_auc_score(yv, pv) if len(set(yv)) > 1 else 0.5
        print(f"ep {ep+1:02d} phase {phase} val AUC {auc:.4f}", flush=True)
        if auc > best + 1e-5: best, bad, state = auc, 0, {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= 8: break
    model.load_state_dict(state)
    pt, yt = evaluate(model, lte, dev)
    os.makedirs(f"{ROOT}/results", exist_ok=True)
    np.save(f"{ROOT}/results/ft_predictions_{a.task}_{a.mode}_{a.tab}_{a.seed}.npy", np.stack([te, pt]))
    print("TEST AUC", roc_auc_score(yt, pt) if len(set(yt)) > 1 else "n/a (smoke)")
