"""Frozen EfficientNet-B0 tokens (7x7x1280) for the BPT cohort -> data/bpt_tokens.npy + data/bpt_cohort_matched.csv (row-aligned).
Needs data/effb0.pth (same weights as before).  Works with any image size (resized to 224)."""
import os, sys, time, numpy as np, pandas as pd, torch
from PIL import Image
from efficientnet_pytorch import EfficientNet
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); torch.set_num_threads(os.cpu_count())
d = pd.read_csv(f"{ROOT}/data/bpt_cohort.csv"); d["img_path"] = [f"{ROOT}/data/images_bpt/{'agn' if l == 1 else 'sf'}/{int(o)}.jpg" for o, l in zip(d.objid, d.label)]
d = d[[os.path.exists(p) for p in d.img_path]].reset_index(drop=True); print("images found:", len(d), d.label.value_counts().to_dict())
m = EfficientNet.from_name("efficientnet-b0"); m.load_state_dict(torch.load(f"{ROOT}/data/effb0.pth", map_location="cpu")); m.eval()
mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1); std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
F = np.zeros((len(d), 1280, 7, 7), np.float16); t0 = time.time()
with torch.no_grad():
    for s in range(0, len(d), 32):
        ims = [np.asarray(Image.open(p).convert("RGB").resize((224, 224), Image.BICUBIC), np.float32) / 255 for p in d.img_path[s:s + 32]]
        F[s:s + 32] = m.extract_features((torch.from_numpy(np.stack(ims)).permute(0, 3, 1, 2) - mean) / std).numpy().astype(np.float16)
        if (s // 32) % 10 == 0: print(f"{min(s+32,len(d))}/{len(d)}  {time.time()-t0:.0f}s", flush=True)
np.save(f"{ROOT}/data/bpt_tokens.npy", F); d.drop(columns=["img_path"]).to_csv(f"{ROOT}/data/bpt_cohort_matched.csv", index=False); print("saved", F.shape)
