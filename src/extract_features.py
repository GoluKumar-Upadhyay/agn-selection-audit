"""Extract frozen ImageNet EfficientNet-B0 spatial tokens (7x7x1280) for every cutout.

Why frozen?  It lets every fusion variant (image-only, tab-only, concat, attention, gate, ...)
be trained on *identical* image features, so differences between them cannot be caused by
backbone fine-tuning noise, and 5-10 seeds x 8 variants run in minutes on a CPU.
For the full fine-tuned model use train_finetune.py on a GPU.
"""
import os, sys, time, numpy as np, torch
from PIL import Image
sys.path.insert(0, os.path.dirname(__file__))
from data import load_clean
from efficientnet_pytorch import EfficientNet

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
torch.set_num_threads(os.cpu_count())
df = load_clean(os.path.join(ROOT, "data", "sdss_agn_dataset.csv"))
have = {"agn": set(f[:-4] for f in os.listdir(f"{ROOT}/data/images/agn")),
        "galaxy": set(f[:-4] for f in os.listdir(f"{ROOT}/data/images/galaxy"))}
oid = df.objid.astype(str)
sel = df[(oid.isin(have["agn"]) | oid.isin(have["galaxy"]))].copy()
sel["img_path"] = [f"{ROOT}/data/images/{'agn' if str(o) in have['agn'] else 'galaxy'}/{o}.jpg" for o in sel.objid]
# guard: image folder must agree with the label (agn folder == label 1)
assert ((sel.img_path.str.contains("/agn/")) == (sel.label == 1)).all(), "image folder/label mismatch"
sel = sel.reset_index(drop=True)
limit = int(sys.argv[1]) if len(sys.argv) > 1 else len(sel)
sel = sel.iloc[:limit]
print("matched image+tabular objects:", len(sel), sel.label.value_counts().to_dict())

m = EfficientNet.from_name("efficientnet-b0")
m.load_state_dict(torch.load(f"{ROOT}/data/effb0.pth", map_location="cpu")); m.eval()
mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1); std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
feats = np.zeros((len(sel), 1280, 7, 7), dtype=np.float16)
t0, B = time.time(), 32
with torch.no_grad():
    for s in range(0, len(sel), B):
        ims = [np.asarray(Image.open(p).convert("RGB").resize((224, 224), Image.BICUBIC), dtype=np.float32) / 255 for p in sel.img_path[s:s + B]]
        x = (torch.from_numpy(np.stack(ims)).permute(0, 3, 1, 2) - mean) / std
        feats[s:s + B] = m.extract_features(x).numpy().astype(np.float16)
        if (s // B) % 10 == 0:
            print(f"{s+B}/{len(sel)}  {time.time()-t0:.0f}s", flush=True)
np.save(f"{ROOT}/data/effb0_tokens.npy", feats)
sel.drop(columns=["img_path"]).to_csv(f"{ROOT}/data/image_cohort.csv", index=False)
print("saved", feats.shape, f"{time.time()-t0:.0f}s")
