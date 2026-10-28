"""Download SDSS cutouts for the matched BPT cohort (narrow-line AGN vs star-forming).  Needs internet; resumable.
Default = the manuscript's specification: 128x128 px at 0.2 arcsec/px (25.6 arcsec field), NO contrast stretch.
(Your QSO-cohort downloader used 64 px at 0.4 arcsec/px = the same 25.6 arcsec field.)  Use --size 64 --scale 0.4 to mimic it.
  python src/download_bpt_images.py [--limit 50]      # try 50 first
Writes data/bpt_cohort.csv (the matched table) and data/images_bpt/{agn,sf}/<objid>.jpg"""
import argparse, io, os, sys, time, numpy as np, pandas as pd, requests
from PIL import Image
sys.path.insert(0, os.path.dirname(__file__))
import external_bpt
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); URL = "https://skyserver.sdss.org/dr17/SkyServerWS/ImgCutout/getjpeg"
ap = argparse.ArgumentParser(); ap.add_argument("--size", type=int, default=128); ap.add_argument("--scale", type=float, default=0.2)
ap.add_argument("--limit", type=int, default=0); ap.add_argument("--sleep", type=float, default=0.15); a = ap.parse_args()
d = external_bpt.prepare(pd.read_csv(f"{ROOT}/data/external_bpt_raw.csv")).reset_index(drop=True)
if a.limit: d = pd.concat([d[d.label == 1].head(a.limit // 2), d[d.label == 0].head(a.limit // 2)]).reset_index(drop=True)
d.to_csv(f"{ROOT}/data/bpt_cohort.csv", index=False); print("cohort", len(d), d.label.value_counts().to_dict(), f"| {a.size}px @ {a.scale}\"/px = {a.size*a.scale:.1f}\" field")
sess = requests.Session(); sess.headers["User-Agent"] = "AGN-selection-audit-research/1.0"; ok = fail = 0
for i, r in enumerate(d.itertuples(index=False), 1):
    out = f"{ROOT}/data/images_bpt/{'agn' if r.label == 1 else 'sf'}/{int(r.objid)}.jpg"; os.makedirs(os.path.dirname(out), exist_ok=True)
    if os.path.exists(out):
        try:
            with Image.open(out) as im: im.verify()
            ok += 1; continue
        except Exception: os.remove(out)
    for attempt in range(3):
        try:
            resp = sess.get(URL, params=dict(ra=float(r.ra), dec=float(r.dec), scale=a.scale, width=a.size, height=a.size), timeout=30); resp.raise_for_status()
            with Image.open(io.BytesIO(resp.content)) as im: im.convert("RGB").save(out, "JPEG", quality=95)
            ok += 1; break
        except Exception as e:
            if attempt == 2: fail += 1; print("FAILED", int(r.objid), e)
            time.sleep(2)
    if i % 50 == 0: print(f"{i}/{len(d)} ok={ok} failed={fail}", flush=True)
    time.sleep(a.sleep)
print(f"Finished. valid={ok} failed={fail}")
