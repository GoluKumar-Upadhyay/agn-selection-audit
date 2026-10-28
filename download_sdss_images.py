import argparse
import io
import time
from pathlib import Path

import pandas as pd
import requests
from PIL import Image

ROOT = Path(__file__).resolve().parent
CSV = ROOT / "data" / "sdss_agn_dataset.csv"
IMAGE_ROOT = ROOT / "data" / "images"
URL = "https://skyserver.sdss.org/dr17/SkyServerWS/ImgCutout/getjpeg"

parser = argparse.ArgumentParser()
parser.add_argument("--limit", type=int, default=0,
                    help="Maximum objects to process; 0 means all")
args = parser.parse_args()

df = pd.read_csv(CSV).drop_duplicates("objid")
df = df.dropna(subset=["objid", "ra", "dec", "label"])
df = df[df["label"].isin([0, 1])]

if args.limit > 0:
    df = df.head(args.limit)

session = requests.Session()
session.headers["User-Agent"] = "AGN-selection-audit-research/1.0"

success = 0
failed = 0

for i, row in enumerate(df.itertuples(index=False), start=1):
    objid = str(int(row.objid))
    label = int(row.label)
    folder = IMAGE_ROOT / ("agn" if label == 1 else "galaxy")
    folder.mkdir(parents=True, exist_ok=True)
    output = folder / f"{objid}.jpg"

    if output.exists():
        try:
            with Image.open(output) as im:
                im.verify()
            success += 1
            continue
        except Exception:
            output.unlink(missing_ok=True)

    params = {
        "ra": float(row.ra),
        "dec": float(row.dec),
        "scale": 0.4,
        "width": 64,
        "height": 64,
    }

    try:
        response = session.get(URL, params=params, timeout=30)
        response.raise_for_status()

        with Image.open(io.BytesIO(response.content)) as im:
            im.convert("RGB").save(output, format="JPEG", quality=95)

        success += 1
    except Exception as exc:
        failed += 1
        output.unlink(missing_ok=True)
        print(f"FAILED {objid}: {exc}")

    if i % 25 == 0:
        print(f"Processed {i}/{len(df)}; downloaded/valid={success}; failed={failed}")

    time.sleep(0.15)

print(f"\nFinished. Valid images: {success}; failures: {failed}")
print("Image directory:", IMAGE_ROOT)
