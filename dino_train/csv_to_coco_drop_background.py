#!/usr/bin/env python3
"""
csv_to_coco_drop_background.py  (v2 – adds COCO “info” header)

* drops images without objects
* stratified 85/15 train/val split
* copies the remaining images
* writes fully‑compliant COCO JSON (info + licenses present)

Usage
-----
python csv_to_coco_drop_background.py \
       --src ./train \
       --dst ./train_coco \
       --val-ratio 0.15 \
       --seed 42
"""

from __future__ import annotations
import argparse, json, random, shutil, datetime as dt
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ------------------------- CLI -------------------------
ap = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
ap.add_argument("--src", type=Path, required=True,
                help="source root with images/ and annotations/")
ap.add_argument("--dst", type=Path, required=True,
                help="destination root (will be created fresh)")
ap.add_argument("--val-ratio", type=float, default=0.15,
                help="ratio of images for validation split")
ap.add_argument("--seed", type=int, default=42,
                help="random seed for reproducibility")
args = ap.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)

SRC_IMG_DIR = args.src / "images"
SRC_ANN_DIR = args.src / "annotations"
DST_ROOT    = args.dst.resolve()

# ----------------- 0. Sanity checks -------------------
assert SRC_IMG_DIR.is_dir() and SRC_ANN_DIR.is_dir(), \
       "source must contain images/ and annotations/"
if DST_ROOT.exists():
    raise SystemExit(f"{DST_ROOT} already exists – remove it first.")

# ----------------- 1. Merge CSVs ----------------------
dfs = [pd.read_csv(p) for p in sorted(SRC_ANN_DIR.glob("*.csv"))]
df  = pd.concat(dfs, ignore_index=True)

# keep only images that actually have at least one box
objects_per_img = df.groupby("image_id").size()
keep_ids        = objects_per_img[objects_per_img > 0].index
df              = df[df["image_id"].isin(keep_ids)].reset_index(drop=True)

# ----------------- 2. Stratified split ----------------
def majority_label(sub: pd.DataFrame) -> int:
    return Counter(sub["label"]).most_common(1)[0][0]

img_summary = (df.groupby("image_id")
                 .apply(majority_label)
                 .rename("major_label")
                 .reset_index())

train_ids, val_ids = train_test_split(
    img_summary["image_id"],
    test_size=args.val_ratio,
    random_state=args.seed,
    shuffle=True,
    stratify=img_summary["major_label"],
)


split_map: dict[str, str] = {iid: "train" for iid in train_ids}
split_map.update({iid: "val" for iid in val_ids})

# ----------------- 3. Make folders --------------------
for split in ("train", "val"):
    (DST_ROOT / split / "images").mkdir(parents=True)
(DST_ROOT / "annotations").mkdir(parents=True)

# ----------------- 4. Copy images ---------------------
print("Copying images …")
for img_id, split in tqdm(split_map.items()):
    matches = list(SRC_IMG_DIR.glob(f"{img_id}.*"))
    if not matches:
        print(f"⚠  missing image for id {img_id}, skipped.")
        continue
    shutil.copy2(matches[0], DST_ROOT / split / "images" / matches[0].name)

# ----------------- 5. Build COCO dicts ----------------
def build_info() -> dict:
    """Return a minimal but valid COCO ‘info’ block."""
    today = dt.date.today().isoformat()
    return {
        "description": "Mines object‑detection dataset",
        "version": "1.0",
        "contributor": "",
        "url": "",
        "date_created": today
    }

licenses = [{
    "id": 1,
    "name": "Unknown",
    "url": ""
}]

categories = [
    {"id": 1, "name": "Explosives",          "supercategory": "mine"},
    {"id": 2, "name": "Anti-personnel mine", "supercategory": "mine"},
    {"id": 3, "name": "Anti-vehicle mine",   "supercategory": "mine"},
]

coco = {
    s: {
        "info": build_info(),
        "licenses": licenses,
        "images": [],
        "annotations": [],
        "categories": categories
    }
    for s in ("train", "val")
}

def clamp_bbox(x, y, w, h, img_w, img_h):
    x0, y0 = max(0.0, x), max(0.0, y)
    x1, y1 = min(x + w, img_w), min(y + h, img_h)
    new_w, new_h = x1 - x0, y1 - y0
    return None if new_w <= 0 or new_h <= 0 else [x0, y0, new_w, new_h]

ann_id = 1
for img_id, sub in tqdm(df.groupby("image_id"), desc="Converting"):
    split = split_map[img_id]
    H = int(sub.iloc[0]["image_height"])
    W = int(sub.iloc[0]["image_width"])

    dst_img = next((DST_ROOT / split / "images").glob(f"{img_id}.*")).name
    coco[split]["images"].append({
        "id": img_id,
        "file_name": dst_img,
        "height": H,
        "width": W,
        "license": 1
    })

    for _, row in sub.iterrows():
        bbox = clamp_bbox(
            float(row["x"]), float(row["y"]),
            float(row["width"]), float(row["height"]),
            img_w=W, img_h=H
        )
        if bbox is None:
            continue
        x_c, y_c, w_c, h_c = bbox
        coco[split]["annotations"].append({
            "id": ann_id,
            "image_id": img_id,
            "category_id": int(row["label"]) + 1,
            "bbox": [x_c, y_c, w_c, h_c],
            "area": w_c * h_c,
            "iscrowd": 0
        })
        ann_id += 1

# ----------------- 6. Save JSON -----------------------
for split in ("train", "val"):
    out_file = DST_ROOT / "annotations" / f"instances_{split}.json"
    with open(out_file, "w") as fp:
        json.dump(coco[split], fp)
    print(f"✔  wrote {out_file}")

print("\nDone!  Point your MMDetection config to:", DST_ROOT)
