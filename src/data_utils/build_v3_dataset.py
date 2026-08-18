"""
LEGACY (4-class side-view dataset builder, v3 era).

Merges an existing train/valid dataset with a folder of new crops, then reshuffles
everything into a fresh 70/30 train/valid split (seeded).

This is the split logic used for the 4-class side-view classifier datasets. The current
bottom-ring and top-view datasets are 3-class and were split separately — this script
does not build them.

Usage:
    python src/data_utils/build_v3_dataset.py \\
        --new-crops <dir> --old-dataset <dir> --out <dir>

Copies files (never moves or deletes), so the inputs are left intact.
"""

import os
import shutil
import random
import argparse
from pathlib import Path

# ─── CONFIG ───────────────────────────────────────────────────────────────
_parser = argparse.ArgumentParser(description="Build a merged 70/30 classifier dataset")
_parser.add_argument("--new-crops",   required=True, help="Folder of new crops (<brand>/ subfolders)")
_parser.add_argument("--old-dataset", required=True, help="Existing dataset with train/ and valid/")
_parser.add_argument("--out",         required=True, help="Output dataset root")
_parser.add_argument("--train-pct",   type=float, default=0.70, help="Train fraction (default 0.70)")
_parser.add_argument("--seed",        type=int,   default=42,   help="Shuffle seed (default 42)")
_args = _parser.parse_args()

NEW_CROPS_BASE = _args.new_crops
OLD_DATASET    = _args.old_dataset
NEW_DATASET    = _args.out
BRANDS         = ["bharat", "hp", "indane", "unknown"]
TRAIN_PCT      = _args.train_pct
SEED           = _args.seed
# ──────────────────────────────────────────────────────────────────────────

random.seed(SEED)
VALID_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

def collect_images(folder):
    if not os.path.exists(folder):
        return []
    return [f for f in os.listdir(folder)
            if Path(f).suffix.lower() in VALID_EXTS]

# ── Step 1 — collect all images per brand (old + new) ────────────────────
all_images = {}

for brand in BRANDS:
    images = []

    # Old dataset — train
    old_train = os.path.join(OLD_DATASET, "train", brand)
    for f in collect_images(old_train):
        images.append((os.path.join(old_train, f), f"old_train_{f}"))

    # Old dataset — valid
    old_val = os.path.join(OLD_DATASET, "valid", brand)
    for f in collect_images(old_val):
        images.append((os.path.join(old_val, f), f"old_val_{f}"))

    # New YT crops
    new_crops = os.path.join(NEW_CROPS_BASE, brand)
    for f in collect_images(new_crops):
        images.append((os.path.join(new_crops, f), f"new_{f}"))

    all_images[brand] = images
    print(f"{brand}: {len(images)} total images")

# ── Step 2 — shuffle and split 70/30 ─────────────────────────────────────
print("\nSplitting 70/30...")
for brand in BRANDS:
    images = all_images[brand]
    random.shuffle(images)

    n_train = int(len(images) * TRAIN_PCT)
    train   = images[:n_train]
    val     = images[n_train:]

    # Create output folders
    train_dest = os.path.join(NEW_DATASET, "train", brand)
    val_dest   = os.path.join(NEW_DATASET, "valid", brand)
    os.makedirs(train_dest, exist_ok=True)
    os.makedirs(val_dest,   exist_ok=True)

    # Copy train
    for src_path, unique_name in train:
        dest = os.path.join(train_dest, unique_name)
        shutil.copy2(src_path, dest)

    # Copy val
    for src_path, unique_name in val:
        dest = os.path.join(val_dest, unique_name)
        shutil.copy2(src_path, dest)

    print(f"{brand}: {len(train)} train / {len(val)} val")

# ── Step 3 — final summary ────────────────────────────────────────────────
print("\n" + "="*45)
print("DATASET v3 CLEAN — FINAL COUNTS")
print("="*45)
total_train = total_val = 0
for split in ["train", "valid"]:
    for brand in BRANDS:
        folder = os.path.join(NEW_DATASET, split, brand)
        if os.path.exists(folder):
            count = len(os.listdir(folder))
            print(f"  {split}/{brand}: {count}")
            if split == "train":
                total_train += count
            else:
                total_val += count

print(f"\n  Total train: {total_train}")
print(f"  Total val:   {total_val}")
print(f"  Output:      {NEW_DATASET}")