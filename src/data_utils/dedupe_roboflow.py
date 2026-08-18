"""
LEGACY (Roboflow-era workflow).

Deduplicates Roboflow exports by collapsing augmented variants that share a base name
(everything before the `.rf.<hash>` suffix). Kept for reference — the current
bottom-ring/top-view datasets are not sourced from Roboflow.

Usage:
    python src/data_utils/dedupe_roboflow.py --source <dir> [--dupes <dir>]

Moves (does not delete) duplicates into the dupes folder.
"""

import os
import sys
import shutil
import argparse
from pathlib import Path
from collections import defaultdict

# ─── CONFIG ───────────────────────────────────────────────────────────────────
_parser = argparse.ArgumentParser(description="Deduplicate a Roboflow export folder")
_parser.add_argument("--source", required=True, help="Folder of Roboflow images")
_parser.add_argument("--dupes", default=None,
                     help="Where to move duplicates (default: <source>_dupes)")
_args = _parser.parse_args()

SOURCE_FOLDER = _args.source
DUPES_FOLDER  = _args.dupes or (SOURCE_FOLDER.rstrip("/\\") + "_dupes")

if not os.path.isdir(SOURCE_FOLDER):
    sys.exit(f"Source folder not found: {SOURCE_FOLDER}")
# ─────────────────────────────────────────────────────────────────────────────

def get_base_name(fname):
    """
    Extract original image name before .rf. augmentation hash
    e.g. 'GasCylinder__IMG-20250306_angle_355.rf.026eb4b4.jpg' 
      -> 'GasCylinder__IMG-20250306_angle_355'
    """
    stem = Path(fname).stem  # remove extension
    if ".rf." in stem:
        return stem.split(".rf.")[0]
    return stem  # no rf hash — keep as is

def dedupe():
    os.makedirs(DUPES_FOLDER, exist_ok=True)

    all_files = [
        f for f in os.listdir(SOURCE_FOLDER)
        if Path(f).suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    ]

    print(f"Total images: {len(all_files)}")

    # Group by base name
    groups = defaultdict(list)
    for f in all_files:
        base = get_base_name(f)
        groups[base].append(f)

    kept    = 0
    moved   = 0
    singles = 0

    for base, files in groups.items():
        if len(files) == 1:
            singles += 1
            kept += 1
            continue

        # Keep first, move rest to dupes
        keep = files[0]
        dupes = files[1:]
        kept += 1

        for f in dupes:
            src  = os.path.join(SOURCE_FOLDER, f)
            dest = os.path.join(DUPES_FOLDER, f)
            shutil.move(src, dest)
            moved += 1

    print(f"\n{'='*45}")
    print(f"DEDUPE COMPLETE")
    print(f"{'='*45}")
    print(f"  Unique base images:  {len(groups)}")
    print(f"  Singles (no dupes):  {singles}")
    print(f"  Kept:                {kept}")
    print(f"  Moved to dupes/:     {moved}")
    print(f"  Remaining in source: {kept}")

if __name__ == "__main__":
    dedupe()