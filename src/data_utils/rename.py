"""
LEGACY. Batch-renames raw brand folders to `<brand>_<n>.<ext>` in place.

Destructive: renames files where they sit. Defaults to a dry run — pass --apply to
actually rename.

Usage:
    python src/data_utils/rename.py --base <dir>            # preview only
    python src/data_utils/rename.py --base <dir> --apply    # perform renames

Expects <base>/{indane,hp,bharat}/ subfolders.
"""

import os
import sys
import argparse
from pathlib import Path

_parser = argparse.ArgumentParser(description="Batch rename brand image folders")
_parser.add_argument("--base", required=True, help="Folder containing brand subfolders")
_parser.add_argument("--apply", action="store_true",
                     help="Actually rename (default is a dry run)")
_args = _parser.parse_args()

BASE = _args.base
APPLY = _args.apply

if not os.path.isdir(BASE):
    sys.exit(f"Base folder not found: {BASE}")

if not APPLY:
    print("DRY RUN — no files will be renamed. Pass --apply to commit.\n")

brand_map = {
    "indane":    "indane",
    "hp":        "hp",
    "bharat":    "bharat"
}

for folder, prefix in brand_map.items():
    folder_path = os.path.join(BASE, folder)
    if not os.path.exists(folder_path):
        print(f"Skipping {folder} — not found")
        continue

    files = [f for f in os.listdir(folder_path)
             if Path(f).suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}]

    print(f"\n{folder}: {len(files)} images")

    for i, fname in enumerate(files, 1):
        ext     = Path(fname).suffix.lower()
        new_name = f"{prefix}_{i}{ext}"
        src     = os.path.join(folder_path, fname)
        dst     = os.path.join(folder_path, new_name)
        if src == dst:
            continue
        # Never clobber an existing file — the target name may already be taken by
        # another image in this folder.
        if os.path.exists(dst):
            print(f"  SKIP {fname} → {new_name} (target already exists)")
            continue
        if APPLY:
            os.rename(src, dst)
        print(f"  {fname} → {new_name}")

print("\nDone!")