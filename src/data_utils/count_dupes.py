"""
Count exact (MD5-identical) duplicate images in a folder. Read-only — reports only.

Usage:
    python src/data_utils/count_dupes.py --folder <dir>
"""

import os
import sys
import hashlib
import argparse
from pathlib import Path
from collections import defaultdict

parser = argparse.ArgumentParser(description="Count exact duplicate images in a folder")
parser.add_argument("--folder", required=True, help="Folder of images to scan")
FOLDER = parser.parse_args().folder

if not os.path.isdir(FOLDER):
    sys.exit(f"Folder not found: {FOLDER}")

def md5(fpath):
    with open(fpath, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

files = [f for f in os.listdir(FOLDER)
         if Path(f).suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}]

print(f"Total files: {len(files)}")

hash_groups = defaultdict(list)
for fname in files:
    fpath = os.path.join(FOLDER, fname)
    h = md5(fpath)
    hash_groups[h].append(fname)

exact_dupes = {h: names for h, names in hash_groups.items() if len(names) > 1}

print(f"Unique images: {len(hash_groups)}")
print(f"Exact dupe groups: {len(exact_dupes)}")
print(f"Redundant files: {sum(len(v)-1 for v in exact_dupes.values())}")

if exact_dupes:
    print("\nDupe groups:")
    for h, names in list(exact_dupes.items())[:25]:  # show first 25
        print(f"  {names}")