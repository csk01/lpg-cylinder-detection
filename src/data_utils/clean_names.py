"""
LEGACY (Roboflow-era workflow).

Strips Roboflow `.rf.<hash>` suffixes from filenames, then opens a tkinter resolver for
name collisions (keep new / keep existing / keep both).

Usage:
    python src/data_utils/clean_names.py --folder <dir> [--dupes <dir>]

Note: renames files in place and MOVES losers into the dupes folder. Back up first.
"""

import os
import re
import sys
import shutil
import argparse
import tkinter as tk
from PIL import Image, ImageTk
from pathlib import Path

_parser = argparse.ArgumentParser(description="Clean Roboflow hashes from filenames")
_parser.add_argument("--folder", required=True, help="Folder of images to clean")
_parser.add_argument("--dupes", default=None,
                     help="Where to move duplicates (default: <folder>_dupes)")
_args = _parser.parse_args()

FOLDER       = _args.folder
DUPES_FOLDER = _args.dupes or (FOLDER.rstrip("/\\") + "_dupes")

if not os.path.isdir(FOLDER):
    sys.exit(f"Folder not found: {FOLDER}")

os.makedirs(DUPES_FOLDER, exist_ok=True)

def clean_name(fname):
    cleaned = re.sub(r'\.rf\.[a-f0-9]+', '', fname)
    if '__' in cleaned:
        cleaned = cleaned.split('__', 1)[1]
    return cleaned

# ── Collect all renames needed ─────────────────────────────────────────────
files = [f for f in os.listdir(FOLDER)
         if Path(f).suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}]

conflicts = []   # (original_path, clean_name, existing_path)
safe      = []   # (original_path, clean_name)

for fname in files:
    fpath    = os.path.join(FOLDER, fname)
    new_name = clean_name(fname)
    new_path = os.path.join(FOLDER, new_name)

    if new_name == fname:
        continue  # already clean

    if os.path.exists(new_path) and new_path != fpath:
        conflicts.append((fpath, new_name, new_path))
    else:
        safe.append((fpath, new_name))

print(f"Safe renames:  {len(safe)}")
print(f"Conflicts:     {len(conflicts)}")

# ── Do safe renames immediately ────────────────────────────────────────────
for fpath, new_name in safe:
    new_path = os.path.join(FOLDER, new_name)
    if os.path.exists(new_path):
        shutil.move(fpath, os.path.join(DUPES_FOLDER, Path(fpath).name))
    else:
        os.rename(fpath, new_path)
print("Safe renames done!")

if not conflicts:
    print("No conflicts — all done!")
    exit()

# ── Visual conflict resolver ───────────────────────────────────────────────
class ConflictResolver:
    def __init__(self, root, conflicts):
        self.root      = root
        self.conflicts = conflicts
        self.index     = 0
        self.total     = len(conflicts)

        self.root.title("Conflict Resolver")
        self.root.geometry("1100x700")
        self.root.configure(bg="#1e1e1e")

        self.info = tk.Label(root, text="", font=("Helvetica", 12),
                             bg="#1e1e1e", fg="white")
        self.info.pack(pady=8)

        # Two image panels
        frame = tk.Frame(root, bg="#1e1e1e")
        frame.pack(fill=tk.BOTH, expand=True)

        self.left_label  = tk.Label(frame, text="NEW", font=("Helvetica", 11),
                                    bg="#1e1e1e", fg="#aaaaaa")
        self.left_label.grid(row=0, column=0, padx=20)

        self.right_label = tk.Label(frame, text="EXISTING", font=("Helvetica", 11),
                                    bg="#1e1e1e", fg="#aaaaaa")
        self.right_label.grid(row=0, column=1, padx=20)

        self.left_canvas  = tk.Canvas(frame, width=480, height=480,
                                      bg="#2d2d2d", highlightthickness=0)
        self.left_canvas.grid(row=1, column=0, padx=20, pady=10)

        self.right_canvas = tk.Canvas(frame, width=480, height=480,
                                      bg="#2d2d2d", highlightthickness=0)
        self.right_canvas.grid(row=1, column=1, padx=20, pady=10)

        btn_frame = tk.Frame(root, bg="#1e1e1e")
        btn_frame.pack(pady=10)

        tk.Button(btn_frame, text="K - Keep NEW (move existing to dupes)",
                  font=("Helvetica", 11), bg="#2ecc71", fg="white",
                  command=self.keep_new, width=35).grid(row=0, column=0, padx=10)

        tk.Button(btn_frame, text="E - Keep EXISTING (move new to dupes)",
                  font=("Helvetica", 11), bg="#e74c3c", fg="white",
                  command=self.keep_existing, width=35).grid(row=0, column=1, padx=10)

        tk.Button(btn_frame, text="B - Keep BOTH (rename new with suffix)",
                  font=("Helvetica", 11), bg="#3498db", fg="white",
                  command=self.keep_both, width=35).grid(row=1, column=0, columnspan=2,
                                                          pady=8)

        self.status = tk.Label(root, text="", font=("Helvetica", 11),
                               bg="#1e1e1e", fg="#00ff88")
        self.status.pack()

        root.bind("<k>", lambda e: self.keep_new())
        root.bind("<e>", lambda e: self.keep_existing())
        root.bind("<b>", lambda e: self.keep_both())

        self.root.update()
        self.show_conflict()

    def show_conflict(self):
        if self.index >= self.total:
            self.info.config(text="✅ All conflicts resolved!")
            self.left_canvas.delete("all")
            self.right_canvas.delete("all")
            return

        new_path, new_name, existing_path = self.conflicts[self.index]

        self.info.config(
            text=f"[{self.index+1}/{self.total}]  Conflict: {new_name}"
        )

        self._show_img(self.left_canvas,  new_path,      "left")
        self._show_img(self.right_canvas, existing_path, "right")

    def _show_img(self, canvas, path, side):
        try:
            img = Image.open(path).convert("RGB")
            img.thumbnail((460, 460))
            tk_img = ImageTk.PhotoImage(img)
            canvas.delete("all")
            canvas.create_image(240, 240, anchor=tk.CENTER, image=tk_img)
            if side == "left":
                self.left_tk  = tk_img
            else:
                self.right_tk = tk_img
        except Exception as e:
            print(f"Error loading {path}: {e}")

    def keep_new(self):
        new_path, new_name, existing_path = self.conflicts[self.index]
        shutil.move(existing_path, os.path.join(DUPES_FOLDER, Path(existing_path).name))
        os.rename(new_path, os.path.join(FOLDER, new_name))
        self.status.config(text="✅ Kept NEW", fg="#2ecc71")
        self._next()

    def keep_existing(self):
        new_path, new_name, existing_path = self.conflicts[self.index]
        shutil.move(new_path, os.path.join(DUPES_FOLDER, Path(new_path).name))
        self.status.config(text="✅ Kept EXISTING", fg="#e74c3c")
        self._next()

    def keep_both(self):
        new_path, new_name, existing_path = self.conflicts[self.index]
        stem    = Path(new_name).stem
        ext     = Path(new_name).suffix
        renamed = f"{stem}_{self.index}{ext}"
        os.rename(new_path, os.path.join(FOLDER, renamed))
        self.status.config(text=f"✅ Kept both — renamed to {renamed}", fg="#3498db")
        self._next()

    def _next(self):
        self.index += 1
        self.show_conflict()

if __name__ == "__main__":
    if conflicts:
        root = tk.Tk()
        app  = ConflictResolver(root, conflicts)
        root.mainloop()
    else:
        print("No conflicts to resolve!")