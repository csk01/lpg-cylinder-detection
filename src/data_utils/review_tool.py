"""
LEGACY (Roboflow-era workflow).

Tkinter tool for sorting full-cylinder crops into brand folders, with perceptual-hash
duplicate removal on load. Superseded by crop_bottom_ring.py for the current
bottom-ring dataset, but still usable for sorting whole-cylinder images.

Requires the `imagehash` package (not in requirements.txt — pip install imagehash).
Run locally only; tkinter will not work in Colab.

Usage:
    python src/data_utils/review_tool.py --review <dir> [--output <dir>]

Note: this tool MOVES and DELETES files in the review folder. Back up first.
"""

import os
import sys
import shutil
import hashlib
import argparse
import tkinter as tk
from PIL import Image, ImageTk

try:
    import imagehash
except ImportError:
    sys.exit("imagehash is required: pip install imagehash")

_parser = argparse.ArgumentParser(description="Review and sort cylinder crops by brand")
_parser.add_argument("--review", required=True, help="Folder of images to review")
_parser.add_argument("--output", default=None,
                     help="Output root for brand folders (default: parent of --review)")
_args = _parser.parse_args()

REVIEW_FOLDER = _args.review
OUTPUT_BASE   = _args.output or os.path.dirname(os.path.abspath(REVIEW_FOLDER))

if not os.path.isdir(REVIEW_FOLDER):
    sys.exit(f"Review folder not found: {REVIEW_FOLDER}")

CLASSES = ["bharat_gas", "hp_gas", "indane", "unknown"]
KEYS    = {"1": "bharat_gas", "2": "hp_gas", "3": "indane", "4": "unknown", "d": "DELETE", "s": "SKIP"}

def file_hash(fpath):
    with open(fpath, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

class ReviewTool:
    def __init__(self, root):
        self.root = root
        self.root.title("LPG Review Tool")
        self.root.geometry("1000x800")
        self.root.configure(bg="#1e1e1e")

        # Create output folders
        for brand in CLASSES:
            os.makedirs(f"{OUTPUT_BASE}/{brand}", exist_ok=True)
        dupes_folder = os.path.join(OUTPUT_BASE, "dupes")
        os.makedirs(dupes_folder, exist_ok=True)

        # Load images
        all_images = [
            f for f in os.listdir(REVIEW_FOLDER)
            if os.path.splitext(f)[1].lower() in {".jpg", ".jpeg", ".png"}
        ]

        # Perceptual hash with Hamming distance threshold
        print("Scanning for duplicates...")
        seen_hashes = []
        unique_images = []
        dupe_count = 0
        THRESHOLD = 8  # tune this — lower = stricter, higher = catches more similars

        for f in all_images:
            fpath = os.path.join(REVIEW_FOLDER, f)
            try:
                img = Image.open(fpath).convert("RGB")
                h = imagehash.phash(img)

                # Check against all seen hashes
                is_dupe = any((h - seen_h) <= THRESHOLD for seen_h in seen_hashes)

                if not is_dupe:
                    seen_hashes.append(h)
                    unique_images.append(f)
                else:
                    shutil.move(fpath, os.path.join(dupes_folder, f))
                    dupe_count += 1
            except Exception as e:
                print(f"Error hashing {f}: {e}")
                unique_images.append(f)

        print(f"Moved {dupe_count} duplicates to dupes/ folder.")
        print(f"{len(unique_images)} unique images to review.")

        self.images      = unique_images
        self.index       = 0
        self.total       = len(self.images)
        self.zoom        = 1.0
        self.current_img = None

        # ── UI ──────────────────────────────────────────────
        self.info_label = tk.Label(root, text="", font=("Helvetica", 12),
                                   bg="#1e1e1e", fg="white")
        self.info_label.pack(pady=6)

        self.canvas_frame = tk.Frame(root, bg="#1e1e1e")
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(self.canvas_frame, bg="#2d2d2d",
                                highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.key_label = tk.Label(
            root,
            text="1=Bharat Gas  |  2=HP Gas  |  3=Indane  |  4=Unknown  |  D=Delete  |  S=Skip  |  Scroll=Zoom  |  R=Reset zoom",
            font=("Helvetica", 10), bg="#1e1e1e", fg="#aaaaaa"
        )
        self.key_label.pack(pady=6)

        self.status_label = tk.Label(root, text="", font=("Helvetica", 11),
                                     bg="#1e1e1e", fg="#00ff88")
        self.status_label.pack(pady=4)

        # ── Bindings ─────────────────────────────────────────
        root.bind("<Key>", self.on_key)
        root.bind("<MouseWheel>", self.on_scroll)
        root.bind("<Button-4>", self.on_scroll)
        root.bind("<Button-5>", self.on_scroll)

        self.root.update()
        self.show_image()

    def show_image(self):
        if self.index >= self.total:
            self.info_label.config(text="✅ All images reviewed!")
            self.canvas.delete("all")
            return

        fname = self.images[self.index]
        fpath = os.path.join(REVIEW_FOLDER, fname)
        self.current_img = Image.open(fpath).convert("RGB")
        self.render_image()

    def render_image(self):
        if self.current_img is None:
            return

        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()

        if canvas_w < 10: canvas_w = 800
        if canvas_h < 10: canvas_h = 550

        if self.zoom == 1.0:
            ratio = min(canvas_w / max(self.current_img.width, 1),
                        canvas_h / max(self.current_img.height, 1))
            w = max(int(self.current_img.width  * ratio), 1)
            h = max(int(self.current_img.height * ratio), 1)
        else:
            w = max(int(self.current_img.width  * self.zoom), 1)
            h = max(int(self.current_img.height * self.zoom), 1)

        img_resized = self.current_img.resize((w, h), Image.LANCZOS)
        self.tk_img = ImageTk.PhotoImage(img_resized)

        self.canvas.delete("all")
        self.canvas.create_image(canvas_w // 2, canvas_h // 2,
                                  anchor=tk.CENTER, image=self.tk_img)

        fname = self.images[self.index] if self.index < self.total else ""
        self.info_label.config(
            text=f"[{self.index+1}/{self.total}]  {fname}   🔍 Zoom: {round(self.zoom*100)}%"
        )

    def on_scroll(self, event):
        if event.num == 4 or event.delta > 0:
            self.zoom = min(self.zoom * 1.15, 5.0)
        else:
            self.zoom = max(self.zoom / 1.15, 0.2)
        self.render_image()

    def on_key(self, event):
        key = event.char.lower()

        if key == "r":
            self.zoom = 1.0
            self.render_image()
            return

        if key not in KEYS:
            return

        fname  = self.images[self.index]
        src    = os.path.join(REVIEW_FOLDER, fname)
        action = KEYS[key]

        if action == "DELETE":
            os.remove(src)
            self.status_label.config(text="🗑️  Deleted", fg="#ff4444")
        elif action == "SKIP":
            self.status_label.config(text="⏭️  Skipped", fg="#aaaaaa")
        else:
            dest = os.path.join(OUTPUT_BASE, action, fname)
            shutil.move(src, dest)
            self.status_label.config(text=f"✅  Moved to {action}", fg="#00ff88")

        self.zoom  = 1.0
        self.index += 1
        self.show_image()

if __name__ == "__main__":
    root = tk.Tk()
    app  = ReviewTool(root)
    root.mainloop()