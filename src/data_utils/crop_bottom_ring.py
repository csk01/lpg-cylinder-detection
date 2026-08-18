"""
Manual bottom-ring crop + sort tool.

Draw a rectangle around the bottom ring, then press 1/2/3 to save that crop into the
matching brand folder. Multiple crops can be taken from a single source image.

Usage:
    python src/data_utils/crop_bottom_ring.py --input <src_dir> --output <out_dir>

Defaults can also be supplied via environment variables:
    LPG_CROP_SOURCE   source directory of images to review
    LPG_CROP_OUTPUT   output dataset root (brand subfolders are created inside)

Output structure:
    <output>/bharat/<name>_crop1.jpg
    <output>/hp/...
    <output>/indane/...
"""

import os
import argparse
import tkinter as tk
from PIL import Image, ImageTk
from pathlib import Path

# Repo root = .../lpg-cylinder-detection (this file lives in src/data_utils/)
REPO_ROOT = Path(__file__).resolve().parents[2]

# Project-relative defaults — override with --input/--output or the env vars above.
DEFAULT_SOURCE = os.environ.get(
    "LPG_CROP_SOURCE", str(REPO_ROOT / "datasets" / "raw" / "to_sort")
)
DEFAULT_OUTPUT = os.environ.get(
    "LPG_CROP_OUTPUT", str(REPO_ROOT / "datasets" / "lpg_bottom_ring_dataset_v1" / "cropped")
)

BRANDS = {
    "1": "indane",
    "2": "hp",
    "3": "bharat"
}

class CropSortTool:
    def __init__(self, root, source=DEFAULT_SOURCE, output=DEFAULT_OUTPUT):
        self.root = root
        self.source = source
        self.output = output
        self.root.title("Bottom Ring Crop + Sort")
        self.root.geometry("1100x850")
        self.root.configure(bg="#1e1e1e")

        if not os.path.isdir(self.source):
            raise SystemExit(
                f"Source directory not found: {self.source}\n"
                "Pass --input <dir> or set LPG_CROP_SOURCE."
            )

        for brand in BRANDS.values():
            os.makedirs(os.path.join(self.output, brand), exist_ok=True)

        self.images = [
            f for f in os.listdir(self.source)
            if Path(f).suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
        ]
        self.index      = 0
        self.total      = len(self.images)
        self.crop_count = 0   # crops saved from the current image (see save_crops)
        self.current_img = None
        self.tk_img      = None
        self.rotation    = 0
        self.rects       = []   # list of (rect_id, x1, y1, x2, y2)
        self.drawing     = False
        self.start       = None
        self.temp_rect   = None
        self.scale_x     = 1.0
        self.scale_y     = 1.0
        self.offset_x    = 0
        self.offset_y    = 0

        print(f"Found {self.total} images")

        # ── UI ────────────────────────────────────────────────────────────
        self.info = tk.Label(root, text="", font=("Arial", 12),
                             bg="#1e1e1e", fg="white")
        self.info.pack(pady=4)

        self.canvas = tk.Canvas(root, bg="#2d2d2d", width=900, height=620,
                                highlightthickness=0, cursor="crosshair")
        self.canvas.pack()

        # Rotation row
        rot_frame = tk.Frame(root, bg="#1e1e1e")
        rot_frame.pack(pady=4)
        tk.Label(rot_frame, text="Rotate:", font=("Arial", 10),
                 bg="#1e1e1e", fg="#aaaaaa").pack(side=tk.LEFT, padx=4)
        for lbl, deg in [("↺90", -90), ("↺45", -45), ("↻45", 45), ("↻90", 90)]:
            tk.Button(rot_frame, text=lbl, font=("Arial", 10),
                      bg="#2d2d2d", fg="white", relief="flat", padx=6,
                      command=lambda d=deg: self.rotate(d)).pack(side=tk.LEFT, padx=2)
        tk.Button(rot_frame, text="Reset", font=("Arial", 10),
                  bg="#444", fg="white", relief="flat", padx=6,
                  command=self.reset_rotation).pack(side=tk.LEFT, padx=6)
        self.rot_label = tk.Label(rot_frame, text="0°", font=("Arial", 10),
                                  bg="#1e1e1e", fg="#00ff88", width=4)
        self.rot_label.pack(side=tk.LEFT)

        # Hint row
        tk.Label(root,
                 text="Draw multiple regions  |  1=Indane  2=HP  3=Bharat  |  R=Redraw last  |  S=Skip  |  D=Delete image  |  ←→=Rotate",
                 font=("Arial", 10), bg="#1e1e1e", fg="#aaaaaa").pack(pady=2)

        self.status = tk.Label(root, text="", font=("Arial", 11),
                               bg="#1e1e1e", fg="#00ff88")
        self.status.pack()

        # ── Bindings ──────────────────────────────────────────────────────
        self.canvas.bind("<ButtonPress-1>",   self.on_down)
        self.canvas.bind("<B1-Motion>",       self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_up)
        root.bind("<Key>", self.on_key)

        self.root.update()
        self.show_image()

    # ── Image display ──────────────────────────────────────────────────────
    def show_image(self):
        if self.index >= self.total:
            self.info.config(text="✅ All done!")
            self.canvas.delete("all")
            return

        fname = self.images[self.index]
        fpath = os.path.join(self.source, fname)
        self.info.config(text=f"[{self.index+1}/{self.total}]  {fname}")
        self.current_img = Image.open(fpath).convert("RGB")
        self.rotation    = 0
        self.rot_label.config(text="0°")
        self.crop_count  = 0   # reset numbering for each new source image
        self.rects       = []
        self.temp_rect   = None
        self.render()

    def render(self):
        if self.current_img is None:
            return

        rotated = self.current_img.rotate(-self.rotation, expand=True)
        self.rotated_img = rotated

        cw, ch = 900, 620
        iw, ih = rotated.size
        scale  = min(cw / iw, ch / ih)
        nw, nh = int(iw * scale), int(ih * scale)

        self.scale_x  = iw / nw
        self.scale_y  = ih / nh
        self.offset_x = (cw - nw) // 2
        self.offset_y = (ch - nh) // 2

        display     = rotated.resize((nw, nh), Image.LANCZOS)
        self.tk_img = ImageTk.PhotoImage(display)

        self.canvas.delete("all")
        self.canvas.create_image(self.offset_x, self.offset_y,
                                  anchor=tk.NW, image=self.tk_img)

        # Redraw saved rects
        for (rid, x1, y1, x2, y2, label) in self.rects:
            new_rid = self.canvas.create_rectangle(
                x1, y1, x2, y2, outline="#00ff88", width=2)
            self.canvas.create_text(
                (x1+x2)//2, y1+10, text=label,
                fill="#00ff88", font=("Arial", 9, "bold"))

    # ── Mouse ──────────────────────────────────────────────────────────────
    def on_down(self, e):
        self.start = (e.x, e.y)
        if self.temp_rect:
            self.canvas.delete(self.temp_rect)

    def on_drag(self, e):
        if self.start:
            if self.temp_rect:
                self.canvas.delete(self.temp_rect)
            self.temp_rect = self.canvas.create_rectangle(
                self.start[0], self.start[1], e.x, e.y,
                outline="#ffaa00", width=2, dash=(4, 4))

    def on_up(self, e):
        self.end = (e.x, e.y)

    # ── Keyboard ───────────────────────────────────────────────────────────
    def on_key(self, event):
        key = event.char.lower()

        if key in ("1", "2", "3"):
            self.save_crops(key)
        elif key == "r":
            self.redraw_last()
        elif key == "s":
            self.skip()
        elif key == "d":
            self.delete_image()
        elif event.keysym == "Left":
            self.rotate(-45)
        elif event.keysym == "Right":
            self.rotate(45)

    def save_crops(self, key):
        if not self.start or not hasattr(self, "end"):
            self.status.config(text="⚠️  Draw a region first", fg="#ffaa00")
            return

        brand = BRANDS[key]
        fname = self.images[self.index]

        cx1 = min(self.start[0], self.end[0])
        cy1 = min(self.start[1], self.end[1])
        cx2 = max(self.start[0], self.end[0])
        cy2 = max(self.start[1], self.end[1])

        # Convert to image coords
        ix1 = max(0, int((cx1 - self.offset_x) * self.scale_x))
        iy1 = max(0, int((cy1 - self.offset_y) * self.scale_y))
        ix2 = min(self.rotated_img.width,  int((cx2 - self.offset_x) * self.scale_x))
        iy2 = min(self.rotated_img.height, int((cy2 - self.offset_y) * self.scale_y))

        if ix2 - ix1 < 20 or iy2 - iy1 < 20:
            self.status.config(text="⚠️  Too small — redraw", fg="#ffaa00")
            return

        crop = self.rotated_img.crop((ix1, iy1, ix2, iy2))

        # Unique filename if multiple crops from same image.
        # The counter must NOT be derived from len(self.rects): rotate() clears that list
        # and redraw_last() pops from it, which restarted numbering and silently
        # overwrote crops already saved from this image. Use a per-image counter and
        # skip past any filename that already exists on disk.
        stem = Path(fname).stem
        ext  = Path(fname).suffix
        self.crop_count += 1
        count = self.crop_count
        out_fname = f"{stem}_crop{count}{ext}"
        while os.path.exists(os.path.join(self.output, brand, out_fname)):
            self.crop_count += 1
            count = self.crop_count
            out_fname = f"{stem}_crop{count}{ext}"
        crop.save(os.path.join(self.output, brand, out_fname))

        # Store rect for display
        label = f"{brand} #{count}"
        self.rects.append((self.temp_rect, cx1, cy1, cx2, cy2, label))
        self.temp_rect = None
        self.start     = None

        # Redraw with label
        self.render()
        self.status.config(
            text=f"✅  Saved crop {count} → {brand}/{out_fname}  (draw more or press 1/2/3 for next crop)",
            fg="#00ff88")

    def redraw_last(self):
        if self.rects:
            self.rects.pop()
            self.render()
            self.status.config(text="↺  Last crop removed", fg="#aaaaaa")
        else:
            if self.temp_rect:
                self.canvas.delete(self.temp_rect)
                self.temp_rect = None
            self.start = None
            self.status.config(text="↺  Cleared", fg="#aaaaaa")

    def skip(self):
        self.status.config(text="⏭️  Skipped", fg="#aaaaaa")
        self.index += 1
        self.show_image()

    def delete_image(self):
        fname = self.images[self.index]
        os.remove(os.path.join(self.source, fname))
        self.status.config(text=f"🗑️  Deleted {fname}", fg="#ff4444")
        self.index += 1
        self.show_image()

    def rotate(self, deg):
        self.rotation = (self.rotation + deg) % 360
        self.rot_label.config(text=f"{self.rotation}°")
        self.rects = []
        self.temp_rect = None
        self.render()

    def reset_rotation(self):
        self.rotation = 0
        self.rot_label.config(text="0°")
        self.rects = []
        self.render()


def main():
    parser = argparse.ArgumentParser(description="Manual bottom-ring crop + sort tool")
    parser.add_argument("--input",  default=DEFAULT_SOURCE,
                        help=f"Source image folder (default: {DEFAULT_SOURCE})")
    parser.add_argument("--output", default=DEFAULT_OUTPUT,
                        help=f"Output dataset root (default: {DEFAULT_OUTPUT})")
    args = parser.parse_args()

    root = tk.Tk()
    CropSortTool(root, source=args.input, output=args.output)
    root.mainloop()


if __name__ == "__main__":
    main()