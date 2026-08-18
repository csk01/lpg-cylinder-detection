"""
LPG Cylinder Crop Tool
Usage:
  python crop_tool.py --auto   --input <src_dir> --output <out_dir>
  python crop_tool.py --manual --input <src_dir> --output <out_dir>
  python crop_tool.py --auto --manual --input <src_dir> --output <out_dir>
"""

import os
import sys
import shutil
import argparse
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
# Resolved relative to the repo root so this works on any machine/checkout.
REPO_ROOT     = Path(__file__).resolve().parents[2]
DETECTOR_PATH = Path(os.environ.get(
    "LPG_DETECTOR_PATH", REPO_ROOT / "models" / "yolov11x_lpg_v1_best.pt"
))
CONF          = 0.55

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


# ── Auto-crop ──────────────────────────────────────────────────────────────────

def run_autocrop(src_dir, auto_dir, manual_needed_dir):
    from ultralytics import YOLO
    from PIL import Image

    for d in (auto_dir, manual_needed_dir):
        d.mkdir(parents=True, exist_ok=True)

    images = sorted(f for f in src_dir.iterdir() if f.suffix.lower() in IMAGE_EXTS)
    if not images:
        print(f"No images found in {src_dir}")
        return

    print(f"Loading detector...")
    model = YOLO(str(DETECTOR_PATH))
    print(f"Processing {len(images)} images...\n")

    auto_count   = 0
    manual_count = 0

    for i, img_path in enumerate(images, 1):
        print(f"  [{i:03d}/{len(images)}] {img_path.name}", end="  ")

        results = model(str(img_path), conf=CONF, verbose=False)
        boxes   = results[0].boxes
        n       = len(boxes)
        img     = Image.open(img_path).convert("RGB")

        if n == 0:
            shutil.copy(img_path, manual_needed_dir / img_path.name)
            manual_count += 1
            print("→ manual needed (0 detections)")

        elif n == 1:
            box  = boxes[0].xyxy[0].cpu().numpy()
            crop = _crop(img, box)
            if crop is None:
                shutil.copy(img_path, manual_needed_dir / img_path.name)
                manual_count += 1
                print("→ manual needed (crop too small)")
            else:
                crop.save(auto_dir / img_path.name)
                auto_count += 1
                print("→ auto-cropped")

        else:
            saved = 0
            for j, b in enumerate(boxes):
                box  = b.xyxy[0].cpu().numpy()
                crop = _crop(img, box)
                if crop is None:
                    continue
                out = auto_dir / f"{img_path.stem}_crop{j}{img_path.suffix}"
                crop.save(out)
                saved += 1
            auto_count += saved
            print(f"→ {saved} crops saved")

    print(f"\n{'='*45}")
    print(f"AUTO-CROP COMPLETE")
    print(f"{'='*45}")
    print(f"  Auto-cropped:    {auto_count}")
    print(f"  Manual needed:   {manual_count}")
    print(f"  Total images:    {len(images)}")
    print(f"\n  Auto crops    → {auto_dir}")
    print(f"  Manual needed → {manual_needed_dir}")


def _crop(img, box):
    x1 = max(0, min(img.width,  int(min(box[0], box[2]))))
    y1 = max(0, min(img.height, int(min(box[1], box[3]))))
    x2 = max(0, min(img.width,  int(max(box[0], box[2]))))
    y2 = max(0, min(img.height, int(max(box[1], box[3]))))
    if x2 - x1 < 10 or y2 - y1 < 10:
        return None
    return img.crop((x1, y1, x2, y2))


# ── Manual crop GUI ────────────────────────────────────────────────────────────

class ManualCropTool:

    def __init__(self, manual_needed_dir, manual_dir):
        import tkinter as tk
        from PIL import Image, ImageTk

        self.tk       = tk
        self.Image    = Image
        self.ImageTk  = ImageTk
        self.manual_dir = manual_dir

        manual_dir.mkdir(parents=True, exist_ok=True)

        self.images = sorted(
            f for f in manual_needed_dir.iterdir()
            if f.suffix.lower() in IMAGE_EXTS
        )
        if not self.images:
            print(f"No images in {manual_needed_dir} — run auto mode first.")
            return

        self.idx       = 0
        self.saved     = 0
        self.deleted   = 0

        # Rectangle draw state
        self.rect_start = None
        self.rect_end   = None
        self.rect_id    = None

        # Image display state
        self.photo      = None
        self.scale      = 1.0
        self.img_offset = (0, 0)
        self.orig_img   = None

        self.root = tk.Tk()
        self.root.configure(bg="black")
        self.root.attributes("-fullscreen", True)

        self._build_ui()
        self.root.update()          # let window render before measuring canvas
        self._load_current()
        self.root.mainloop()

        skipped = self.idx - self.saved - self.deleted
        print(f"\n{'='*45}")
        print(f"MANUAL CROP COMPLETE")
        print(f"{'='*45}")
        print(f"  Saved:   {self.saved}")
        print(f"  Deleted: {self.deleted}")
        print(f"  Skipped: {skipped}")
        print(f"\n  Manual crops → {self.manual_dir}")

    # ── UI construction ────────────────────────────────────────────────────────

    def _build_ui(self):
        tk = self.tk

        self.status_var = tk.StringVar()
        tk.Label(
            self.root, textvariable=self.status_var,
            bg="#1a1a1a", fg="white", font=("Arial", 13),
            anchor="w", padx=12, pady=4
        ).pack(side=tk.TOP, fill=tk.X)

        self.canvas = tk.Canvas(
            self.root, bg="black", cursor="crosshair", highlightthickness=0
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)

        tk.Label(
            self.root,
            text="  ENTER / S: save crop     R: redraw     D: delete     N: skip     ESC: quit",
            bg="#1a1a1a", fg="#888888", font=("Arial", 11),
            anchor="w", padx=12, pady=3
        ).pack(side=tk.BOTTOM, fill=tk.X)

        # Mouse
        self.canvas.bind("<ButtonPress-1>",   self._on_press)
        self.canvas.bind("<B1-Motion>",       self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)

        # Keyboard
        for key in ("<Return>", "s", "S"):
            self.root.bind(key, self._save)
        for key in ("r", "R"):
            self.root.bind(key, self._reset_rect)
        for key in ("d", "D"):
            self.root.bind(key, self._delete)
        for key in ("n", "N"):
            self.root.bind(key, self._skip)
        self.root.bind("<Escape>", lambda _: self.root.destroy())

    # ── Image loading ──────────────────────────────────────────────────────────

    def _load_current(self):
        if self.idx >= len(self.images):
            self.root.destroy()
            return

        path = self.images[self.idx]
        self.orig_img = self.Image.open(path).convert("RGB")
        self._reset_rect(None)
        self._render_image()
        self.status_var.set(
            f"  [{self.idx + 1} / {len(self.images)}]   {path.name}"
            f"   —   saved: {self.saved}   deleted: {self.deleted}"
        )

    def _render_image(self):
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw < 10 or ch < 10:
            cw = self.root.winfo_screenwidth()
            ch = self.root.winfo_screenheight() - 80

        iw, ih = self.orig_img.size
        self.scale = min(cw / iw, ch / ih)
        nw = int(iw * self.scale)
        nh = int(ih * self.scale)
        self.img_offset = ((cw - nw) // 2, (ch - nh) // 2)

        resized    = self.orig_img.resize((nw, nh), self.Image.LANCZOS)
        self.photo = self.ImageTk.PhotoImage(resized)

        self.canvas.delete("all")
        ox, oy = self.img_offset
        self.canvas.create_image(ox, oy, anchor="nw", image=self.photo)

    # ── Mouse events ───────────────────────────────────────────────────────────

    def _on_press(self, event):
        self.rect_start = (event.x, event.y)
        if self.rect_id:
            self.canvas.delete(self.rect_id)
            self.rect_id = None

    def _on_drag(self, event):
        if not self.rect_start:
            return
        if self.rect_id:
            self.canvas.delete(self.rect_id)
        self.rect_end = (event.x, event.y)
        self.rect_id  = self.canvas.create_rectangle(
            *self.rect_start, *self.rect_end,
            outline="#00ff00", width=2
        )

    def _on_release(self, event):
        self.rect_end = (event.x, event.y)

    # ── Canvas → original image coordinates ───────────────────────────────────

    def _to_orig(self, cx, cy):
        ox, oy     = self.img_offset
        iw, ih     = self.orig_img.size
        x = max(0, min(iw, int((cx - ox) / self.scale)))
        y = max(0, min(ih, int((cy - oy) / self.scale)))
        return x, y

    # ── Actions ────────────────────────────────────────────────────────────────

    def _save(self, event=None):
        if not (self.rect_start and self.rect_end):
            self.status_var.set(
                f"  [{self.idx + 1}/{len(self.images)}]  "
                "Draw a rectangle first, then press S or ENTER."
            )
            return

        x1, y1 = self._to_orig(*self.rect_start)
        x2, y2 = self._to_orig(*self.rect_end)
        x1, x2 = sorted([x1, x2])
        y1, y2 = sorted([y1, y2])

        if x2 - x1 < 10 or y2 - y1 < 10:
            self.status_var.set(
                f"  [{self.idx + 1}/{len(self.images)}]  Rectangle too small — redraw."
            )
            return

        crop     = self.orig_img.crop((x1, y1, x2, y2))
        out_path = self.manual_dir / self.images[self.idx].name
        crop.save(out_path)
        self.saved += 1

        self._show_preview(crop)
        self.idx += 1
        self._load_current()

    def _show_preview(self, crop):
        tk  = self.tk
        win = tk.Toplevel(self.root)
        win.configure(bg="black")
        win.overrideredirect(True)

        # Position bottom-right
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        win.geometry(f"+{sw - 340}+{sh - 380}")

        preview = crop.copy()
        preview.thumbnail((300, 300))
        photo = self.ImageTk.PhotoImage(preview)

        tk.Label(win, image=photo, bg="black").pack(padx=8, pady=(8, 2))
        photo_ref = photo          # keep reference

        tk.Label(
            win, text="Saved!", bg="black", fg="#00cc44",
            font=("Arial", 14, "bold")
        ).pack(pady=(2, 8))

        win.after(1200, win.destroy)
        win.update()
        win._photo = photo_ref     # prevent GC

    def _reset_rect(self, event):
        self.rect_start = None
        self.rect_end   = None
        if self.rect_id:
            self.canvas.delete(self.rect_id)
            self.rect_id = None

    def _delete(self, event=None):
        # Remove from manual_needed — original is still in hp_new_snips
        try:
            self.images[self.idx].unlink()
        except FileNotFoundError:
            pass
        self.deleted += 1
        self.idx += 1
        self._load_current()

    def _skip(self, event=None):
        self.idx += 1
        self._load_current()


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    global DETECTOR_PATH   # run_autocrop() reads this module-level default
    parser = argparse.ArgumentParser(description="LPG Cylinder Crop Tool")
    parser.add_argument("--auto",   action="store_true", help="Run YOLO batch crop")
    parser.add_argument("--manual", action="store_true", help="Run tkinter manual crop GUI")
    parser.add_argument("--input",  required=True, type=Path, help="Source image folder")
    parser.add_argument("--output", required=True, type=Path, help="Output root folder")
    parser.add_argument("--detector", type=Path, default=None,
                        help=f"YOLO detector weights (default: {DETECTOR_PATH})")
    args = parser.parse_args()

    if not args.auto and not args.manual:
        parser.error("Specify at least one of --auto or --manual")

    if args.detector is not None:
        DETECTOR_PATH = args.detector

    if args.auto and not DETECTOR_PATH.exists():
        parser.error(
            f"Detector weights not found: {DETECTOR_PATH}\n"
            "Pass --detector <path> or set LPG_DETECTOR_PATH."
        )

    auto_dir          = args.output / "auto"
    manual_needed_dir = args.output / "manual_needed"
    manual_dir        = args.output / "manual"

    if args.auto:
        run_autocrop(args.input, auto_dir, manual_needed_dir)
    if args.manual:
        ManualCropTool(manual_needed_dir, manual_dir)


if __name__ == "__main__":
    main()
