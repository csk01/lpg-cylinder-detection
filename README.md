# LPG Cylinder Detection & Classification

Computer-vision pipeline that identifies the **brand** of an LPG gas cylinder presented at a
kiosk. A YOLOv11x detector locates the cylinder, then one or more EfficientNet classifiers
decide the brand from different views of it.

**Brands:** `bharat` · `hp` · `indane` (`unknown` is supported by the side-view model only)

> **Read this first.** This README is written for handoff and is deliberately explicit about
> what is *not* done. Anything marked *Not currently documented / needs confirmation* could not
> be verified from the code in this repository — do not assume it works.

---

## Overview

The kiosk problem is: a customer puts a cylinder down, and the system has to say which brand it
is. Brand cues live in different places on the cylinder — the painted body (side view), the
hole pattern in the foot ring (bottom view), and the valve guard (top view). The project
started as a single side-view classifier and has since grown into a **three-view ensemble**,
because the side view alone is unreliable when the paint is faded, dirty, or repainted.

Two things are true at once and are easy to confuse:

- `src/predict.py` — the **original single-model** path (detector → one 4-class classifier).
- `src/predict_ensemble.py` — the **current** path (detector → three classifiers → weighted vote).

Both work. The ensemble is what the Gradio app runs.

---

## Project Objective

Identify the brand of an LPG cylinder from kiosk camera images reliably enough to be used in an
automated exchange/vending flow, where a wrong brand call has a real cost. `unknown` is a
legitimate and desirable output when the model is not confident.

---

## Current Status

### Completed (verified in this repository)

| Item | Evidence |
|---|---|
| Cylinder detector (YOLOv11x) | `models/yolov11x_lpg_v1_best.pt` loads and detects |
| Side-view classifier (4-class, attention) | `models/classifier_best_v6_attention.pth`, val 95.15% |
| Bottom-ring classifier (3-class) | `models/classifier_bottomring_v2.pth`, val 87.69% |
| Top-view classifier (3-class) | `models/classifier_topview_v1.pth`, val 77.78% |
| Ensemble inference | `src/predict_ensemble.py` — runs end to end |
| Single-model inference | `src/predict.py` — runs end to end |
| Gradio demo (ensemble) | `src/app.py` |
| Streamlit demo (single model) | `src/streamlit_app.py` |
| Manual bottom-ring crop tool | `src/data_utils/crop_bottom_ring.py` |
| Semi-automatic crop tool (YOLO + manual fallback) | `src/data_utils/crop_tool.py` |
| Bottom-ring & top-view datasets | `datasets/*.zip` (counts below) |

### In progress

- **Bottom-ring dataset growth** — 776 crops today, produced by hand; class balance is uneven.
- **Top-view classifier** — trained but weakest of the three (77.78%); used only as an optional,
  low-weight ensemble branch.
- **Ensemble tuning** — the fusion weights are hand-set constants, never fitted or validated
  (see [Known Limitations](#known-limitations)).

### Not yet completed

- **No evaluation script.** There is no script in this repo that measures ensemble accuracy.
  Every number quoted here is a *per-model validation* figure taken from its own checkpoint.
- **No held-out test set.** Only train/valid splits exist. Model selection used the validation
  set, so these figures are optimistically biased.
- **No training code for the shipped bottom-ring model.** See [Training](#training).
- **No REST API.** `predict_ensemble()` is importable but not served.
- **No ONNX / CPU-optimised export.**
- **No automated tests, CI, or linting configuration.**

---

## System / Pipeline Overview

```
                    side camera image                  top camera image (optional)
                            │                                    │
                            ▼                                    ▼
                    YOLOv11x detector                    YOLOv11x detector
                     (conf = 0.45)                        (conf = 0.45)
                            │                                    │
              0 boxes → "no_cylinder"                   exactly 1 box required,
             >1 boxes → "multi_cylinder"                otherwise top view is skipped
                            │                                    │
                    exactly 1 box                                │
                            │                                    │
          ┌─────────────────┴─────────────────┐                  │
          ▼                                   ▼                  ▼
   side region crop                    bottom-ring crop      top crop
   (25%–85% of box height)           (75%–100% of height)   (whole box)
          │                                   │                  │
          ▼                                   ▼                  ▼
  EfficientNetB2+attention           EfficientNetB0        EfficientNetB0
      (4 classes)                      (3 classes)          (3 classes)
    weight 0.40                       weight 0.45          weight 0.15
          └─────────────────┬─────────────────┴──────────────────┘
                            ▼
              drop any view with conf < 60%
              renormalise remaining weights to 1.0
              weighted sum of (confidence × weight) per brand
                            │
                            ▼
              final conf < 25%  → "unknown"   (inside compose())
              final conf < 50%  → "unknown"   (CONFIDENCE_THRESHOLD)
                            │
                            ▼
                    {brand, confidence}
```

Note the **side and bottom crops are both geometric slices of the same side-camera bounding
box** — they overlap between 75% and 85% of box height. Only the top view needs a second camera.

---

## Dataset

The canonical dataset and model assets are maintained **externally in Google Drive**, not in
Git, because of image and checkpoint size:

**Dataset & assets:** https://drive.google.com/drive/folders/1WAswEjkLx9BuuNROlCt_6ksZgCjGjali?usp=drive_link

Snapshots are also committed as zips under `datasets/`, but these are **git-ignored**
(`datasets/*.zip` in `.gitignore`) — a fresh clone will not contain them. Get them from Drive.

### Dataset structure (verified counts from the zips)

`datasets/lpg_bottom_ring_dataset_v1.zip` — **776 images, flat, no split**

```
lpg_bottom_ring_dataset_v1/
├── bharat/   182
├── hp/       342
└── indane/   252
```

`datasets/lpg_top_view_dataset_v1.zip` — **304 images, flat, no split**

```
lpg_top_view_dataset_v1/
├── bharat/   118
├── hp/        82
└── indane/   104
```

`datasets/sorted_crops.zip` — **658 whole-cylinder crops (4-class, legacy side-view)**

```
sorted_crops/
├── bharat_gas/ 142
├── hp_gas/     155
├── indane/     260
└── unknown/    101
```

`datasets/lpg_dataset_clean.zip` — **795 raw/uncropped images**, mixed sub-foldering
(`raw/`, `renamed/`, `Jumbo/`, `Bulk Tank/`, `Composite/`, `nulls/`,
`other_brand_gas_cylinder/`). This is source material, not a training-ready set.

> ⚠️ **Reproducibility gap.** Both current dataset zips are **flat brand folders with no
> `train/` and `valid/` split**, but the training notebooks call
> `datasets.ImageFolder(TRAIN_DIR)` / `(VALID_DIR)` and expect
> `bottomring/train/{bharat,hp,indane}` + `bottomring/valid/{...}`. **The split step that
> produced the training zips is not in this repository.** Whoever retrains must create the
> split themselves. `src/data_utils/build_v3_dataset.py` does a seeded 70/30 split but is
> written for the 4-class side-view layout, so it needs adapting.

---

## Data Preparation

Distinguishing automated from manual:

| Step | Status |
|---|---|
| Collect raw images | **Manual** |
| Sort by brand | **Manual** (`review_tool.py`, keyboard-driven) |
| Identify bottom-view images | **Manual** (visual) |
| Crop the bottom ring | **Manual** (`crop_bottom_ring.py`, draw a rectangle) |
| Whole-cylinder crop | **Automated with manual fallback** (`crop_tool.py --auto --manual`) |
| Deduplicate | **Automated** (`dedupe_roboflow.py`, `count_dupes.py`, phash in `review_tool.py`) |
| Train/valid split | **Not in repo** for the current datasets — see gap above |
| Train | **Automated** (Colab notebooks) |
| Evaluate | **Partial** — per-model confusion matrix inside the notebooks only |
| Ensemble | **Automated** (`predict_ensemble.py`), but weights are hand-set |

Flow as actually practised:

```
Raw / collected images        [manual collection]
        ↓
Image sorting                 [manual, review_tool.py]
        ↓
Identify bottom-view images   [manual]
        ↓
Manual bottom-ring crop       [manual, crop_bottom_ring.py]
        ↓
Bottom-ring dataset           → datasets/lpg_bottom_ring_dataset_v1.zip
        ↓
Train/valid split             [NOT IN REPO — must be recreated]
        ↓
Train classifier              [Colab notebook]
        ↓
Evaluate classifier           [in-notebook only; no standalone script]
        ↓
Ensemble classifier           [predict_ensemble.py, hand-set weights]
        ↓
Final prediction
```

---

## Bottom-Ring Crop Pipeline

`src/data_utils/crop_bottom_ring.py` — a tkinter tool for hand-cropping the foot ring.

```bash
python src/data_utils/crop_bottom_ring.py --input <src_dir> --output <out_dir>
```

Defaults resolve relative to the repo root and can also be set via environment variables
`LPG_CROP_SOURCE` and `LPG_CROP_OUTPUT`. The default output is
`datasets/lpg_bottom_ring_dataset_v1/cropped`.

- **Input:** a flat folder of images. Formats: `.jpg`, `.jpeg`, `.png`, `.webp`.
- **Output:** `<output>/<brand>/<originalname>_crop<N>.<ext>` — brand subfolders are created
  automatically.
- **Controls:** drag to draw a rectangle, then
  `1` = indane · `2` = hp · `3` = bharat · `R` = undo last · `S` = skip · `D` = **delete source
  image from disk** · `←`/`→` = rotate 45° · rotation buttons for ±45°/±90°.
- Multiple crops can be taken from one image; each gets an incrementing `_cropN` suffix.
- Crops smaller than 20×20 px (in image coordinates) are rejected.

> `D` permanently deletes the source file. Work on a copy.

---

## Models

All three classifiers share the same preprocessing at inference time
(`src/predict_ensemble.py`): resize to **224×224**, `ToTensor`, ImageNet normalisation
(mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`). No test-time augmentation.

### Detector

| Property | Value |
|---|---|
| File | `models/yolov11x_lpg_v1_best.pt` (114 MB) |
| Architecture | YOLOv11x |
| Inference confidence | `0.45` (`predict_ensemble`), `0.55` (`crop_tool.py`) |
| mAP50 | ~0.969 — *from prior project notes, not reproducible from this repo* |

Training config for the detector is in `notebooks/lpg_detection_v1.ipynb`.

### Side-view classifier — current primary

| Property | Value (from checkpoint metadata) |
|---|---|
| File | `models/classifier_best_v6_attention.pth` |
| Architecture | `efficientnet_b2 + spatial_attention` |
| Backbone | EfficientNetB2 |
| Head | `Dropout(0.4) → Linear(1408, 256) → ReLU → Dropout(0.3) → Linear(256, 4)` |
| Attention | 7×7 conv over (avg-pool + max-pool) channel maps, sigmoid gate |
| Input | 224×224 |
| Classes | 4 — `bharat, hp, indane, unknown` |
| Best val acc | **95.15%** |
| Epochs trained | 36 |
| Train class counts | bharat 934 · hp 732 · indane 879 · unknown 93 |
| Checkpoint threshold | 0.60 |
| Loss / optimizer / LR / scheduler | *Not recorded in the checkpoint; the matching notebook is not in this repo — needs confirmation* |

### Bottom-ring classifier

| Property | Value (from checkpoint metadata) |
|---|---|
| File | `models/classifier_bottomring_v2.pth` |
| Architecture | `efficientnet_b0` |
| Head | `Dropout(0.5) → Linear(1280, 3)` |
| Input | 224×224 |
| Classes | 3 — `bharat, hp, indane` |
| Best val acc | **87.69%** |
| Epochs trained | 40 |
| Batch size | 32 |
| LR | 3e-4 |
| Weight decay | 5e-4 |
| Label smoothing | 0.15 |
| Rotation augmentation | 360° (ring orientation is unconstrained) |
| Early stopping | patience 15 |

### Top-view classifier

| Property | Value (from checkpoint metadata) |
|---|---|
| File | `models/classifier_topview_v1.pth` |
| Architecture | `efficientnet_b0_frozen_head` — backbone frozen, head only trained |
| Head | `Dropout(0.5) → Linear(1280, 3)` |
| Input | 224×224 |
| Classes | 3 — `bharat, hp, indane` |
| Best val acc | **77.78%** |
| Epochs trained | 22 |
| Batch size | 16 |
| LR | 1e-3 |
| Weight decay | 5e-4 |
| Label smoothing | 0.15 |
| Train split | 0.65 |

> ⚠️ **The top-view notebook does not match the shipped checkpoint.**
> `notebooks/nb_topview_classifier.ipynb` builds an **EfficientNetB2 + attention** model and
> saves `architecture: 'efficientnet_b2_attention'`. The actual file is an
> **EfficientNetB0 with a frozen backbone**. `predict_ensemble.py` loads it as B0, matching the
> checkpoint — so **inference is correct and the notebook is stale.** Do not "fix" the loader
> to match the notebook.

### Ensemble Classifier

Implemented in `src/predict_ensemble.py` — `load_ensemble_models()` and `predict_ensemble()`.

| Property | Value |
|---|---|
| Constituent models | side (B2+attention, 4-class), bottom-ring (B0, 3-class), top-view (B0, 3-class) |
| Views | side + bottom derived from **one** side-camera bbox; top from an optional second camera |
| Fusion method | **Confidence-weighted vote** over each view's top-1 prediction |
| Combines | Top-1 label + confidence per view — **not** full logits or probability vectors |
| Weights | side `0.40`, bottom `0.45`, top `0.15` — **hard-coded constants, not fitted** |
| Per-view gate | a view is dropped if its confidence < **60%** |
| Renormalisation | remaining weights rescaled to sum to 1.0 |
| Score | `score[brand] += (conf/100) × normalised_weight` |
| Decision | `argmax(score)`; `unknown` if final < 25%, then again if < `CONFIDENCE_THRESHOLD` (50%) |
| Min crop size | 30 px per side, else that view is skipped |
| Checkpoint/config location | no separate ensemble artifact — weights live in `compose()` |

**How it differs from the single-model classifier:** `predict.py` crops the *whole* bbox and
runs one 4-class model with a 0.70 threshold. The ensemble slices the bbox into
side and bottom regions, runs a specialised model on each, optionally adds a top view, and
votes. The ensemble can therefore still identify a cylinder whose paint is unreadable, via the
foot-ring hole pattern.

Because fusion happens on **top-1 label + confidence** rather than full probability vectors, a
view that is "second-guessing" contributes nothing — its runner-up mass is discarded. That is a
design limitation worth revisiting.

---

## Training

Training runs in Google Colab. The notebooks mount Drive, unzip a dataset, train, plot curves,
print a confusion matrix, and save a checkpoint back to Drive.

| Notebook | Trains | Matches shipped checkpoint? |
|---|---|---|
| `notebooks/nb_bottomring_classifier.ipynb` | Bottom-ring B0 | ❌ **No** — notebook is **v1**; shipped model is **v2** |
| `notebooks/nb_topview_classifier.ipynb` | Top view | ❌ **No** — notebook builds B2+attention; shipped model is frozen B0 |
| `notebooks/lpg_classification_model.ipynb` | Side-view classifier | *Needs confirmation* — not verified against v6 |
| `notebooks/lpg_detection_v1.ipynb` | YOLOv11 detector | *Needs confirmation* |

> ⚠️ **No notebook in this repo reproduces any currently shipped classifier.** The bottom-ring
> notebook is v1 (rotation 180°, dropout 0.3, weight decay 1e-4, 30 epochs, no label smoothing),
> while `classifier_bottomring_v2.pth` records rotation 360°, dropout 0.5, weight decay 5e-4,
> label smoothing 0.15, 40 epochs, early-stopping patience 15. **The v2 training code is
> missing.** Recovering it from Drive is the single highest-value handoff action.

Common recipe across both bottom-ring and top-view notebooks: `ImageFolder` +
`WeightedRandomSampler` for class imbalance, `AdamW`, `CosineAnnealingLR`,
`CrossEntropyLoss`, and **checkpoint selection = best validation accuracy** (`best_state`
deep-copied whenever `val_acc` improves, reloaded at the end).

---

## Evaluation

**There is no standalone evaluation script in this repository.** Evaluation exists only as
cells inside the training notebooks (`classification_report` + a seaborn confusion matrix over
the validation loader).

Consequences:
- The **ensemble has never been measured as a system.** Its accuracy is unknown. The weights
  0.40 / 0.45 / 0.15 are not backed by any measurement in this repo.
- Per-model accuracies come from different validation sets of different sizes and are **not
  comparable to each other**.
- No confusion matrices, per-class precision/recall, or calibration data are stored as
  artifacts.

---

## Current Results

### Latest classifier training run (30 epochs)

Best validation accuracy **83.59%**, reached at **epoch 9** and matched again at **epoch 25**.

| Metric | Value |
|---|---|
| Best val acc | **83.59%** (epochs 9 and 25) |
| Final train acc (epoch 30) | 96.73% |
| Final val acc (epoch 30) | 81.03% |
| Final train loss | 0.0992 |
| Final val loss | 0.5584 |
| Train/val accuracy gap (final) | **~15.7 points** |

Reading of the curve:

- Validation accuracy climbs quickly to ~83% by epoch 9, then **plateaus for the remaining 21
  epochs** — it oscillates in an 78–84% band and never improves on epoch 9.
- Validation loss bottoms out around **0.52 at epoch 9** and then drifts *upward* and flattens
  near 0.55–0.58, while training loss keeps falling from 0.28 to 0.099.
- That divergence — training loss still dropping while validation loss stops improving — is
  **textbook overfitting**. The last ~20 epochs bought nothing.

> **96.73% is training accuracy and must not be quoted as this model's expected performance.**
> The realistic expectation from this run is **~83%**, and even that is a validation figure used
> for model selection, so true held-out performance is likely lower.

> ⚠️ **This run does not correspond to any checkpoint in this repository.** No file in `models/`
> records 30 epochs or an 83.59% best val acc (`bottomring_v2` = 40 epochs / 87.69%;
> `topview_v1` = 22 epochs / 77.78%; `v6_attention` = 36 epochs / 95.15%). **Which model this
> log belongs to needs confirmation**, and its checkpoint does not appear to be committed.

### Per-model validation accuracy (from checkpoint metadata)

| Model | Classes | Val acc | Epochs |
|---|---|---|---|
| `classifier_best_v6_attention.pth` (side) | 4 | 95.15% | 36 |
| `classifier_bottomring_v2.pth` (bottom) | 3 | 87.69% | 40 |
| `classifier_topview_v1.pth` (top) | 3 | 77.78% | 22 |
| `classifier_best_v4_3.pth` (archive) | 4 | 97.62% | 34 |

These are **not** comparable — different datasets, sizes, and class counts. The archived
v4_3's 97.62% is on an easier/smaller validation set and is not evidence it beats v6.

### Ensemble performance

**Not measured.** No script, no metric, no test set.

---

## Known Limitations

### Dataset

1. **Small.** 776 bottom-ring and 304 top-view images. The top-view set is ~100 images per
   class — very thin for a 3-class problem, which is why its backbone had to be frozen.
2. **Class imbalance.** Bottom ring is hp 342 / indane 252 / bharat 182 — hp has ~1.9× bharat.
   Mitigated at train time by `WeightedRandomSampler`, not fixed at the data level.
3. **Severe `unknown` scarcity.** The side model saw only **93** `unknown` images against
   934 bharat — a ~10:1 imbalance on the very class used as the safety valve.
4. **Manual crop quality varies.** Bottom-ring crops are hand-drawn rectangles, so framing,
   tightness, and centring are inconsistent between sessions.
5. **Duplicates / near-duplicates.** Tooling exists (`count_dupes.py`, phash in
   `review_tool.py`) but there is **no record that it was run** on the current datasets, and no
   dedup report is stored.
6. **Source bias.** Part of the legacy data came from Roboflow exports and YouTube frames
   (`new_yt_crops` in `build_v3_dataset.py`); frames from one video are highly correlated.
7. **Lighting / camera / background variation is undocumented.** No metadata about capture
   conditions is recorded, so robustness to a new kiosk's lighting is unknown.
8. **Geographic bias.** Indian-market cylinders only; the brand set is India-specific.
9. **View distribution is uneven** — 776 bottom vs 304 top vs ~2,600 legacy side crops.

### Model

1. **Overfitting** in the latest run — ~15.7-point train/val gap, validation loss rising after
   epoch 9 while training loss keeps falling.
2. **~83% realistic accuracy** for that run, far below the 95%+ headline figures attached to
   other checkpoints.
3. **Validation accuracy is optimistically biased** — checkpoint selection is
   "best val acc", so the reported number is a maximum over 30–40 noisy evaluations of the same
   set. There is no held-out test set anywhere in the project.
4. **Top-view model is weak** (77.78%) and its backbone is frozen, capping it.
5. **Viewpoint dependence.** Side and bottom crops are fixed geometric slices (25–85% and
   75–100% of bbox height). A tilted, occluded, or unusually proportioned cylinder puts the
   wrong pixels into each model.
6. **Crop-quality sensitivity.** A loose or shifted detector box propagates directly into both
   slices; there is no re-detection or refinement.
7. **Confusion between visually similar brands is not quantified** — no confusion matrices are
   stored, so *which* brands get confused is unknown.
8. **No calibration.** Softmax confidences drive three separate thresholds (60% per-view, 25%
   composite, 60% final) but were never checked against actual correctness.

### Pipeline

1. **Manual bottom-ring cropping** is the throughput bottleneck and the main barrier to scaling
   the dataset.
2. **The shipped bottom-ring model cannot be retrained from this repo** — v2's training code is
   missing; only the v1 notebook is present.
3. **The top-view notebook contradicts its checkpoint** (B2+attention vs frozen B0).
4. **The train/valid split step is missing** for both current datasets, which ship flat.
5. **No dataset versioning.** Zips are named `_v1` with no manifest, checksum, or changelog;
   they are git-ignored, so history is not recoverable from Git.
6. **No model versioning discipline.** Version numbers skip (`v4_3` → `v6`; no `v5` present)
   and `models/archive/` is undocumented.
7. **No automated evaluation, tests, CI, or linting.**
8. **Ensemble weights are unjustified constants.**
9. **`predict.py` and `predict_ensemble.py` duplicate** the attention architecture, transforms,
   and crop helpers — they will drift apart.
10. **Both write a temp file (`temp.jpg` / `temp_top.jpg`) into the current working directory**
    on every prediction — not concurrency-safe and pollutes the repo root.

---

## Future Scope

### High priority

1. **Recover the v2 bottom-ring training code from Drive** and commit it. Until then the best
   bottom-ring model is unreproducible.
2. **Build a held-out test set** that is never used for model selection, and stop quoting
   selection-time validation accuracy as expected performance.
3. **Write an evaluation script** (`src/evaluate.py`) producing per-class precision/recall and a
   confusion matrix for each model **and for the ensemble as a whole**.
4. **Measure the ensemble**, then fit the fusion weights on validation data instead of using
   0.40 / 0.45 / 0.15 by hand.
5. **Address the overfitting** in the 83.59% run — early stopping at ~epoch 9–12, stronger
   augmentation/regularisation, or more data. ~20 of 30 epochs are currently wasted.
6. **Expand and rebalance the dataset**, especially `unknown` (93 images) and top view (304).
7. **Commit the train/valid split script** so the datasets are reproducible from the flat zips.

### Medium priority

8. **Automate bottom-ring ROI extraction** — train a detector/segmenter for the foot ring to
   remove the manual crop bottleneck.
9. **Fuse probability vectors instead of top-1 labels**, so a view's second choice still counts.
10. **Calibrate confidences** (temperature scaling) and re-derive the three thresholds from data.
11. **Dataset versioning** — manifests, checksums, per-version counts, and a changelog.
12. **Automated dataset quality checks** — dedup, min-size, corrupt-file, and class-balance
    reports run on every dataset build.
13. **Deduplicate `predict.py` / `predict_ensemble.py`** into a shared module.
14. **Error-analysis dashboard** over `flagged_predictions/`.

### Long term

15. **End-to-end pipeline**: cylinder detection → view classification → brand classification,
    so the system picks the right model rather than assuming a fixed camera geometry.
16. **Segmentation-based ROI extraction** to replace fixed percentage slices.
17. **Edge/CPU-optimised inference** — ONNX/TensorRT export; the 114 MB YOLOv11x is heavy for a
    kiosk.
18. **Production REST API** (the schema sketched in `CLAUDE.md`).
19. **Model monitoring** in deployment — drift, confidence distributions, `unknown` rate.
20. **Active-learning loop** feeding flagged kiosk predictions back into training.

---

## Repository Structure

```
lpg-cylinder-detection/
├── README.md                              ← this file
├── CLAUDE.md                              ← agent/dev notes (partly outdated, see Handoff Notes)
├── requirements.txt
│
├── models/                                ← git-ignored; get from Drive
│   ├── yolov11x_lpg_v1_best.pt            ← detector (114 MB)
│   ├── classifier_best_v6_attention.pth   ← side view, 4-class, 95.15%
│   ├── classifier_bottomring_v2.pth       ← bottom ring, 3-class, 87.69%
│   ├── classifier_topview_v1.pth          ← top view, 3-class, 77.78%
│   └── archive/                           ← superseded checkpoints, kept for reference
│       ├── best.pt                        ← original YOLOv11n detector
│       ├── classifier_best.pth            ← B0 v1 (overfit)
│       ├── classifier_best_v2.pth
│       ├── classifier_best_v3.pth
│       └── classifier_best_v4_3.pth       ← 4-class B2, 97.62% (not comparable — see Results)
│
├── src/
│   ├── predict_ensemble.py                ← CURRENT inference path (3-view ensemble)
│   ├── predict.py                         ← legacy single-model inference path
│   ├── app.py                             ← Gradio demo (uses the ensemble)
│   ├── streamlit_app.py                   ← Streamlit demo (uses predict.py)
│   └── data_utils/
│       ├── crop_bottom_ring.py            ← manual bottom-ring crop tool (current)
│       ├── crop_tool.py                   ← YOLO auto-crop + manual fallback (current)
│       ├── build_v3_dataset.py            ← LEGACY 70/30 split builder (4-class layout)
│       ├── review_tool.py                 ← LEGACY brand sorter + phash dedup
│       ├── clean_names.py                 ← LEGACY Roboflow filename cleaner
│       ├── dedupe_roboflow.py             ← LEGACY Roboflow dedup
│       ├── rename.py                      ← LEGACY batch renamer (dry-run by default)
│       └── count_dupes.py                 ← duplicate counter (read-only)
│
├── notebooks/                             ← Colab training notebooks (see Training caveats)
│   ├── nb_bottomring_classifier.ipynb     ← v1 — does NOT match shipped v2
│   ├── nb_topview_classifier.ipynb        ← builds B2+attn — does NOT match shipped B0
│   ├── lpg_classification_model.ipynb
│   └── lpg_detection_v1.ipynb
│
├── datasets/                              ← *.zip are git-ignored; get from Drive
│   ├── lpg_bottom_ring_dataset_v1.zip     ← 776 imgs, flat
│   ├── lpg_top_view_dataset_v1.zip        ← 304 imgs, flat
│   ├── sorted_crops.zip                   ← 658 legacy 4-class crops
│   ├── lpg_dataset_clean.zip              ← 795 raw images
│   └── dataset_info.md                    ← currently EMPTY
│
└── flagged_predictions/                   ← git-ignored; wrong predictions from the demos
```

---

## Reproducibility

Honest summary: **you can reproduce inference; you cannot currently reproduce training.**

### 1. Clone

```bash
git clone <repo-url>
cd lpg-cylinder-detection
```

### 2. Install dependencies

```bash
conda create -n lpg_env python=3.10
conda activate lpg_env
pip install -r requirements.txt
```

`requirements.txt` pins `torch==2.7.1+cu118` / `torchvision==0.22.1+cu118`, which need the CUDA
118 index. For CPU-only, install torch separately first. `imagehash` is used by
`review_tool.py` but is **not** in `requirements.txt` — `pip install imagehash` if needed.

### 3. Get models and data

Both `models/` and `datasets/*.zip` are git-ignored — a fresh clone has **no weights and no
data**. Download from the [Drive folder](https://drive.google.com/drive/folders/1WAswEjkLx9BuuNROlCt_6ksZgCjGjali?usp=drive_link)
and place them so that `models/yolov11x_lpg_v1_best.pt` and the three `.pth` files exist.

### 4. Run inference ✅ *works*

```bash
python src/app.py            # Gradio, ensemble — http://localhost:7860
python -m streamlit run src/streamlit_app.py   # Streamlit, single model
```

`share=True` is blocked by the local Windows firewall; use `ngrok http 7860` in a second
terminal for a public link.

### 5. Prepare a dataset ⚠️ *partially works*

Unzip a dataset into `datasets/`. Crop tools:

```bash
python src/data_utils/crop_tool.py --auto --manual --input <src> --output <out>
python src/data_utils/crop_bottom_ring.py --input <src> --output <out>
```

### 6. Split into train/valid ❌ **breaks here**

The zips are flat brand folders; the notebooks need `train/` and `valid/`. **No script in this
repo performs this split for the 3-class datasets.** You must write one (or adapt
`build_v3_dataset.py`, which assumes the 4-class layout).

### 7. Train ⚠️ *works, but not for the shipped models*

Upload a notebook to Colab, mount Drive, upload the split zip, run. This reproduces
**a** model — but **not** the currently shipped bottom-ring v2 or top-view v1, whose recipes
differ from the committed notebooks (see [Training](#training)).

### 8. Evaluate ❌ **not available**

No standalone evaluation script; only in-notebook confusion matrices. Ensemble evaluation does
not exist.

### 9. Verify your setup

```bash
python -m compileall -q src/          # all sources compile
python -c "import sys; sys.path.insert(0,'src'); \
from predict_ensemble import load_ensemble_models; load_ensemble_models()"
```

The second command prints each model's best validation accuracy on success.

---

## Inference

```python
import sys; sys.path.insert(0, "src")
from PIL import Image
from predict_ensemble import load_ensemble_models, predict_ensemble

detector, side_clf, bottom_clf, top_clf, device = load_ensemble_models()

result = predict_ensemble(
    Image.open("cylinder.jpg"),
    detector, side_clf, bottom_clf, top_clf, device,
    top_image=None,          # optional PIL image from the top camera
)
print(result["brand"], result["confidence"])
```

`status` is one of `ok`, `no_cylinder`, or `multi_cylinder`. On `ok` the dict also carries
`full_crop`, `side_crop`, `bottom_crop`, `top_crop`, the per-view predictions
(`side_pred`/`side_conf`, `bottom_pred`/`bottom_conf`, `top_pred`/`top_conf`), and
`probabilities` (the **side** model's full distribution, including `unknown`).

Verified working example output on the bundled `temp.jpg`:

```
brand=indane  confidence=74.4
side=indane (74.7%)   bottom=indane (74.2%)   top=None
probabilities={'bharat': 3.9, 'hp': 15.8, 'indane': 74.7, 'unknown': 5.6}
```

---

## Model Artifacts

Weights are **not committed** (`.gitignore`) — share via the Drive folder. Keep it that way;
the detector alone is 114 MB.

| File | Role | Val acc | Keep? |
|---|---|---|---|
| `yolov11x_lpg_v1_best.pt` | Detector — current | mAP50 ~0.969 (unverified here) | ✅ current |
| `classifier_best_v6_attention.pth` | Side view — current | 95.15% | ✅ current |
| `classifier_bottomring_v2.pth` | Bottom ring — current | 87.69% | ✅ current |
| `classifier_topview_v1.pth` | Top view — current | 77.78% | ✅ current |
| `archive/classifier_best_v4_3.pth` | Previous single-model best | 97.62% | reference |
| `archive/classifier_best_v3.pth` | Older | — | reference |
| `archive/classifier_best_v2.pth` | Older | — | reference |
| `archive/classifier_best.pth` | B0 v1, overfit | — | reference |
| `archive/best.pt` | Original YOLOv11n detector | — | reference |

Checkpoint format — every classifier `.pth` is a dict, not a bare state dict:

```python
{
  "model_state_dict":     ...,
  "best_val_acc":         95.15,
  "classes":              ["bharat", "hp", "indane", "unknown"],
  "architecture":         "efficientnet_b2 + spatial_attention",
  "confidence_threshold": 0.60,
  # plus per-model extras: epochs_trained, batch_size, lr, weight_decay,
  # label_smoothing, img_size, class_counts, train_history, notes
}
```

Always read `classes` from the checkpoint rather than assuming a global order — the models do
**not** agree on class count (see Configuration).

---

## Configuration

### Inference constants

| Constant | File | Value | Meaning |
|---|---|---|---|
| `CONFIDENCE_THRESHOLD` | `predict_ensemble.py` | `0.50`* | Final composed confidence below this → `unknown` |
| `MIN_CROP_SIZE` | `predict_ensemble.py` | `30` px | Skip a view whose crop is smaller |
| per-view gate | `compose()` | `60.0` | Drop a view's vote below this confidence |
| composite floor | `compose()` | `25.0` | Composed score below this → `unknown` |
| weights | `compose()` | 0.40 / 0.45 / 0.15 | side / bottom / top |
| `side_conf_thresh` | `predict_ensemble()` | `0.45` | YOLO detection confidence |
| `CONFIDENCE_THRESHOLD` | `predict.py` | `0.70` | Legacy single-model threshold |

\* Note `predict_ensemble.CONFIDENCE_THRESHOLD` is `0.50` in code while the checkpoints record
`confidence_threshold: 0.60`. The code value wins at runtime. *Which is intended needs
confirmation.*

### ⚠️ Class lists differ per model

The side classifier is **4-class** (`bharat, hp, indane, unknown`); the bottom-ring and
top-view classifiers are **3-class**. `load_ensemble_models()` reads each checkpoint's
`classes` key and attaches it to the model as `.classes`, and `_classify_crop()` uses that.
**Never index one model's logits with another's class list** — the module-level `CLASSES`
constant is a 3-item fallback only.

### Environment variables

| Variable | Used by | Purpose |
|---|---|---|
| `LPG_CROP_SOURCE` | `crop_bottom_ring.py` | Default input folder |
| `LPG_CROP_OUTPUT` | `crop_bottom_ring.py` | Default output folder |
| `LPG_DETECTOR_PATH` | `crop_tool.py` | Detector weights location |

All model paths in `src/` resolve relative to the repo root, with a fallback for Hugging Face
Spaces (`/app/models/`). No machine-specific paths remain in `src/`.

---

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `FileNotFoundError` on a `.pt`/`.pth` | Weights are git-ignored — download from Drive into `models/` |
| Gradio public link fails | Windows firewall blocks `share=True`; run `ngrok http 7860` instead |
| `ModuleNotFoundError: torch` | Wrong environment — `conda activate lpg_env` |
| `ModuleNotFoundError: imagehash` | `review_tool.py` only — `pip install imagehash` |
| tkinter tool won't open | Run locally; tkinter does not work in Colab/headless |
| Notebook asserts on class order | `ImageFolder` sorts alphabetically — folders must be exactly `bharat`, `hp`, `indane` |
| Training notebook can't find `train/`/`valid/` | Dataset zips are flat — you must split them first (known gap) |
| Slow first inference | YOLOv11x is 114 MB; load is slow, inference is <200 ms |
| SSL cert error on `conda activate` | Cosmetic, does not affect functionality |

---

## Handoff Notes

**What is working.** Both inference paths run end to end and are verified: `src/app.py`
(Gradio, ensemble) and `src/streamlit_app.py` (Streamlit, single model). All four checkpoints
load. All of `src/` compiles. Every data utility has a working `--help` and no longer contains
machine-specific paths.

**Do NOT accidentally change these.**

- **The top-view loader in `predict_ensemble.py`.** It builds an EfficientNet**B0** while
  `nb_topview_classifier.ipynb` describes B2+attention. **The loader is right, the notebook is
  stale.** "Fixing" the loader will break inference.
- **The per-model `.classes` mechanism.** The side model has 4 classes and the others have 3.
  Collapsing them back to one global `CLASSES` list reintroduces a crash (fixed — see below).
- **`CLASSES` order.** It must match each checkpoint's `classes` key; `ImageFolder` assigns
  labels alphabetically.
- **`models/archive/`.** Superseded but useful for comparison — don't delete.
- **The ensemble weights**, until you have measured the ensemble. Changing them now is guessing.

**Where the latest data lives.** Google Drive:
https://drive.google.com/drive/folders/1WAswEjkLx9BuuNROlCt_6ksZgCjGjali?usp=drive_link —
plus git-ignored local zips in `datasets/`. Current focus is
`lpg_bottom_ring_dataset_v1` (776 manually cropped foot-ring images).

**Where the models live.** `models/` (git-ignored), mirrored on Drive.

**Current best model.** For a full prediction, the **three-view ensemble** in
`src/predict_ensemble.py` is what ships. Its strongest single component is the side-view
`classifier_best_v6_attention.pth` (95.15% val). Note the archived `classifier_best_v4_3.pth`
shows a *higher* number (97.62%) on a different, smaller validation set — that is **not**
evidence it is better.

**What still needs validation.**

- The ensemble has **never been evaluated as a system.** Its accuracy is genuinely unknown.
- No held-out test set exists; every quoted number is selection-time validation accuracy.
- The 0.40 / 0.45 / 0.15 weights and the 60 / 25 / 60 thresholds are unmeasured.

**Biggest technical risk.** **The shipped bottom-ring model cannot be retrained from this
repository.** `classifier_bottomring_v2.pth` records hyperparameters (360° rotation,
dropout 0.5, weight decay 5e-4, label smoothing 0.15, 40 epochs, patience 15) that the
committed v1 notebook does not contain. If that checkpoint is lost or needs to change, the
recipe is gone. Close behind: no test set means the project cannot currently tell whether a
change helps or hurts.

**Recommended next experiment.** Build the held-out test set and an `src/evaluate.py`, then
measure the ensemble against each individual model. That single result tells you whether the
three-view design is earning its complexity — and gives you the harness needed to fit the
fusion weights instead of guessing them.

### Changes made in this handoff pass

1. **Fixed a crash in `src/predict_ensemble.py`.** The module-level
   `CLASSES = ["bharat", "hp", "indane"]` (3 items) was used to index **all** classifiers, but
   the side model outputs **4** logits. A side-view `unknown` prediction raised
   `IndexError: list index out of range`, taking down the Gradio app; the `unknown` probability
   was also silently dropped from the returned distribution. Each model now carries its own
   class list from its checkpoint. Verified: predictions are unchanged, `unknown` now appears in
   `probabilities`, and the former crash path returns cleanly.
2. **Fixed silent data loss in `crop_bottom_ring.py`.** The crop counter was derived from
   `len(self.rects)`, which `rotate()` clears and `redraw_last()` pops — so numbering restarted
   and **overwrote crops already saved from the same image**. Now uses a per-image counter that
   also skips existing filenames.
3. **Removed all machine-specific paths from `src/`** (was `C:\Users\Krishnan CS\OneDrive\...`
   in 8 files). Utilities now take `--input`/`--output`-style arguments with repo-relative
   defaults and env-var overrides.
4. **Made `rename.py` dry-run by default** (`--apply` to commit) and stopped it overwriting
   existing files.
5. **Added a missing `import hashlib`** in `review_tool.py` (`file_hash()` would have raised
   `NameError`) and an `imagehash` availability check.
6. **Marked legacy utilities as LEGACY** in their docstrings rather than deleting them.
7. **Rewrote this README** from the code as source of truth.

Model weights, dataset contents, labels, and all model behaviour were left untouched.

---

## Dataset / Asset Links

- **Dataset & model assets (Google Drive):**
  https://drive.google.com/drive/folders/1WAswEjkLx9BuuNROlCt_6ksZgCjGjali?usp=drive_link

The dataset is maintained externally in Google Drive and is **not committed to Git** because of
image and checkpoint size (`datasets/*.zip` and `models/` are both git-ignored). Request access
from the project owner; no credentials are stored in this repository.

Colab training artifacts are organised in Drive as:

```
LPG Cylinder Detection and Classification/
├── Dataset/
├── Detection/yolov11x_v1/weights/best.pt
└── Classifier/
    ├── bottomring_v1/
    └── topview_v1/
```
