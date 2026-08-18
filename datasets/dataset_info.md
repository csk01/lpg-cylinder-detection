# Dataset Info

Where the data lives and what is in it. See the main [README](../README.md) for how the data is
produced and used.

## Canonical location

All datasets and model weights are maintained **externally in Google Drive**, not in Git:

https://drive.google.com/drive/folders/1WAswEjkLx9BuuNROlCt_6ksZgCjGjali?usp=drive_link

`datasets/*.zip` is git-ignored, so a fresh clone contains **no image data**. Download from
Drive and place the zips in this folder. Request access from the project owner — no credentials
are stored in this repository.

## Contents (verified counts)

### `lpg_bottom_ring_dataset_v1.zip` — current focus

776 manually cropped foot-ring images, **flat, no train/valid split**.

```
lpg_bottom_ring_dataset_v1/
├── bharat/   182
├── hp/       342
└── indane/   252
```

Produced by hand with `src/data_utils/crop_bottom_ring.py`. Trains
`models/classifier_bottomring_v2.pth` (3-class, 87.69% val).

### `lpg_top_view_dataset_v1.zip`

304 top-view images, **flat, no train/valid split**.

```
lpg_top_view_dataset_v1/
├── bharat/   118
├── hp/        82
└── indane/   104
```

Trains `models/classifier_topview_v1.pth` (3-class, 77.78% val). Small enough that the
backbone was frozen to avoid overfitting.

### `sorted_crops.zip` — legacy side-view

658 whole-cylinder crops, 4-class (note the older folder naming).

```
sorted_crops/
├── bharat_gas/ 142
├── hp_gas/     155
├── indane/     260
└── unknown/    101
```

### `lpg_dataset_clean.zip` — raw source material

795 raw/uncropped images with mixed sub-foldering (`raw/`, `renamed/`, `Jumbo/`, `Bulk Tank/`,
`Composite/`, `nulls/`, `other_brand_gas_cylinder/`). Not training-ready — this is input to the
crop/sort tools.

## Known gaps

- **No train/valid split.** Both current zips ship flat, but the training notebooks call
  `datasets.ImageFolder` on `train/` and `valid/` subfolders. The script that produced the
  split used for the shipped models is **not in this repository** and must be recreated before
  retraining. `build_v3_dataset.py` does a seeded 70/30 split but assumes the 4-class layout.
- **No manifests or checksums.** Zips are versioned only by filename; there is no changelog
  recording what changed between versions.
- **Class imbalance** — bottom ring is hp 342 / indane 252 / bharat 182. Handled at train time
  with `WeightedRandomSampler`, not fixed in the data.
- **Deduplication status unrecorded.** Tooling exists (`count_dupes.py`, phash in
  `review_tool.py`) but there is no report showing it was run on these datasets.
- **Folder naming is inconsistent** across generations: `bharat_gas`/`hp_gas` in the legacy sets
  vs `bharat`/`hp` in the current ones. `ImageFolder` labels alphabetically, so folder names
  must exactly match the checkpoint's `classes` list.

## Local paths

Historic local source folders (author's machine, not required to use this repo) were under
`OneDrive/Documents/Downloads/`. All scripts now take explicit `--input`/`--output` arguments
with repo-relative defaults, so no personal paths are needed.
