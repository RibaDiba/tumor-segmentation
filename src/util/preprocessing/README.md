# Preprocessing Module

Converts raw scanner triplets (RGB annotation image, texture image, `.bin` point cloud) into Detectron2-registered COCO datasets across three modalities: **RGB**, **Depth**, and **RGD** (RGB with depth fused into the blue channel).

---

## Directory Structure

```
preprocessing/
├── TumorDataset/               # Main Dataset class, split into mixins
│   ├── tumor_dataset.py        # Dataset entry point — inherits all mixins
│   ├── _preprocessing.py       # PreprocessingMixin — full image pipeline
│   ├── _splitting.py           # SplittingMixin — train/val/test split
│   ├── _caching.py             # CachingMixin — save/load preprocessed data
│   ├── _coco.py                # CocoMixin — COCO JSON + Detectron2 registration
│   ├── _subset.py              # SubsetMixin — filter dataset by index
│   └── _utils.py               # UtilsMixin — sanity checks
├── PreprocessingFunctions/     # Unbound image processing functions
│   ├── _io.py                  # File I/O: read raw images, .bin files, folders
│   ├── _transforms.py          # Crop, pad, zoom, translate
│   ├── _masks.py               # Binary mask creation and hole-filling
│   ├── _depth.py               # Point cloud → depth image, RGD channel fusion
│   └── _splitting.py           # Standalone split utility (not used by Dataset)
├── augmentations.py            # AugmentationClass — flip and rotation augmentations
└── process_coco_json.py        # Binary mask → COCO JSON format
```

---

## Full Usage Workflow

### Step 1 — Instantiate

```python
from preprocessing.TumorDataset.tumor_dataset import Dataset

d = Dataset(data_path="/path/to/raw/data")
```

`data_path` should point to a directory containing matched triplets:
- `*.jpg` — red-highlight annotation image
- `*_texture.jpg` — raw texture image
- `*.bin` — point cloud file

### Step 2 — Preprocess

```python
d.preprocess_images(add_negative=False, read_bins=True)
```

Runs the full pipeline for all three modalities. After this call:

| Attribute | Description |
|---|---|
| `d.images_rgb` | List of 256×256 RGB arrays |
| `d.masks` | List of binary mask arrays (shared across modalities) |
| `d.images_depth_maps` | List of grayscale depth contour arrays |
| `d.images_rgd` | List of RGD arrays (blue channel = depth) |
| `d.depth_info` | Raw point cloud data (x, y, z grids) |
| `d.filenames` | Original filenames for tracking |
| `d.og_masks` / `d.og_images` | Unmodified originals, kept for reference |

Set `add_negative=True` to include tumor-free images from `data/raw_data/no_tumor/`.
Set `read_bins=False` to skip depth and RGD processing.

**RGB pipeline steps:**
1. Circle crop raw images (center 320,240, radius 180)
2. Circle crop masks (center 288,307, radius 200)
3. Add padding (auto-detected from image width: 492px MC Data, 577px Invotive)
4. Zoom masks 1.333×
5. Chroma-key binary mask extraction (red HSV range)
6. Crop all to 256×256
7. Apply −25px x-offset crop to masks
8. Fill mask holes via contour detection

**Depth pipeline steps:**
1. Convert `.bin` point cloud to contour plot (matplotlib `Grays`, 256×256)
2. Crop and pad depth images to match RGB pipeline

**RGD pipeline steps:**
1. Generate depth contour images (same as above)
2. Normalize depth to 0–255 and replace the blue channel of the texture image

### Step 3 — Augment (optional)

```python
from preprocessing.augmentations import AugmentationClass

aug = AugmentationClass(
    tumor_dataset=d,
    flip_prob=0.5,
    rotate_prob=0.5,
    rotate_degrees=15.0,
    test_only=False,   # set True to run on first image only
)
aug.augment_images()
```

Augmentation must happen **after** `preprocess_images()` and **before** `split_train_val_test()`.

- The same random flip and rotation are applied across all modalities at each index, preserving spatial correspondence between RGB, depth, and RGD images.
- Augmented pairs are **appended** to the existing lists — originals are not modified.
- Runs two passes, approximately doubling the dataset size.
- `BORDER_REFLECT` is used for rotation to avoid black border artifacts on binary masks.

> **Note:** Augmentation was found to degrade model performance in experiments. It is kept for reference but is not used in the default training pipeline. See the paper for details.

### Step 4 — Split

```python
d.split_train_val_test(70, 10, 20)   # percentages, must sum to 100
```

Creates separate lists for each split and modality:

| Attribute | Description |
|---|---|
| `d.train_images_rgb` / `d.val_images_rgb` / `d.test_images_rgb` | RGB splits |
| `d.train_images_depth` / `d.val_images_depth` / `d.test_images_depth` | Depth splits |
| `d.train_images_rgd` / `d.val_images_rgd` / `d.test_images_rgd` | RGD splits |
| `d.train_masks` / `d.val_masks` / `d.test_masks` | Mask splits |
| `d.train_filenames` / `d.val_filenames` / `d.test_filenames` | Filename splits |

### Step 5 — Cache to Disk

```python
d.cashe_data()
```

Saves preprocessed splits to `data/processed_data/`:

```
data/processed_data/
├── rgb/
│   ├── train/images/    ← .jpg files
│   ├── val/images/
│   └── test/images/
├── depth/
│   └── ...
└── rgd/
    └── ...
```

Masks are shared across modalities and saved to each modality's `masks/Tumor/` directory as `.png` files.

To reload cached data without re-running the full pipeline:

```python
d.load_data(rgb=True, depth=True, rgd=True)
```

### Step 6 — Convert to COCO JSON

```python
d.convert_binary_to_coco()
```

Reads the cached binary masks and writes `train.json`, `val.json`, and `test.json` for each modality into their respective `images/` directories. Each JSON follows the COCO instance segmentation format with category `"Tumor"` (id=1).

### Step 7 — Register with Detectron2

```python
d.register_instances(rgb=True)       # or depth=True, rgd=True
```

Registers the COCO datasets under the names:
- `my_dataset_train`
- `my_dataset_val`
- `my_dataset_test`

Only one modality should be registered per training run.

---

## Mixin Responsibilities

| Mixin | File | Responsibility |
|---|---|---|
| `PreprocessingMixin` | `_preprocessing.py` | Full image processing pipeline |
| `SplittingMixin` | `_splitting.py` | Percentage-based train/val/test split |
| `CachingMixin` | `_caching.py` | Save to and load from disk |
| `CocoMixin` | `_coco.py` | COCO JSON generation and Detectron2 registration |
| `SubsetMixin` | `_subset.py` | Filter dataset by index (remove specific images) |
| `UtilsMixin` | `_utils.py` | `check()` — plots first image/mask pair for sanity |

All mixins are combined in `TumorDataset/tumor_dataset.py` via multiple inheritance.

---

## PreprocessingFunctions Modules

These are **unbound functions** — they take `self` as their first argument and are assigned as class methods on `Dataset` in `tumor_dataset.py`. They can also be called as standalone utilities.

| Module | Contents |
|---|---|
| `_io.py` | `read_images_to_array`, `read_neg_images`, `read_bin`, `read_folder_to_array`, `read_to_array_post` |
| `_transforms.py` | `crop_raw_images`, `crop_masks`, `add_padding`, `zoom_at`, `crop_images`, `crop_images_offset`, `translate_images` |
| `_masks.py` | `create_binary_masks` (red HSV chroma key), `correct_binary_masks` (hole fill), `create_neg_masks` |
| `_depth.py` | `read_contours_array_depth` (point cloud → contour plot), `infuse_depth_into_blue_channel` |
| `_splitting.py` | Standalone `split_train_val_test()` (not used by `Dataset`) |

---

## AugmentationClass

```python
AugmentationClass(
    tumor_dataset,      # Dataset instance passed by reference
    flip_prob,          # probability of horizontal flip per image
    rotate_prob,        # probability of rotation per image
    rotate_degrees,     # max rotation angle (uniform in [-degrees, degrees])
    test_only=False,    # if True, only processes the first image
)
```

**Methods:**

- `augment_images()` — main entry point; applies flip and rotation, appends results in-place
- `_flip(images, mask, prob)` — horizontally flips all modality images and mask together
- `__rotate(images, mask, prob, degrees)` — rotates using a shared rotation matrix

The class holds a reference to the `Dataset` instance and reads/writes its image lists directly. No copies of the dataset are made.

---

## Known Issues

- **`read_neg_images()`** (`_io.py`): `os.listdir` is called without its argument, causing a runtime error when negative images are used.
- **`correct_binary_masks()`** (`_masks.py`): uses `cv2.COLOR_BAYER_BG2GRAY` instead of `cv2.COLOR_BGR2GRAY`, producing incorrect grayscale conversion.
- **`subet_automation()`** (`_subset.py`): function name is a typo ("subet" instead of "subset").
- **`create_neg_masks()`** (`_masks.py`): hardcodes mask dimensions to 495×492; not used in the current pipeline.
