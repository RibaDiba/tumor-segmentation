# FailureRecreation — Context and Instructions

## Purpose

This directory generates synthetic augmented images from the **test dataset** to stress-test the RGD model's depth advantage. The two augmentations simulate real-world failure scenarios for RGB-only models:

- **Red regions**: Patches of artificial red/pink discoloration at the tumor border, simulating skin irritation or vascularity that could confuse a color-based model. Depth information is unaffected, so the RGD model should still segment correctly.
- **Shadows**: Darkened arcs along the tumor border (R and G channels reduced), simulating shadow artifacts from lighting. Again, the blue channel (depth) is preserved.

The output is a new COCO JSON dataset with augmented images, used to directly compare RGB vs. RGD inference performance under these adversarial conditions.

---

## Dataset Context

| Property | Value |
|----------|-------|
| Image dimensions | 256×256 pixels (downsampled from 492×492 or 577×577) |
| Modalities | rgb, depth, rgd — all share identical COCO annotations |
| Total annotations | 3,087 across 1,029 test images |
| Split | 700 train / 100 val / 200 test |
| COCO JSON (canonical) | `data/processed_data/rgb/test/images/test.json` |

### Tumor size distribution (256×256 images)

| Percentile | Equiv. radius (px) | BBox width (px) | Area (px²) |
|------------|-------------------|-----------------|------------|
| P10 | 44 | ~70 | 6,113 |
| P25 | 51 | ~90 | 8,240 |
| P50 | 62 | 124 | 12,173 |
| P75 | 79 | ~160 | 19,693 |
| P90 | 95 | ~190 | 28,373 |

Size tier breakdown used by this module:
- **Small**: bbox_width ≤ 90px (~25% of data, equiv_r ~25–45px)
- **Medium**: bbox_width 90–160px (~50% of data, equiv_r ~45–80px)
- **Large**: bbox_width > 160px (~25% of data, equiv_r ~80–120px)

---

## Architecture

```
FailureRecreation (FailureRecreation.py)
├── inherits RedRegionMixin  (_red.py)
└── inherits ShadowMixin     (_shadow.py)
```

**FailureRecreation** is the entry point. It:
1. Loads `config.yaml` (passed in as `yaml_config`)
2. Iterates the test dataset images via the Detectron2 `cfg` object
3. For each image, calls `_extract_values(image, mask)` to determine size tier and build a parameter datastructure
4. Delegates to `RedRegionMixin._create_red_regions()` and/or `ShadowMixin.generate_shadows()` depending on `options` flags
5. Saves augmented images and assembles a new COCO JSON at `output_path`

Each augmentation mixin also saves a per-image YAML file recording all randomly generated values (circle centers, radii, percentages, etc.) for full reproducibility.

**`image_recreation_test()`** runs on a single image — use this for debugging before running the full dataset loop.

---

## config.yaml Field Reference

### `options`
| Field | Type | Description |
|-------|------|-------------|
| `red_region` | bool | Enable red region augmentation |
| `shadows` | bool | Enable shadow augmentation |
| `output_path` | str | Where to write augmented images + COCO JSON |
| `test_mode` | bool | If true, run on a single image only (calls `image_recreation_test`) |

### `size_thresholds`
Thresholds are **bounding box width** values in pixels (for 256×256 images).
- A tumor with bbox_width ≤ `small` uses the `small` config block
- A tumor with bbox_width ≤ `medium` (and > `small`) uses the `medium` config block
- Anything above `medium` uses the `large` config block

### `red_region.<tier>`
| Field | Description |
|-------|-------------|
| `interest_points` | Number of primary circles generated on the tumor border |
| `point_offset_x/y` | Max random ±pixel displacement from the chosen border point (x and y independently) |
| `radius_min/max` | Random radius range for each primary circle (pixels) |
| `red_effect_min/max` | % increase applied to the red channel within the circle region. Overlapping circles stack additively. Clamped to 255. |
| `addtional_points.range` | Max number of sub-circles generated per primary circle (random in [0, range]) |
| `addtional_points.offset_x/y` | Max random offset for sub-circle centers, chosen from pixels inside the parent circle |

### `shadow.<tier>`
| Field | Description |
|-------|-------------|
| `number_regions` | Number of shadow arcs placed on the tumor border (random in [1, number_regions]) |
| `length_min/max` | Length of each shadow arc in border pixels (random per region) |
| `offset_min/max` | Width of the shadow band extending outward from the border (pixels). Defines how far the darkening reaches into surrounding tissue. |
| `decrease_min/max` | % decrease applied to the R and G channels within the shadow region. The B channel (depth) is never modified. |

Note: shadow border segments must not overlap (validated internally).

---

## How to Recalibrate config.yaml When the Dataset Changes

If the dataset changes (new images added, resolution changed, different split), recalibrate as follows.

**Step 1 — Recompute size distribution.**
Parse the canonical COCO JSON:
```python
import json, math, numpy as np

with open("data/processed_data/rgb/test/images/test.json") as f:
    coco = json.load(f)

bbox_widths = [ann["bbox"][2] for ann in coco["annotations"]]
areas = [ann["area"] for ann in coco["annotations"]]
equiv_r = [math.sqrt(a / math.pi) for a in areas]

print(f"BBox width  P25={np.percentile(bbox_widths, 25):.0f}  P50={np.percentile(bbox_widths, 50):.0f}  P75={np.percentile(bbox_widths, 75):.0f}")
print(f"Equiv. r    P25={np.percentile(equiv_r, 25):.0f}  P50={np.percentile(equiv_r, 50):.0f}  P75={np.percentile(equiv_r, 75):.0f}")
```

**Step 2 — Update `size_thresholds`.**
Set `small = P25(bbox_width)` and `medium = P75(bbox_width)`. This keeps each tier covering roughly equal proportions of the dataset.

**Step 3 — Recompute per-tier equiv_r means.**
```python
p25 = np.percentile(bbox_widths, 25)
p75 = np.percentile(bbox_widths, 75)

small_r  = np.mean([r for w, r in zip(bbox_widths, equiv_r) if w <= p25])
medium_r = np.mean([r for w, r in zip(bbox_widths, equiv_r) if p25 < w <= p75])
large_r  = np.mean([r for w, r in zip(bbox_widths, equiv_r) if w > p75])
```

**Step 4 — Scale red_region parameters.**

Use these scaling rules (all values in pixels, for current 256×256 images):

| Parameter | Formula |
|-----------|---------|
| `radius_min` | `round(0.70 * equiv_r_mean)` |
| `radius_max` | `round(1.15 * equiv_r_mean)` |
| `point_offset_x/y` | `round(0.55 * equiv_r_mean)` |
| `addtional_points.offset_x/y` | same as `point_offset_x/y` |
| `interest_points` | small=3, medium=5, large=7 (fixed — scales with perimeter category) |
| `addtional_points.range` | small=2, medium=3, large=4 (fixed) |
| `red_effect_min/max` | keep at 75/100 — color intensity is size-independent |

**Step 5 — Scale shadow parameters.**

| Parameter | Formula |
|-----------|---------|
| `length_min` | `round(0.35 * equiv_r_mean)` |
| `length_max` | `round(0.90 * equiv_r_mean)` |
| `offset_min` | `round(0.55 * equiv_r_mean)` |
| `offset_max` | `round(1.65 * equiv_r_mean)` |
| `number_regions` | small=5, medium=7, large=9 (fixed) |
| `decrease_min/max` | keep at 40/70 — color intensity is size-independent |

**Step 6 — If image resolution changes.**
All pixel values above scale linearly with image resolution. Multiply by `new_dim / 256` when moving to a different resolution.

---

## Per-Image YAML Output Format

Each augmented image produces a companion YAML file recording all random values used:

```yaml
# example for red_region
filename: "032224_MCF7_EdPIT_Control_1_aug.jpg"
circles_generated:
  - center: [132, 87]
    offset: [3, -7]
    radius: 38
    red_effect: 91
    sub_circles:
      - center: [140, 92]
        radius: 22
      - center: [128, 80]
        radius: 19
```

These files are used to reproduce any specific augmented image exactly.

---

## Inferencing

Run inferencing on augmented test sets using the hooks in `src/Detectron2/hooks/`. Compare AP and IoU results between:
1. Baseline (unaugmented test set)
2. Red-region augmented
3. Shadow augmented
4. Both augmented

If both RGB and RGD models degrade equally under augmentation, color information alone may not explain the RGD advantage.
