"""
Compute per-variant depth-channel normalization stats over the TRAIN split.

The early-fusion RGBD mapper stacks the raw stored depth values (the
``<stem>_depth.npy`` files) as the 4th input channel; detectron2 then normalizes
with ``MODEL.PIXEL_MEAN``/``MODEL.PIXEL_STD``. This script computes the depth
mean/std over those raw stored values so the 4th entries of PIXEL_MEAN/PIXEL_STD
reflect the real data.

Usage:
    python src/util/preprocessing/compute_depth_stats.py --variant rgbd_rawgrid
    python src/util/preprocessing/compute_depth_stats.py --variant rgbd_contour

Then paste the printed depth mean/std into the 4th entry of the matching config
(configs/<variant>.yaml).
"""

import argparse
import glob
import os

import numpy as np

# project root: .../tumor-segmentation (4 levels up from this file)
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../..")
)


def compute_stats(variant: str, data_root: str) -> tuple:
    img_dir = os.path.join(
        data_root, "data/processed_data", variant, "train/images"
    )
    paths = sorted(glob.glob(os.path.join(img_dir, "*_depth.npy")))
    if not paths:
        raise FileNotFoundError(
            f"no *_depth.npy files under {img_dir} — cache the {variant} "
            "modality first (train.py --split-cache ...)"
        )

    # streaming mean/std via sum and sum-of-squares to avoid loading all at once
    total = 0
    s = 0.0
    s2 = 0.0
    for p in paths:
        arr = np.load(p).astype(np.float64).ravel()
        total += arr.size
        s += arr.sum()
        s2 += np.square(arr).sum()

    mean = s / total
    var = max(s2 / total - mean * mean, 0.0)
    std = float(np.sqrt(var))
    return float(mean), std, len(paths)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--variant",
        choices=("rgbd_contour", "rgbd_rawgrid"),
        required=True,
        help="RGBD variant whose train-split depth stats to compute",
    )
    p.add_argument(
        "--data-root",
        default=PROJECT_ROOT,
        help="project root containing data/processed_data (default: repo root)",
    )
    args = p.parse_args()

    mean, std, n = compute_stats(args.variant, args.data_root)
    # guard against a degenerate std that would blow up normalization
    safe_std = std if std > 1e-6 else 1.0

    print(f"\n{args.variant}: depth stats over {n} train images")
    print(f"  depth mean = {mean:.4f}")
    print(f"  depth std  = {std:.4f}")
    print("\nPaste into configs/%s.yaml (4th entry):" % args.variant)
    print("MODEL:")
    print(f"  PIXEL_MEAN: [103.530, 116.280, 123.675, {mean:.4f}]")
    print(f"  PIXEL_STD: [1.0, 1.0, 1.0, {safe_std:.4f}]")


if __name__ == "__main__":
    main()
