"""
Export the dataset images whose per-image IoU meets a threshold.

Reads the three per-modality ``iou_results.json`` files produced by a training run
(RGB, Depth, RGD), keeps every image whose ``per_image_iou[<name>]["mean_iou"]`` is at
or above ``--threshold``, resolves each name against ``--data-path``, and copies the
matching ``.jpg`` files into ``--output-dir``.

This is the inverse of ``src/util/json_manipulation/find_images.py``, which buckets the
images that fall *below* a threshold and renders 4-panel prediction comparisons. Here
there is no inference, no rendering — just the image files themselves, so a filtered
subset can be inspected or reused directly.

Usage
-----
python filter_by_iou.py \\
    --threshold   0.90 \\
    --rgb-json    models/AUG_1000/RGB-1/IoU_AP_Final/json_RGB-1/iou_results.json \\
    --depth-json  models/AUG_1000/DEPTH-1/IoU_AP_Final/json_DEPTH-1/iou_results.json \\
    --rgd-json    models/AUG_1000/RGD-1/IoU_AP_Final/json_RGD-1/iou_results.json \\
    --data-path   data/processed_data/rgb \\
    --output-dir  src/util/eval/filtered_090

``--data-path`` is a single modality directory (one containing ``train/``, ``val/`` and
``test/``). All three result sets are resolved against it, so you choose which modality's
rendering you want to look at independently of which run's IoU you filter on.

By default each modality gets its own output subdirectory. ``--combine union`` writes one
flat directory of images passing in any modality; ``--combine intersection`` writes only
those passing in all three.
"""

import argparse
import json
import os
import shutil
import sys

# ---------------------------------------------------------------------------
# Path setup — mirrors evaluate_hausdorff.py so imports resolve from any cwd
# ---------------------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
util_dir = os.path.abspath(os.path.join(current_dir, ".."))
if util_dir not in sys.path:
    sys.path.insert(0, util_dir)

from paths import PROCESSED_DATA_DIR

MODALITIES = ("rgb", "depth", "rgd")
SPLITS = ("train", "val", "test")


# ---------------------------------------------------------------------------
# JSON loading
# ---------------------------------------------------------------------------


def _load_iou_json(json_path: str) -> dict:
    """Load an IoU results JSON, tolerating both on-disk shapes.

    ``ap_final_hook.py`` writes the evaluator dict flat, while ``iou_hook.py`` nests it
    under a ``"results"`` key. Both are produced by the same run, so accept either.
    """
    with open(json_path, "r") as f:
        data = json.load(f)
    if "per_image_iou" not in data and isinstance(data.get("results"), dict):
        data = data["results"]
    if "per_image_iou" not in data:
        raise KeyError(f"No 'per_image_iou' block found in {json_path}")
    return data


def _get_passing(json_data: dict, threshold: float) -> set:
    """Basenames whose mean_iou is at or above ``threshold``.

    ``mean_iou`` is ``None`` whenever an image has no ground-truth instances
    (``iou_evaluator.py``), so those entries are skipped rather than compared.
    """
    return {
        img_name
        for img_name, entry in json_data["per_image_iou"].items()
        if entry.get("mean_iou") is not None and entry["mean_iou"] >= threshold
    }


# ---------------------------------------------------------------------------
# Image resolution + copying
# ---------------------------------------------------------------------------


def _resolve_image_path(data_path: str, basename: str) -> str | None:
    """Locate ``basename`` under ``data_path`` by probing each split.

    Mirrors ``find_images.py._resolve_image_path``: the split an image landed in can
    change between the run that wrote the JSON and now, so try all of them.
    """
    for split in SPLITS:
        candidate = os.path.join(data_path, split, "images", basename)
        if os.path.exists(candidate):
            return candidate
    return None


def _copy_images(basenames: set, data_path: str, dest_dir: str) -> tuple[int, list]:
    """Copy each resolved image into ``dest_dir``. Returns (copied, unresolved)."""
    os.makedirs(dest_dir, exist_ok=True)
    copied = 0
    unresolved = []

    for basename in sorted(basenames):
        src = _resolve_image_path(data_path, basename)
        if src is None:
            unresolved.append(basename)
            continue
        shutil.copy2(src, os.path.join(dest_dir, basename))
        copied += 1

    return copied, unresolved


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def filter_by_iou(
    json_paths: dict,
    threshold: float,
    data_path: str,
    output_dir: str,
    combine: str = "per-modality",
) -> dict:
    """Filter each modality's results by ``threshold`` and export the images.

    :param json_paths: mapping of modality name -> iou_results.json path
    :param threshold: minimum mean_iou an image must reach to be exported
    :param data_path: a modality directory containing train/ val/ test/
    :param output_dir: destination root; created if absent, never cleared
    :param combine: one of "per-modality", "union", "intersection"
    :returns: manifest dict describing what was written
    """
    if not os.path.isdir(data_path):
        raise NotADirectoryError(f"--data-path is not a directory: {data_path}")

    passing = {
        modality: _get_passing(_load_iou_json(path), threshold)
        for modality, path in json_paths.items()
    }
    for modality, names in passing.items():
        print(f"{modality}: {len(names)} image(s) with mean_iou >= {threshold}")

    os.makedirs(output_dir, exist_ok=True)
    manifest = {
        "threshold": threshold,
        "combine": combine,
        "data_path": os.path.abspath(data_path),
        "json_paths": {m: os.path.abspath(p) for m, p in json_paths.items()},
        "matched": {m: len(names) for m, names in passing.items()},
        "copied": {},
        "unresolved": {},
    }

    if combine == "per-modality":
        targets = {
            m: (names, os.path.join(output_dir, m)) for m, names in passing.items()
        }
    elif combine == "union":
        targets = {"union": (set().union(*passing.values()), output_dir)}
    elif combine == "intersection":
        targets = {"intersection": (set.intersection(*passing.values()), output_dir)}
    else:
        raise ValueError(f"Unknown combine mode: {combine}")

    for label, (names, dest) in targets.items():
        copied, unresolved = _copy_images(names, data_path, dest)
        manifest["copied"][label] = copied
        manifest["unresolved"][label] = unresolved
        print(f"{label}: copied {copied}/{len(names)} -> {dest}")
        if unresolved:
            print(f"  {len(unresolved)} not found under {data_path}:")
            for name in unresolved:
                print(f"    {name}")

    with open(os.path.join(output_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    return manifest


def parse_args():
    parser = argparse.ArgumentParser(
        description="Copy out the dataset images whose per-image IoU meets a threshold."
    )
    parser.add_argument(
        "--threshold",
        type=float,
        required=True,
        help="Minimum mean_iou an image must reach to be exported",
    )
    parser.add_argument(
        "--rgb-json",
        dest="rgb_json",
        required=True,
        help="Path to the RGB run's iou_results.json",
    )
    parser.add_argument(
        "--depth-json",
        dest="depth_json",
        required=True,
        help="Path to the Depth run's iou_results.json",
    )
    parser.add_argument(
        "--rgd-json",
        dest="rgd_json",
        required=True,
        help="Path to the RGD run's iou_results.json",
    )
    parser.add_argument(
        "--data-path",
        dest="data_path",
        default=str(PROCESSED_DATA_DIR / "rgb"),
        help="Modality directory containing train/ val/ test/ (images are pulled from here)",
    )
    parser.add_argument(
        "--output-dir",
        dest="output_dir",
        required=True,
        help="Destination root for the copied images",
    )
    parser.add_argument(
        "--combine",
        default="per-modality",
        choices=("per-modality", "union", "intersection"),
        help="per-modality: one subdir each; union: passed anywhere; "
        "intersection: passed everywhere",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    filter_by_iou(
        json_paths={
            "rgb": args.rgb_json,
            "depth": args.depth_json,
            "rgd": args.rgd_json,
        },
        threshold=args.threshold,
        data_path=args.data_path,
        output_dir=args.output_dir,
        combine=args.combine,
    )


if __name__ == "__main__":
    main()
