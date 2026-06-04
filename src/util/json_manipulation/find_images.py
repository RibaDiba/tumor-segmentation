import json
import os
import shutil

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from detectron2.structures import Boxes, BoxMode, Instances
from detectron2.utils.visualizer import Visualizer
from pycocotools import mask as mask_util
from tqdm import tqdm

matplotlib.use("Agg")


def _load_iou_json(json_path: str) -> dict:
    with open(json_path, "r") as f:
        return json.load(f)


def _load_inference_outputs(json_path: str) -> dict:
    """Load inference outputs JSON, indexed by image basename (test + val splits)."""
    with open(json_path, "r") as f:
        data = json.load(f)
    index = {}
    for split in ("test", "val"):
        for entry in data.get(split, []):
            index[os.path.basename(entry["file_name"])] = entry
    return index


def _load_coco_gt(coco_json_path: str) -> dict:
    """Load COCO JSON and index annotations by image filename."""
    with open(coco_json_path, "r") as f:
        coco = json.load(f)
    id_to_name = {img["id"]: img["file_name"] for img in coco["images"]}
    gt_index = {}
    for ann in coco["annotations"]:
        fname = id_to_name[ann["image_id"]]
        gt_index.setdefault(fname, []).append(ann)
    return gt_index


def _get_below_threshold(json_data: dict, threshold: float) -> set:
    return {
        img_name
        for img_name, entry in json_data["per_image_iou"].items()
        if entry["mean_iou"] < threshold
    }


def _render_prediction(img_bgr: np.ndarray, entry: dict) -> np.ndarray:
    """Reconstruct Instances from a saved inference entry and render via Visualizer."""
    h, w = img_bgr.shape[:2]
    instances = Instances((h, w))

    if len(entry["boxes"]) > 0:
        instances.pred_boxes = Boxes(torch.tensor(entry["boxes"], dtype=torch.float32))
        instances.scores = torch.tensor(entry["scores"], dtype=torch.float32)
        instances.pred_classes = torch.tensor(entry["classes"], dtype=torch.int64)

        decoded_masks = []
        for rle in entry["masks"]:
            rle_copy = {"size": rle["size"], "counts": rle["counts"].encode("utf-8")}
            decoded_masks.append(mask_util.decode(rle_copy))
        instances.pred_masks = torch.from_numpy(np.stack(decoded_masks)).bool()
    else:
        instances.pred_boxes = Boxes(torch.empty((0, 4), dtype=torch.float32))
        instances.scores = torch.empty(0, dtype=torch.float32)
        instances.pred_classes = torch.empty(0, dtype=torch.int64)
        instances.pred_masks = torch.empty((0, h, w), dtype=torch.bool)

    v = Visualizer(img_bgr[:, :, ::-1])
    vis_output = v.draw_instance_predictions(instances)
    return vis_output.get_image()[:, :, ::-1]  # back to BGR for cv2.imwrite


def _render_ground_truth(img_bgr: np.ndarray, annotations: list) -> np.ndarray:
    """Draw COCO ground-truth annotations on an image using Detectron2 Visualizer."""
    h, w = img_bgr.shape[:2]
    dataset_dict = {
        "height": h,
        "width": w,
        "annotations": [
            {
                "bbox": ann["bbox"],
                "bbox_mode": BoxMode.XYWH_ABS,
                "segmentation": ann["segmentation"],
                "category_id": 0,  # Visualizer expects 0-indexed
            }
            for ann in annotations
        ],
    }
    v = Visualizer(img_bgr[:, :, ::-1])
    vis_output = v.draw_dataset_dict(dataset_dict)
    return vis_output.get_image()[:, :, ::-1]


def _create_comparison_plot(
    pred_images: dict, gt_image: np.ndarray | None, save_path: str
) -> None:
    """Create a 1x4 subplot with RGB/Depth/RGD predictions and ground truth."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    panels = [
        ("RGB Pred", pred_images.get("rgb")),
        ("Depth Pred", pred_images.get("depth")),
        ("RGD Pred", pred_images.get("rgd")),
        ("Ground Truth", gt_image),
    ]
    for ax, (title, img) in zip(axes, panels):
        if img is not None:
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        else:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center", fontsize=14)
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _resolve_image_path(file_name: str) -> str | None:
    """
    Resolve an image path from an inference entry's file_name.

    Tries the stored path first. If it doesn't exist (e.g. dataset was
    re-split or augmented after inference), searches for the same basename
    across train/val/test splits under the same mode directory.
    """
    if os.path.exists(file_name):
        return file_name

    # file_name looks like .../processed_data/{mode}/{split}/images/{basename}
    # Go up from {split}/images/ to the mode dir, then try other splits.
    images_dir = os.path.dirname(file_name)        # .../mode/split/images
    split_dir = os.path.dirname(images_dir)         # .../mode/split
    mode_dir = os.path.dirname(split_dir)            # .../mode
    basename = os.path.basename(file_name)

    for split in ("train", "val", "test"):
        candidate = os.path.join(mode_dir, split, "images", basename)
        if os.path.exists(candidate):
            return candidate

    return None


def _process_images(
    filenames: set,
    outputs: dict,
    dest_dir: str,
    label: str,
    iou_data: dict,
    gt_index: dict,
) -> list:
    """
    For each failing image, create a named subfolder under dest_dir containing:
      - comparison.png     (4-panel subplot: RGB/Depth/RGD predictions + GT)
      - iou_comparison.json (IoU values from all 3 models)
    Returns list of filenames whose images could not be found in any mode.
    """
    os.makedirs(dest_dir, exist_ok=True)
    not_found = []

    print("Finding images that go below your threshold...")

    for filename in tqdm(filenames, desc=label, unit="img"):
        stem = os.path.splitext(filename)[0]
        folder = os.path.join(dest_dir, stem)
        os.makedirs(folder, exist_ok=True)

        found_any = False
        pred_renders = {}
        rgb_img_bgr = None

        for mode in ["rgb", "depth", "rgd"]:
            entry = outputs[mode].get(filename)
            if entry is None:
                print(f"  Warning: no saved predictions for {filename} in {mode}")
                continue

            img_path = _resolve_image_path(entry["file_name"])
            if img_path is None:
                print(f"  Warning: image not found: {entry['file_name']}")
                continue

            found_any = True
            img = cv2.imread(img_path)
            pred_renders[mode] = _render_prediction(img, entry)

            if mode == "rgb":
                rgb_img_bgr = img

        # Render ground truth on the RGB image
        gt_image = None
        if rgb_img_bgr is not None:
            gt_anns = gt_index.get(filename, [])
            if gt_anns:
                gt_image = _render_ground_truth(rgb_img_bgr, gt_anns)
            else:
                gt_image = rgb_img_bgr.copy()

        if found_any:
            _create_comparison_plot(
                pred_renders, gt_image, os.path.join(folder, "comparison.png")
            )

        # Write IoU comparison JSON
        iou_comp = {"image": filename}
        for mode in ["rgb", "depth", "rgd"]:
            per_image = iou_data[mode].get("per_image_iou", {})
            entry = per_image.get(filename)
            iou_comp[f"{mode}_iou"] = entry["mean_iou"] if entry else None
        with open(os.path.join(folder, "iou_comparison.json"), "w") as f:
            json.dump(iou_comp, f, indent=2)

        if not found_any:
            not_found.append(filename)

    return not_found


def get_threshold(
    rgb_json: str,
    depth_json: str,
    rgd_json: str,
    rgb_outputs_json: str,
    depth_outputs_json: str,
    rgd_outputs_json: str,
    threshold: float,
    output_dir: str,
    coco_json_paths: list[str] | None = None,
) -> None:
    """
    Creates a folder of per-image subfolders for images with mean_iou below threshold.

    Reads three iou_results.json files (one per model: RGB, Depth, RGD), finds
    every image whose mean_iou is below `threshold`, and writes categorized
    subdirectories under output_dir:

        output_dir/
          RGB/      <- failed only in RGB
            <image_stem>/
              comparison.png      (4-panel subplot)
              iou_comparison.json (IoU from all 3 models)
          Depth/    <- failed only in Depth
          RGD/      <- failed only in RGD
          Overlap/  <- failed in 2+ models

    Prediction overlays are rendered from pre-saved inference outputs using
    the Detectron2 Visualizer (no live model inference).
    """
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    print("Loading inference outputs...")
    outputs = {
        "rgb": _load_inference_outputs(rgb_outputs_json),
        "depth": _load_inference_outputs(depth_outputs_json),
        "rgd": _load_inference_outputs(rgd_outputs_json),
    }

    iou_data = {
        "rgb": _load_iou_json(rgb_json),
        "depth": _load_iou_json(depth_json),
        "rgd": _load_iou_json(rgd_json),
    }

    rgb_bad = _get_below_threshold(iou_data["rgb"], threshold)
    depth_bad = _get_below_threshold(iou_data["depth"], threshold)
    rgd_bad = _get_below_threshold(iou_data["rgd"], threshold)

    # Load ground-truth annotations from COCO JSONs
    gt_index = {}
    if coco_json_paths:
        print("Loading COCO ground-truth annotations...")
        for path in coco_json_paths:
            gt_index.update(_load_coco_gt(path))

    all_bad = rgb_bad | depth_bad | rgd_bad
    overlap = {img for img in all_bad if sum([img in rgb_bad, img in depth_bad, img in rgd_bad]) >= 2}

    rgb_only = rgb_bad - depth_bad - rgd_bad
    depth_only = depth_bad - rgb_bad - rgd_bad
    rgd_only = rgd_bad - rgb_bad - depth_bad

    categories = {
        "RGB": rgb_only,
        "Depth": depth_only,
        "RGD": rgd_only,
        "Overlap": overlap,
    }

    all_not_found = []
    for folder, filenames in tqdm(categories.items(), desc="Categories", unit="cat"):
        dest = os.path.join(output_dir, folder)
        not_found = _process_images(
            filenames, outputs, dest, label=folder,
            iou_data=iou_data, gt_index=gt_index,
        )
        all_not_found.extend(not_found)
        print(f"{folder}: {len(filenames)} images \u2192 {dest}")

    if all_not_found:
        print(f"\nWarning: {len(all_not_found)} image(s) not found in any modality:")
        for name in all_not_found:
            print(f"  {name}")


def main():
    import sys
    _util_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if _util_dir not in sys.path:
        sys.path.insert(0, _util_dir)
    from paths import PROJECT_ROOT
    base = PROJECT_ROOT
    get_threshold(
        rgb_json=str(base / "src/pipeline/slurm_output/AUG_1000/RGB-1/IoU_AP_Final/json_RGB-1/iou_results.json"),
        depth_json=str(base / "src/pipeline/slurm_output/AUG_1000/DEPTH-1/IoU_AP_Final/json_DEPTH-1/iou_results.json"),
        rgd_json=str(base / "src/pipeline/slurm_output/AUG_1000/RGD-1/IoU_AP_Final/json_RGD-1/iou_results.json"),
        rgb_outputs_json=str(base / "src/pipeline/slurm_output/AUG_1000/RGB-1/outputs/RGB-1_inference_outputs.json"),
        depth_outputs_json=str(base / "src/pipeline/slurm_output/AUG_1000/DEPTH-1/outputs/DEPTH-1_inference_outputs.json"),
        rgd_outputs_json=str(base / "src/pipeline/slurm_output/AUG_1000/RGD-1/outputs/RGD-1_inference_outputs.json"),
        threshold=0.75,
        output_dir=str(base / "src/util/failure_analysis"),
        coco_json_paths=[
            str(base / "data/processed_data/rgb/test/images/test.json"),
            str(base / "data/processed_data/rgb/val/images/val.json"),
        ],
    )


if __name__ == "__main__":
    main()
