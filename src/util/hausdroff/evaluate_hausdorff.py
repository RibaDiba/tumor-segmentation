"""
Compute the average Hausdorff distance (HD95) of a single trained checkpoint on a
COCO-format test set.

Given one Mask R-CNN ``.pth`` checkpoint and a directory holding a COCO test JSON
(plus its images), this builds the matching Detectron2 model, runs inference, and
reports a single ``mean_hd95`` per checkpoint so RGB and RGBD models can be compared
on boundary quality (the complement to the existing AP/IoU metrics).

Per image, each ground-truth tumour is scored against its best-IoU predicted instance
(mirrors the ``best_per_gt`` logic in ``src/pipeline/hooks/iou_evaluator.py``), and the
95th-percentile symmetric Hausdorff distance of that matched pair is recorded. The
headline number is the mean of those HD95 values across all matched tumours.

Usage
-----
python evaluate_hausdorff.py \\
    --checkpoint /models/7030_SPLIT/RGB-1/model_final.pth \\
    --coco-json  data/processed_data/rgb/test/images \\
    --json-name  test.json \\
    --modality   rgb \\
    --output-path hausdorff_rgb.json

For ``--modality rgbd_early`` the ``--coco-json`` directory must also contain the paired
``<stem>_depth.npy`` files that ``RGBDDatasetMapper`` loads as the 4th channel.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from pycocotools import mask as mask_util

# ---------------------------------------------------------------------------
# Path setup — mirrors train.py / _run_inference.py so imports resolve from any cwd
# ---------------------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../.."))
src_dir = os.path.join(project_root, "src")
util_dir = os.path.join(project_root, "src", "util")

for p in (src_dir, util_dir):
    if p not in sys.path:
        sys.path.insert(0, p)

# ---------------------------------------------------------------------------
# Detectron2 / monai imports
# ---------------------------------------------------------------------------
from detectron2 import model_zoo
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, build_detection_test_loader
from detectron2.data.datasets import register_coco_instances
from detectron2.modeling import build_model
from detectron2.utils.logger import setup_logger

from monai.metrics import compute_hausdorff_distance

setup_logger()

CONFIGS_DIR = os.path.join(project_root, "configs")
DATASET_NAME = "hausdorff_eval_test"

# ---------------------------------------------------------------------------
# cfg + model
# ---------------------------------------------------------------------------


def build_cfg(args):
    """Build a Detectron2 cfg for inference, mirroring ``_run_inference._build_cfg``.

    Merges the stock Mask R-CNN R101-FPN config with the modality YAML. For
    ``rgbd_early`` the YAML's 4-entry ``MODEL.PIXEL_MEAN`` is what makes
    ``build_model`` create a 4-channel ``conv1`` matching the trained checkpoint.
    """
    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file(
            "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"
        )
    )
    modality_yaml = os.path.join(args.config_dir, f"{args.modality}.yaml")
    if os.path.isfile(modality_yaml):
        cfg.merge_from_file(modality_yaml)

    cfg.MODEL.WEIGHTS = args.checkpoint
    cfg.DATASETS.TEST = (DATASET_NAME,)
    cfg.DATALOADER.NUM_WORKERS = 1
    cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = False

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.score_thresh
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.3

    if not torch.cuda.is_available():
        cfg.MODEL.DEVICE = "cpu"

    cfg.freeze()
    return cfg


def load_model(cfg):
    """Build the model and load trained weights (eval mode).

    Note: unlike training (``RGBDTrainer.build_model``), we do *not* call
    ``inflate_conv1`` — the trained ``.pth`` already carries 4-channel ``conv1``
    weights for rgbd_early, so ``DetectionCheckpointer.load`` restores them directly.
    """
    weights_path = cfg.MODEL.WEIGHTS
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"Checkpoint not found: {weights_path}")
    model = build_model(cfg)
    DetectionCheckpointer(model).load(weights_path)
    model.eval()
    return model


def build_test_loader(cfg, modality):
    """Build the test loader, using the 4-channel mapper for rgbd_early."""
    if modality == "rgbd_early":
        from pipeline.trainer.rgbd_mapper import RGBDDatasetMapper

        mapper = RGBDDatasetMapper(cfg, is_train=False)
        return build_detection_test_loader(cfg, DATASET_NAME, mapper=mapper)
    return build_detection_test_loader(cfg, DATASET_NAME)


# ---------------------------------------------------------------------------
# Ground-truth masks (polygon -> per-instance binary), mirrors iou_evaluator.py
# ---------------------------------------------------------------------------


def preload_gt_masks(dataset_name):
    """Return ``image_id -> list[(H,W) uint8 binary mask]`` for each GT instance."""
    imageid_to_gt = {}
    for d in DatasetCatalog.get(dataset_name):
        img_id = d.get("image_id", d.get("id"))
        H, W = d.get("height"), d.get("width")
        masks = []
        for ann in d.get("annotations", []) or []:
            seg = ann.get("segmentation")
            if not seg:
                continue
            rles = mask_util.frPyObjects(seg, H, W)
            if isinstance(rles, list):
                dec = mask_util.decode(rles)
                union = np.any(dec, axis=2) if dec.ndim == 3 else dec
            else:
                union = mask_util.decode(rles)
            masks.append((union > 0).astype(np.uint8))
        imageid_to_gt[img_id] = masks
    return imageid_to_gt


def preds_to_binary(instances):
    """Extract a list of (H,W) uint8 binary masks from a prediction Instances."""
    if instances is None or not instances.has("pred_masks"):
        return []
    pm = instances.pred_masks
    if isinstance(pm, torch.Tensor):
        pm = pm.cpu().numpy()
    pm = np.asarray(pm)
    return [(pm[i] > 0).astype(np.uint8) for i in range(pm.shape[0])]


# ---------------------------------------------------------------------------
# HD95 of a matched pair
# ---------------------------------------------------------------------------


def hd95_pair(pred_mask, gt_mask):
    """95th-percentile symmetric Hausdorff distance between two (H,W) binary masks.

    Passed as (1, 1, H, W) tensors — the single channel *is* the tumour, so
    ``include_background=True`` keeps it. Returns a float (``inf``/``nan`` possible
    if a mask is empty; callers should only pass non-empty matched pairs).
    """
    pred_t = torch.as_tensor(pred_mask, dtype=torch.float32)[None, None]
    gt_t = torch.as_tensor(gt_mask, dtype=torch.float32)[None, None]
    hd = compute_hausdorff_distance(
        pred_t, gt_t, include_background=True, percentile=95
    )
    return float(hd.item())


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------


def evaluate(cfg, model, loader, imageid_to_gt, miss_penalty):
    """Run inference and compute per-GT best-match HD95.

    Returns (summary dict, per_image dict).
    """
    per_image = {}
    hd_values = []
    total_gt = matched_gt = missed_gt = 0

    with torch.no_grad():
        for batch in loader:
            outputs = model(batch)
            for inp, out in zip(batch, outputs):
                image_id = inp.get("image_id", inp.get("id"))
                name = os.path.basename(inp.get("file_name", str(image_id)))
                H, W = inp.get("height"), inp.get("width")
                diag = float(np.hypot(H, W))

                gt_masks = imageid_to_gt.get(image_id, [])
                pred_masks = preds_to_binary(out["instances"].to("cpu"))
                num_gt = len(gt_masks)
                total_gt += num_gt

                img_hds = []
                img_missed = 0
                if num_gt and pred_masks:
                    gt_rles = [
                        mask_util.encode(np.asfortranarray(m)) for m in gt_masks
                    ]
                    pred_rles = [
                        mask_util.encode(np.asfortranarray(m)) for m in pred_masks
                    ]
                    iou_mat = mask_util.iou(pred_rles, gt_rles, [0] * num_gt)
                    iou_mat = np.asarray(iou_mat).reshape(len(pred_masks), num_gt)
                    for g in range(num_gt):
                        best = int(iou_mat[:, g].argmax())
                        if iou_mat[best, g] > 0:
                            hd = hd95_pair(pred_masks[best], gt_masks[g])
                            if np.isfinite(hd):
                                img_hds.append(hd)
                                matched_gt += 1
                            else:
                                img_missed += 1
                        else:
                            img_missed += 1
                else:
                    img_missed = num_gt

                # Misses: optionally penalize so a model that detects nothing
                # is not rewarded with a perfect score.
                missed_gt += img_missed
                if miss_penalty is not None and img_missed:
                    penalty = diag if miss_penalty < 0 else miss_penalty
                    img_hds.extend([penalty] * img_missed)

                hd_values.extend(img_hds)
                per_image[name] = {
                    "hd95": float(np.mean(img_hds)) if img_hds else None,
                    "num_gt": num_gt,
                    "matched": num_gt - img_missed,
                    "missed": img_missed,
                }

    summary = {
        "checkpoint": cfg.MODEL.WEIGHTS,
        "modality": cfg.DATASETS.TEST[0] if cfg.DATASETS.TEST else None,
        "mean_hd95": float(np.mean(hd_values)) if hd_values else None,
        "num_images": len(per_image),
        "total_gt": total_gt,
        "matched_gt": matched_gt,
        "missed_gt": missed_gt,
        "match_rate": (matched_gt / total_gt) if total_gt else None,
    }
    return summary, per_image


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="Average Hausdorff distance (HD95) of a checkpoint on a COCO test set."
    )
    parser.add_argument("--checkpoint", required=True, help="Path to trained .pth")
    parser.add_argument(
        "--coco-json",
        required=True,
        help="Directory containing the test COCO json and images",
    )
    parser.add_argument("--json-name", default="test.json", help="JSON filename")
    parser.add_argument(
        "--modality",
        default="rgb",
        choices=("rgb", "depth", "rgd", "rgbd_early"),
        help="Selects config + dataloader mapper",
    )
    parser.add_argument("--config-dir", default=CONFIGS_DIR, help="Modality YAML dir")
    parser.add_argument(
        "--output-path", default=None, help="Optional JSON output path"
    )
    parser.add_argument("--score-thresh", type=float, default=0.3)
    parser.add_argument(
        "--miss-penalty",
        type=float,
        default=None,
        help="Distance assigned to GTs with no overlapping prediction. "
        "Use a negative value to penalize with the per-image diagonal. "
        "Omit to skip misses from the HD mean (still counted in match_rate).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    image_dir = args.coco_json
    json_path = os.path.join(image_dir, args.json_name)
    if not os.path.isfile(json_path):
        raise FileNotFoundError(f"COCO json not found: {json_path}")

    if DATASET_NAME in DatasetCatalog.list():
        DatasetCatalog.remove(DATASET_NAME)
    register_coco_instances(DATASET_NAME, {}, json_path, image_dir)

    cfg = build_cfg(args)
    model = load_model(cfg)
    loader = build_test_loader(cfg, args.modality)
    imageid_to_gt = preload_gt_masks(DATASET_NAME)

    summary, per_image = evaluate(
        cfg, model, loader, imageid_to_gt, args.miss_penalty
    )
    summary["modality"] = args.modality  # overwrite dataset-name placeholder

    print(json.dumps(summary, indent=2))

    if args.output_path:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
        with open(args.output_path, "w") as f:
            json.dump({"summary": summary, "per_image": per_image}, f, indent=2)
        print(f"Wrote per-image results to {args.output_path}")


if __name__ == "__main__":
    main()
