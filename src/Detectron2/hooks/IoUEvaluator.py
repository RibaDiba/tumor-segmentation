# per_image_iou_evaluator_coco_json.py
import os
import json
import numpy as np
from detectron2.evaluation import DatasetEvaluator
from detectron2.data import DatasetCatalog
from pycocotools import mask as maskUtils


class PerImageIoUEvaluator(DatasetEvaluator):
    """
    Loads GT annotations from DatasetCatalog (COCO-format dicts) and computes
    certain metrics of IoU between GT masks and predicted masks.

    Metrics Calculated:
    - Mean IoU of batch
    - Per image IoU
    - IoUs at certain thresholds (50, 75, 90)
    - failed IoUs
    """

    def __init__(self, dataset_name, output_dir=None):
        self._dataset_name = dataset_name
        self._output_dir = output_dir
        self.reset()

        # Preload GT RLEs mapped by image_id
        self._imageid_to_gt = (
            {}
        )  # image_id -> {'rles': [rle,...], 'height': H, 'width': W}
        dataset_dicts = DatasetCatalog.get(dataset_name)
        for d in dataset_dicts:
            img_id = d.get("image_id", d.get("id", None))
            H = d.get("height")
            W = d.get("width")
            rles_for_img = []
            for ann in d.get("annotations", []) or []:
                seg = ann.get("segmentation")
                if not seg:
                    continue
                # frPyObjects can return list of rles (for polygon lists) or a single rle
                rles = maskUtils.frPyObjects(seg, H, W)
                # If it's a list (multiple polygons), decode then union to single mask
                if isinstance(rles, list):
                    dec = maskUtils.decode(rles)  # HxWxN or HxW
                    if dec.ndim == 3:
                        union = np.any(dec, axis=2).astype(np.uint8)
                    else:
                        union = dec.astype(np.uint8)
                    rle_union = maskUtils.encode(np.asfortranarray(union))
                    rles_for_img.append(rle_union)
                else:
                    # frPyObjects might directly return an rle dict
                    rles_for_img.append(rles)
            self._imageid_to_gt[img_id] = {
                "rles": rles_for_img,
                "height": H,
                "width": W,
            }

    def reset(self):
        self.image_results = (
            {}
        )  # image_id -> {'mean_iou': float or None, 'num_gt', 'num_pred'}

        # dicts below are the same format
        self.IoU_50 = {}
        self.IoU_75 = {}
        self.IoU_90 = {}

        self.IoU_failed = {}

        self._ordered_image_ids = []

    def _preds_to_rles(self, pred_masks):
        """
        pred_masks: list/array of binary masks (H,W) or a (P,H,W) numpy array / torch tensor.
        Return list of rles accepted by pycocotools.
        """
        rles = []
        if pred_masks is None:
            return rles
        # convert to numpy arrays if tensors present
        try:
            import torch

            if isinstance(pred_masks, torch.Tensor):
                pred_masks = pred_masks.cpu().numpy()
        except Exception:
            pass

        pred_masks = np.asarray(pred_masks)
        if pred_masks.ndim == 2:
            ms = [pred_masks]
        elif pred_masks.ndim == 3:
            ms = [pred_masks[i] for i in range(pred_masks.shape[0])]
        else:
            ms = []

        for m in ms:
            m_bin = (m > 0).astype(np.uint8)
            rle = maskUtils.encode(np.asfortranarray(m_bin))
            rles.append(rle)
        return rles

    def process(self, inputs, outputs):
        # inputs and outputs are lists for a batch
        for inp, out in zip(inputs, outputs):
            # try to get image id (COCO usually uses ints)
            image_id = inp.get("image_id", inp.get("id", None))
            image_name = inp.get("file_name")
            image_name = os.path.basename(image_name)

            if image_id not in self._ordered_image_ids:
                self._ordered_image_ids.append(image_id)

            # get GT rles we preloaded
            gt_entry = self._imageid_to_gt.get(
                image_id,
                {"rles": [], "height": inp.get("height"), "width": inp.get("width")},
            )
            gt_rles = gt_entry["rles"]

            # get predicted masks from model outputs
            pred_masks = []
            if out is not None and "instances" in out:
                inst = out["instances"]
                # many Detectron2 models have inst.pred_masks as torch.Tensor (P,H,W)
                if hasattr(inst, "pred_masks"):
                    pm = inst.pred_masks
                else:
                    # maybe there are no masks
                    pm = None
                pred_masks = self._preds_to_rles(pm)
            else:
                pred_masks = []

            # record counts
            num_gt = len(gt_rles)
            num_pred = len(pred_masks)

            if num_gt == 0:
                mean_iou = None
            elif num_pred == 0:
                mean_iou = 0.0
            else:
                # compute IoU matrix (P, G) using pycocotools; default iscrowd=0
                iscrowd = [0] * num_gt
                try:
                    iou_mat = maskUtils.iou(pred_masks, gt_rles, iscrowd)  # shape (P,G)
                    # For each GT choose best-pred; average over GTs
                    best_per_gt = (
                        iou_mat.max(axis=0)
                        if iou_mat.size
                        else np.zeros((num_gt,), dtype=float)
                    )
                    mean_iou = float(best_per_gt.mean())
                except Exception as e:
                    # fallback: mark as None and continue
                    mean_iou = None
                    print(
                        f"[PerImageIoU] pycocotools.iou failed for image {image_id}: {e}"
                    )

            self.image_results[image_name] = {
                "id": image_id,
                "mean_iou": mean_iou,
                "num_gt": num_gt,
                "num_pred": num_pred,
            }
            if mean_iou is not None and mean_iou >= 0.90:
                self.IoU_90[image_name] = {
                    "id": image_id,
                    "mean_iou": mean_iou,
                    "num_gt": num_gt,
                    "num_pred": num_pred,
                }
            if mean_iou is not None and mean_iou >= 0.75:
                self.IoU_75[image_name] = {
                    "id": image_id,
                    "mean_iou": mean_iou,
                    "num_gt": num_gt,
                    "num_pred": num_pred,
                }
            if mean_iou is not None and mean_iou >= 0.50:
                self.IoU_50[image_name] = {
                    "id": image_id,
                    "mean_iou": mean_iou,
                    "num_gt": num_gt,
                    "num_pred": num_pred,
                }
            if mean_iou is None or mean_iou < 0.50:
                self.IoU_failed[image_name] = {
                    "id": image_id,
                    "mean_iou": mean_iou,
                    "num_gt": num_gt,
                    "num_pred": num_pred,
                }

    def evaluate(self):
        per_image = self.image_results
        ious = [v["mean_iou"] for v in per_image.values() if v["mean_iou"] is not None]
        overall_mean = float(np.mean(ious)) if ious else None

        count_50 = len(self.IoU_50)
        count_75 = len(self.IoU_75)
        count_90 = len(self.IoU_90)
        count_failed = len(self.IoU_failed)

        # while this is useful, due to the model loading issue, I will be loading the json from the hook
        if self._output_dir:
            os.makedirs(self._output_dir, exist_ok=True)
            out_path = os.path.join(self._output_dir, "per_image_iou.json")
            with open(out_path, "w") as f:
                json.dump(
                    {"per_image": per_image, "overall_mean": overall_mean}, f, indent=2
                )

        return {
            "per_image_iou": per_image,
            "per_image_50": self.IoU_50,
            "per_image_75": self.IoU_75,
            "per_image_90": self.IoU_90,
            "per_image_failed": self.IoU_failed,
            "dataset_metrics": {
                "mean_iou": overall_mean,
                "count_50": count_50,
                "count_75": count_75,
                "count_90": count_90,
                "count_failed": count_failed,
            },
        }
