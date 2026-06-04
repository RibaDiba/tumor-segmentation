import os, json
import torch
import numpy as np
from pycocotools import mask as mask_util
from detectron2.engine.hooks import HookBase
from detectron2.data import build_detection_test_loader

"""
saves all outputs from the final versions of the model
"""

class OutputsHook(HookBase):
    def __init__(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def after_train(self):
        cfg = self.trainer.cfg
        model_name = cfg.MODELNAME

        all_outputs = {
            "train": self._collect_outputs(cfg.DATASETS.TRAIN),
            "test":  self._collect_outputs(cfg.DATASETS.TEST[0]) if len(cfg.DATASETS.TEST) > 0 else [],
            "val":   self._collect_outputs(cfg.DATASETS.TEST[1]) if len(cfg.DATASETS.TEST) > 1 else [],
        }

        out_path = os.path.join(self.output_dir, f"{model_name}_inference_outputs.json")
        with open(out_path, "w") as f:
            json.dump(all_outputs, f, indent=4)

    def _collect_outputs(self, dataset_name):
        cfg = self.trainer.cfg
        loader = build_detection_test_loader(cfg, dataset_name)

        results = []
        self.trainer.model.eval()
        with torch.no_grad():
            for batch in loader:
                predictions = self.trainer.model(batch)
                for input_data, prediction in zip(batch, predictions):
                    instances = prediction["instances"].to("cpu")
                    entry = {
                        "image_id": input_data.get("image_id"),
                        "file_name": input_data.get("file_name"),
                        "boxes":   instances.pred_boxes.tensor.tolist() if instances.has("pred_boxes") else [],
                        "scores":  instances.scores.tolist()            if instances.has("scores")     else [],
                        "classes": instances.pred_classes.tolist()      if instances.has("pred_classes") else [],
                        "masks":   [],
                    }
                    if instances.has("pred_masks"):
                        for mask in instances.pred_masks:
                            rle = mask_util.encode(np.asfortranarray(mask.numpy()))
                            rle["counts"] = rle["counts"].decode("utf-8")
                            entry["masks"].append(rle)
                    results.append(entry)
        return results
