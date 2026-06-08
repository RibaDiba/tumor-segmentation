import os, json, sys
import matplotlib.pyplot as plt
import torch
from detectron2.engine import DefaultTrainer
from detectron2.engine.hooks import HookBase
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.utils.events import get_event_storage

from .iou_evaluator import PerImageIoUEvaluator

"""
this hook will calculate per image IoU mean and give final training metrics 
for the validation set only
"""


class IoUHook(HookBase):
    def __init__(
        self, output_dir=None, save_json: bool = False, eval_period: int = 100, mapper=None
    ):
        self.eval_period = eval_period
        self.output_dir = output_dir
        self.save_json = save_json
        # custom DatasetMapper for 4-channel runs; None -> stock mapper
        self.mapper = mapper

        self.training_data = {}

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

    # record after certain eval period
    def after_step(self):
        storage = get_event_storage()
        iteration = storage.iter

        if iteration % self.eval_period == 0 and iteration != 0:
            results = self._get_IoU()
            self.training_data[iteration] = {
                "count_50": results["dataset_metrics"]["count_50"],
                "count_75": results["dataset_metrics"]["count_75"],
                "count_90": results["dataset_metrics"]["count_90"],
                "count_failed": results["dataset_metrics"]["count_failed"],
            }

    def after_train(self):
        self.model_name = self.trainer.cfg.MODELNAME
        self._save_graph()

        if self.save_json:
            results = self._get_IoU()
            out_path = os.path.join(self.output_dir, "json")
            os.makedirs(out_path, exist_ok=True)
            out_path = os.path.join(out_path, f"{self.model_name}_json_results.json")
            with open(out_path, "w") as f:
                json.dump({"results": results}, f, indent=4)

    # helpers
    def _get_IoU(self):
        cfg = self.trainer.cfg
        evaluator = PerImageIoUEvaluator(cfg.DATASETS.TEST[1])
        val_loader = build_detection_test_loader(
            cfg, cfg.DATASETS.TEST[1], mapper=self.mapper
        )

        results = inference_on_dataset(self.trainer.model, val_loader, evaluator)

        return results

    def _save_graph(self):
        iterations = list(self.training_data.keys())
        count_50 = [v["count_50"] for v in self.training_data.values()]
        count_75 = [v["count_75"] for v in self.training_data.values()]
        count_90 = [v["count_90"] for v in self.training_data.values()]
        count_failed = [v["count_failed"] for v in self.training_data.values()]

        plt.plot(iterations, count_50, label="IoU > 50")
        plt.plot(iterations, count_75, label="IoU > 75")
        plt.plot(iterations, count_90, label="IoU > 90")
        plt.plot(iterations, count_failed, label="IoU < 50")

        plt.title(f"IoU during training - Validation Set - {self.model_name}")
        plt.xlabel("Iteration")
        plt.ylabel("Count")
        plt.legend()
        plt.grid(True)

        save_path = os.path.join(self.output_dir, f"{self.model_name}_IoU.png")
        plt.savefig(save_path)
        plt.close()
