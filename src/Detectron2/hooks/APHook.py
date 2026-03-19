import os, json
import matplotlib.pyplot as plt
import torch
from detectron2.engine import DefaultTrainer
from detectron2.engine.hooks import HookBase
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.utils.events import get_event_storage


class APVisualizationHook(HookBase):
    def __init__(self, output_dir: str, cfg, save_data: bool = True):
        super().__init__()
        self.cfg = cfg
        self.output_dir = output_dir  # not from config for now
        self.eval_period = 100
        self.save_data = save_data

        # get info from config
        # use MODEL.NAME, adjust to your config key if different
        self.model_name = cfg.MODELNAME

        self.AP_dict = {}
        self.AP_50 = {}
        self.AP_75 = {}
        os.makedirs(output_dir, exist_ok=True)

    def after_step(self):
        storage = get_event_storage()
        iteration = storage.iter if hasattr(storage, "iter") else storage.iteration

        if iteration % self.eval_period == 0 and iteration != 0:
            self.AP_dict[iteration], self.AP_75[iteration], self.AP_50[iteration] = (
                self._get_ap_numbers()
            )

    def after_train(self):
        plt.plot(list(self.AP_dict.keys()), list(self.AP_dict.values()), label="AP")
        plt.plot(list(self.AP_75.keys()), list(self.AP_75.values()), label="AP75")
        plt.plot(list(self.AP_50.keys()), list(self.AP_50.values()), label="AP50")
        plt.title(f"AP Curve - {self.model_name} - Validation Set")
        plt.xlabel("Iteration")
        plt.ylabel("AP")
        plt.legend()
        plt.grid(True)

        save_path = os.path.join(self.output_dir, f"{self.model_name}_AP_plot.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Saved AP curve to {save_path}")

        if self.save_data:
            self._save_data()

    def _get_ap_numbers(self) -> float:
        cfg = self.trainer.cfg
        evaluator = COCOEvaluator(
            cfg.DATASETS.TEST[1],
            distributed=(cfg.MODEL.DEVICE != "cpu"),
            output_dir=cfg.OUTPUT_DIR,
        )
        val_loader = build_detection_test_loader(cfg, cfg.DATASETS.TEST[1])
        results = inference_on_dataset(self.trainer.model, val_loader, evaluator)

        # pull out just the mAP@[.50:.95]:
        mAP = results.get("segm", {}).get("AP", 0.0)
        mAP75 = results.get("segm", {}).get("AP75", 0.0)
        mAP50 = results.get("segm", {}).get("AP50", 0.0)
        print(f"[Iter {self.trainer.iter} AP] → mAP@[.5:.95] = {mAP:.3f}")
        return mAP, mAP75, mAP50

    def _save_data(self):
        json_out = os.path.join(self.output_dir, f"json_{self.model_name}")
        os.makedirs(json_out, exist_ok=True)

        full_dict = {"AP": self.AP_dict, "AP75": self.AP_75, "AP50": self.AP_50}

        # Save directly in the created directory
        json_file_path = os.path.join(json_out, "results.json")
        with open(json_file_path, "w") as f:
            json.dump(full_dict, f, indent=4)
