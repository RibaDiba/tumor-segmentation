from detectron2.engine import DefaultTrainer
from detectron2.data import (
    build_detection_train_loader,
    DatasetMapper,
    build_detection_test_loader,
)
from detectron2.data import transforms as T
from detectron2.data import detection_utils as utils
from detectron2.engine.hooks import HookBase
from detectron2.utils.events import get_event_storage
from detectron2.evaluation import COCOEvaluator
import matplotlib.pyplot as plt
from collections import defaultdict
import cv2, torch, os, json
import numpy as np

from ..hooks.LossHook import TrainingLossHook
from ..hooks.APHook import APVisualizationHook
from ..hooks.IoUHook import IoUHook
from ..hooks.APFinalHook import AP_IOU_FinalResults
from ..hooks.OutputsHook import OutputsHook

"""
this custom trainer class allows us to include image augmentations
also this is where we can create a hook to visualize training loss 
"""


class Trainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):

        mapper = DatasetMapper(cfg, is_train=True)
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        test_mapper = DatasetMapper(cfg, is_train=False)
        return build_detection_test_loader(cfg, dataset_name, mapper=test_mapper)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        return COCOEvaluator(dataset_name, cfg, False, output_dir=cfg.OUTPUT_DIR)

    def build_hooks(self):
        hooks = super().build_hooks()  # get all hooks

        test_loader = self.build_test_loader(self.cfg, self.cfg.DATASETS.TEST[0])

        val_loss_loader = build_detection_test_loader(
            self.cfg, self.cfg.DATASETS.TEST[1], mapper=DatasetMapper(self.cfg, is_train=True)
        )

        base_out = f"../../../slurm_output/{self.cfg.MODELTYPE}/{self.cfg.MODELNAME}"

        loss_hook_training = TrainingLossHook(
            output_dir=f"{base_out}/loss_plots",
            save_data=True,
            model_name=self.cfg.MODELNAME,
            test_loader=test_loader,
            val_loss_loader=val_loss_loader,
            cfg=self.cfg,
        )  # create an instance of our custom hook
        hooks.append(loss_hook_training)  # append hook

        # now we append all the hooks onto this
        ap_hook = APVisualizationHook(
            output_dir=f"{base_out}/AP_Fig", cfg=self.cfg
        )
        hooks.append(ap_hook)

        iou_hook = IoUHook(output_dir=f"{base_out}/IoU_fig", save_json=True)
        hooks.append(iou_hook)

        outputs_hook = OutputsHook(output_dir=f"{base_out}/outputs")
        hooks.append(outputs_hook)

        # final AP Hook to get scores from the best model
        IoU_AP_Final = AP_IOU_FinalResults(
            output_dir=f"{base_out}/IoU_AP_Final",
            cfg=self.cfg
        )
        hooks.append(IoU_AP_Final)

        return hooks
