import os
from pathlib import Path

from detectron2.engine import DefaultTrainer
from detectron2.data import (
    build_detection_train_loader,
    DatasetMapper,
    build_detection_test_loader,
)
from detectron2.evaluation import COCOEvaluator

from ..hooks.loss_hook import TrainingLossHook
from ..hooks.ap_hook import APVisualizationHook
from ..hooks.iou_hook import IoUHook
from ..hooks.ap_final_hook import AP_IOU_FinalResults
from ..hooks.outputs_hook import OutputsHook

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

    def _eval_mapper(self, is_train=False):
        """Mapper handed to the evaluation hooks. ``None`` means "use the stock
        DatasetMapper" (the default). Subclasses (e.g. RGBDTrainer) override this
        to inject a custom mapper without any modality branching here."""
        return None

    def _hook_mapper(self, is_train=False):
        mapper = self._eval_mapper(is_train=is_train)
        return mapper if mapper is not None else DatasetMapper(self.cfg, is_train=is_train)

    def build_hooks(self):
        hooks = super().build_hooks()  # get all hooks

        test_loader = self.build_test_loader(self.cfg, self.cfg.DATASETS.TEST[0])

        val_loss_loader = build_detection_test_loader(
            self.cfg, self.cfg.DATASETS.TEST[1], mapper=self._hook_mapper(is_train=True)
        )

        # mapper forwarded to the eval hooks (None -> they fall back to stock)
        eval_mapper = self._eval_mapper(is_train=False)

        # All hook artifacts live alongside checkpoints in the run's OUTPUT_DIR
        # (models/<TYPE>/<NAME>/run_<ts>), so everything for a run is in one place.
        base_out = self.cfg.OUTPUT_DIR

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
            output_dir=f"{base_out}/AP_Fig", cfg=self.cfg, mapper=eval_mapper
        )
        hooks.append(ap_hook)

        iou_hook = IoUHook(
            output_dir=f"{base_out}/IoU_fig", save_json=True, mapper=eval_mapper
        )
        hooks.append(iou_hook)

        outputs_hook = OutputsHook(output_dir=f"{base_out}/outputs", mapper=eval_mapper)
        hooks.append(outputs_hook)

        # final AP Hook to get scores from the best model
        IoU_AP_Final = AP_IOU_FinalResults(
            output_dir=f"{base_out}/IoU_AP_Final",
            cfg=self.cfg,
            mapper=eval_mapper,
        )
        hooks.append(IoU_AP_Final)

        return hooks

    def build_writers(self):
        writers = super().build_writers()
        if os.environ.get("WANDB_PROJECT"):
            from ..hooks.wandb_writer import WandbWriter

            run_name = (
                f"{self.cfg.MODELTYPE}/{self.cfg.MODELNAME}/"
                f"{Path(self.cfg.OUTPUT_DIR).name}"
            )
            writers.append(
                WandbWriter(
                    project=os.environ["WANDB_PROJECT"],
                    name=run_name,
                    cfg=self.cfg,
                )
            )
        return writers
