from detectron2.engine import DefaultTrainer
from detectron2.data import build_detection_train_loader, DatasetMapper, build_detection_test_loader
from detectron2.data import transforms as T
from detectron2.data import detection_utils as utils
from detectron2.engine.hooks import HookBase
from detectron2.utils.events import get_event_storage
from detectron2.evaluation import COCOEvaluator
import matplotlib.pyplot as plt
from collections import defaultdict
import cv2, torch, os, json
import numpy as np

from LossHook import TrainingLossHook
from APHook import APVisualizationHook

"""
this custom trainer class allows us to include image augmentations
also this is where we can create a hook to visualize training loss 
"""

class Trainer(DefaultTrainer):
    @classmethod 
    def build_train_loader(cls, cfg):
        # augs = T.AugmentationList([
        #     T.RandomFlip(0.2, horizontal=True, vertical=False),
        #     T.RandomFlip(0.2, horizontal=False, vertical=True),
        #     T.RandomRotation([-15, 15], expand=False)
        # ])

        mapper = DatasetMapper(cfg)
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        test_mapper = DatasetMapper(cfg, is_train=False)
        return build_detection_test_loader(
            cfg,
            dataset_name,
            mapper=test_mapper
        )
    
    @classmethod 
    def build_evaluator(cls, cfg, dataset_name):
        return COCOEvaluator(dataset_name, cfg, False, output_dir=cfg.OUTPUT_DIR)
    
    def build_hooks(self):
        hooks = super().build_hooks() # get all hooks 
        
        test_loader = self.build_test_loader(self.cfg, self.cfg.DATASETS.TEST[0])

        loss_hook_training = TrainingLossHook(
            output_dir='../slurm_output/loss_plots',
            save_data=True,
            model_name=self.cfg.MODELNAME,
            test_loader=test_loader,
            cfg=self.cfg
        ) # create an instance of our custom hook 
        hooks.append(loss_hook_training) # append hook 

        ap_hook = APVisualizationHook(
            output_dir='../slurm_output/AP_Fig',
            cfg=self.cfg
        )
        hooks.append(ap_hook)
        
        return hooks
    

# custom mapper - not used 
@DeprecationWarning
def tumor_mapper(dataset_dict):
    dataset_dict = dataset_dict.copy()
    image = utils.read_image(dataset_dict["file_name"], format="BGR")

    aug_input = T.StandardAugInput(image)
    transforms = augs(aug_input=aug_input)
    image = aug_input.image

    annos = [
        utils.transform_instance_annotations(obj, transforms, image.shape[:2])
        for obj in dataset_dict.pop("annotations")
    ]
    instances = utils.annotations_to_instances(annos, image.shape[:2])

    return {
        "image": torch.as_tensor(image.transpose(2, 0, 1).astype("float32")),
        "instances": instances,
        "height": image.shape[0], 
        "width": image.shape[1] 
    }

