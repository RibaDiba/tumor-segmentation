"""
file contains an extention to the default eval framework for detectron2 
"""

from detectron2.evaluation import DatasetEvaluator
from detectron2.structures import BitMasks, pairwise_iou
import torch
import numpy as np

class IoUEvaluator(DatasetEvaluator): 

    @classmethod
    def reset(self): 
        """
        each dict has a format of 
        "image_name": IoU_number 
        """
        self.all_data = {}
        
        self.IOU_90 = {}
        self.IOU_75 = {}
        self.IOI_50 = {}
        self.under_50 = {}
        self.failed = [] # scans with no prediction from the model 

    @classmethod 
    def process(self, inputs, outputs): 
        """
        inputs: refers to dictionary of the dataset 
        outputs: refers to the model outputs of the dataset 
        """

        for input_data, output in zip(inputs, outputs): 

            # store in failed if no prediction 
            if "instances" not in output or "instances" not in input_data: 
                self.failed.append(output["file_name"])
                continue 

            # further catch the posibility of a failed mask 
            pred_instances = output["instances"].to("cpu")
            if not pred_instances.has("pred_masks") or len(pred_instances) == 0:
                self.failed.append(output["file_name"])
                continue
                
            gt_instances = input_data["instances"].to("cpu")
            if not gt_instances.has("gt_masks") or len(gt_instances) == 0:
                self.failed(input_data["file_name"])
                continue





