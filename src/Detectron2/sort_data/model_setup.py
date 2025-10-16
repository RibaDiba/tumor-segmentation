"""
this file is to set up the model and cfg file for sorting 
"""

import os
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo


"""
this returns a cfg file to be used for eval 

args: 
    model_name (str): name of the model, which would correspond to the dir 
    root_dir (str): path of the root dir where the series of models are located 

returns: 
    cfg: configuration file 
"""
def return_cfg(model_name: str, root_dir: str):
    model_dir = os.path.join(root_dir, model_name)

    cfg = get_cfg()
    cfg.MODELNAME = model_name
    cfg.OUTPUT_DIR = f"../../../../models/rgb-testing/{cfg.MODELNAME}"
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("my_dataset_train", "my_dataset_val")
    cfg.DATASETS.TEST = ("my_dataset_test",)
    cfg.DATALOADER.NUM_WORKERS = 1
    cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = False # this is for our "no tumor" examples 
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 2  # This is the real "batch size" commonly known to deep learning people
    cfg.SOLVER.MAX_ITER = 5000   
    cfg.SOLVER.STEPS = []        
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   # The "RoIHead batch size". 128 is faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  

    # # ROI Head Configuration (Accuracy-focused)
    # cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
    # cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION = 0.5  # More positive samples
    # cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3   # Lower detection threshold
    # cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.3     # Lower NMS for medical

    cfg.SOLVER.BASE_LR = 0.0005   # Conservative LR for medical data
    cfg.SOLVER.STEPS = [3000, 4000]  # Later LR reduction
    cfg.SOLVER.GAMMA = 0.5        # Gentler LR decay
    cfg.SOLVER.WARMUP_ITERS = 800
    cfg.SOLVER.WARMUP_FACTOR = 0.1
    cfg.MODEL_WEIGHTS = os.path.join(model_dir, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5 # confidence threshold 

    return cfg

"""
this returns the detectron2 predictor using the cfg 

args: 
    cfg: this is the cfg for our detectron2 model
"""
def return_predictor(cfg):
    return DefaultPredictor(cfg)


