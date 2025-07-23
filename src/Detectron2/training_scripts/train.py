"""
this file is the python exec to start training 
before this file is ran there should be some checks to make sure that the data is good 
"""

# imports 
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random, importlib, sys, argparse
import matplotlib.pyplot as plt

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog, DatasetCatalog
import random
import cv2
import matplotlib.pyplot as plt
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.config import get_cfg

# some custom classes 
# TODO: update this with della dir
sys.path.append('/Users/abirmodak/documents/LLP/tumor')
from util.preprocessing.tumor_dataset import Dataset
from src.Detectron2.TrainerClass import Trainer

"""
now we're going to setup argparse here 
we could use the native python solution, but this is easier 
"""

parser = argparse.ArgumentParser(description="arguments for training")

# non-optional arguments 
parser.add_argument('model_name', type=str, help="Specifcies model name")
parser.add_argument('iterations', type=int, help="Specifcies iterations")

# optional arguments
parser.add_argument("-rgb", type=bool, default=False,
                    help="Set mode to RGB")
parser.add_argument('-depth', type=bool, default=False, 
                    help="Set mode to depth")
parser.add_argument('-rgd', type=bool, default=False,
                    help="Set mode to rgd")
parser.add_argument('-split-cashe', metavar="split_cashe", type=bool, default=False,
                    help="splits data into train/val/test, and then cashes it, recemended if first time")

# collect arguments 
args = parser.parse_args()
model_name = args.model_name
iterations = args.iterations
rgb_bool = args.rgb 
depth_bool = args.depth
rgd_bool = args.rgd
split_cashe = args.split_cashe

# check arguments 

"""
code is taken from the notebook file 
"""

# TODO: implement the argparser stuff here 
d = Dataset(data_path="../../../data/raw_data/useable_data")
if split_cashe: 
    d.preprocess_images()
    d.split_train_val_test(70, 15, 15)
    d.cashe_data()
d.convert_binary_to_coco()
if rgb_bool: 
    d.register_instances(rgb=True)
elif depth_bool: 
    d.register_instances(depth=True)
elif rgd_bool: 
    d.register_instances(rgd=True)
else: 
    print("Need to specifiy which type of model needs to be trained")

train_metadata = MetadataCatalog.get("my_dataset_train")
train_dataset_dicts = DatasetCatalog.get("my_dataset_train")

val_metadata = MetadataCatalog.get("my_dataset_val")
val_dataset_dicts = DatasetCatalog.get("my_dataset_val")

test_metadata = MetadataCatalog.get("my_dataset_test")
test_dataset_dicts = DatasetCatalog.get("my_dataset_test")

cfg = get_cfg()
cfg.MODELNAME = model_name
cfg.OUTPUT_DIR = f'../../models/detectron2_test/{cfg.MODELNAME}'
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("my_dataset_train", "my_dataset_val")
cfg.DATASETS.TEST = ("my_dataset_test",)
cfg.DATALOADER.NUM_WORKERS = 1
cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = False # this is for our "no tumor" examples 
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2  # This is the real "batch size" commonly known to deep learning people
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
cfg.SOLVER.MAX_ITER = iterations   
cfg.SOLVER.STEPS = []        
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   # The "RoIHead batch size". 128 is faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  

"""
training code 
"""

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = Trainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()


