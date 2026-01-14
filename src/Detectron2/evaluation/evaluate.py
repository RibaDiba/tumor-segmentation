import cv2, os, numpy as np
import matplotlib.pyplot as plt
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.config import get_cfg
from detectron2.data import build_detection_test_loader, MetadataCatalog
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.utils.visualizer import Visualizer

"""
sets of functions below to help evaluate models 
this dir would have different ways to eval models 
"""

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
    cfg.merge_from_file("../training_scripts/cfg.yaml")  # pulls previous settings
    cfg.MODEL_WEIGHTS = os.path.join(model_dir, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # confidence threshold

    return cfg


"""
this visualizes the results from the output 
"""


def visualize(img, outputs, cfg):
    os.makedirs("outputs", exist_ok=True)

    v = Visualizer(img[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TEST[0]))
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # Original image
    ax[0].imshow(img)
    ax[0].set_title("Original Image")
    ax[0].axis("off")

    # Image with predictions (convert back BGR->RGB)
    ax[1].imshow(out.get_image())
    ax[1].set_title("Image with Predictions")
    ax[1].axis("off")

    plt.tight_layout()
    plt.savefig("outputs/test.jpg")
    plt.close()
