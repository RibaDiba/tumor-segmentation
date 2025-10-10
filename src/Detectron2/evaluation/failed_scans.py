import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random, importlib, sys
import matplotlib.pyplot as plt

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
import random
import cv2
import matplotlib.pyplot as plt
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.config import get_cfg

# Add the path to the tumor_dataset module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../util/preprocessing'))
import tumor_dataset

importlib.reload(tumor_dataset)
from tumor_dataset import Dataset

"""
this is to temporarily create a small "failed scans" dataset
there will be a folder in the huggingface if we need to investigate further
"""

class FailedScans(Dataset): 

    def __init__(self, model_path, output_path, image_paths=None):
        self.model_path = model_path
        self.output_path = output_path
        
        self.image_paths = image_paths
        self.images = []
        self.images = self.read_images()

        self.init_metadata()
        self.init_model()

    def read_images(self):
        if self.image_paths is None:
            return
    
        for path in self.image_paths: 
            im = cv2.imread(path)
            self.images.append(im)

        return self.images

    def add_image(self, path): 
        im = cv2.imread(path)
        self.images.append(im)

    def preprocess_images(self):
        self.masks = self.create_neg_masks(len(self.images))  
        self.images = self.crop_raw_images(self.images)
        self.images, self.masks = self.add_padding(self.images, self.masks)
        self.images = self.crop_images(self.images)

    def create_graph(self): 
        # Read the image
        im = self.images[0]
    
        # Run prediction
        outputs = self.predictor(im)
        
        # Create visualizer for the original image (no annotations)
        v_original = Visualizer(im[:, :, ::-1],
                            metadata=self.test_metadata,
                            scale=0.5)
        
        # Create visualizer for predictions
        v_pred = Visualizer(im[:, :, ::-1],
                        metadata=self.test_metadata,
                        scale=0.5,
                        instance_mode=ColorMode.SEGMENTATION)
        
        out_pred = v_pred.draw_instance_predictions(outputs["instances"].to("cpu"))
        
        # Extract image name for title
        image_name = os.path.basename(self.image_paths[0])
        model_name = os.path.basename(os.path.dirname(self.model_path)) if self.model_path else "Model"
        
        # Create the plot
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        fig.suptitle(f"{image_name} - {model_name}", fontsize=14, y=0.95)
        
        ax[0].imshow(im)
        ax[0].set_title('Original Image')
        ax[1].imshow(out_pred.get_image()[:, :, ::-1])
        ax[1].set_title('Predicted Image')
        
        for a in ax:
            a.axis("off")

        plt.tight_layout(rect=[0, 0, 1, 0.92])
        
        out = os.path.join(self.output_path, "example.png")
        plt.savefig("example.png")


    def init_metadata(self): 
        """
        change model type here
        """
        self.d = Dataset(data_path="../../data/raw_data/useable_data")
        self.d.convert_binary_to_coco()
        self.d.register_instances(rgb=True)

        self.test_metadata = MetadataCatalog.get("my_dataset_test")
        self.test_dataset_dicts = DatasetCatalog.get("my_dataset_test")

        iterations = 5000
        model_name = "testing-new-LR-4-5000"

        self.cfg = get_cfg()
        self.cfg.MODELNAME = model_name
        self.cfg.OUTPUT_DIR = f"../../../../models/rgb-testing/{self.cfg.MODELNAME}"
        self.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        self.cfg.DATASETS.TRAIN = ("my_dataset_train", "my_dataset_val")
        self.cfg.DATASETS.TEST = ("my_dataset_test",)
        self.cfg.DATALOADER.NUM_WORKERS = 1
        self.cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = False # this is for our "no tumor" examples 
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
        self.cfg.SOLVER.IMS_PER_BATCH = 2  # This is the real "batch size" commonly known to deep learning people
        self.cfg.SOLVER.MAX_ITER = iterations   
        self.cfg.SOLVER.STEPS = []        
        self.cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   # The "RoIHead batch size". 128 is faster, and good enough for this toy dataset (default: 512)
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  

        # # ROI Head Configuration (Accuracy-focused)
        # self.cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
        # self.cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION = 0.5  # More positive samples
        # self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3   # Lower detection threshold
        # self.cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.3     # Lower NMS for medical

        self.cfg.SOLVER.BASE_LR = 0.0005   # Conservative LR for medical data
        self.cfg.SOLVER.STEPS = [3000, 4000]  # Later LR reduction
        self.cfg.SOLVER.GAMMA = 0.5        # Gentler LR decay
        self.cfg.SOLVER.WARMUP_ITERS = 800
        self.cfg.SOLVER.WARMUP_FACTOR = 0.1

    def init_model(self): 
        self.cfg.MODEL.WEIGHTS = os.path.join(self.model_path)  # path to the model we just trained
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set a custom testing threshold
        self.predictor = DefaultPredictor(self.cfg)

def main(): 
    model_path = "/projects/PUCHALLA/LLP2024/tumor-segmentation/models/rgb-testing/rgb-5000-1/model_final.pth"

    fs = FailedScans(model_path=model_path, output_path="", image_paths=["/projects/PUCHALLA/LLP2024/tumor-segmentation/data/huggingface-repo/examples_used_failed/M7-1_texture_1.jpg"])
    fs.preprocess_images()
    fs.create_graph()


if __name__ == "__main__": 
    main()
