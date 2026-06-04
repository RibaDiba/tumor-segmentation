"""
this file will take the image and create segmentations 
following images will be saved for each raw image 
- image mask 
- RGB segmentation 
- depth segmnetation 
- rgd segmentation

model names here have to be changed manually in the code 
"""

from enum import Enum, auto
import os, sys, cv2
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog, Metadata
from detectron2.utils.visualizer import ColorMode
from collections import defaultdict

_util_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../util"))
if _util_dir not in sys.path:
    sys.path.insert(0, _util_dir)
from paths import MODELS_DIR

"""
defined an enum here that we can use for differing between the different 
model/image types 
"""


class ImageType(Enum):
    RGB = auto()
    DEPTH = auto()
    RGD = auto()


"""
this returns a cfg file to be used for eval 

the root dir for the collection of models has to be 
changed manually as well

args: 
    root_dir (str): path of the root dir where the series of models are located 

returns: 
    cfg: configuration file 
"""


# TODO: make sure this is compatible with different model_types
def return_cfg(model_type: ImageType):
    # these model names are not dynamic, they have
    if model_type == ImageType.RGB:
        model_name = "rgb-5000-1"
    elif model_type == ImageType.DEPTH:
        model_name = "depth-5000-1"
    elif model_type == ImageType.RGD:
        model_name = "rgd-5000-1"

    root_dir = "../../../../models/rgb-testing/"
    model_dir = os.path.join(root_dir, model_name)

    cfg = get_cfg()
    cfg.MODELNAME = model_name
    cfg.OUTPUT_DIR = f"../../../../models/rgb-testing/{cfg.MODELNAME}"
    cfg.merge_from_file(
        model_zoo.get_config_file(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        )
    )
    cfg.DATASETS.TRAIN = ("my_dataset_train", "my_dataset_val")
    cfg.DATASETS.TEST = ("my_dataset_test",)
    cfg.DATALOADER.NUM_WORKERS = 1
    cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = (
        False  # this is for our "no tumor" examples
    )
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    )  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = (
        2  # This is the real "batch size" commonly known to deep learning people
    )
    cfg.SOLVER.MAX_ITER = 5000
    cfg.SOLVER.STEPS = []
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512  # The "RoIHead batch size". 128 is faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1

    # # ROI Head Configuration (Accuracy-focused)
    # cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
    # cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION = 0.5  # More positive samples
    # cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3   # Lower detection threshold
    # cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.3     # Lower NMS for medical

    cfg.SOLVER.BASE_LR = 0.0005  # Conservative LR for medical data
    cfg.SOLVER.STEPS = [3000, 4000]  # Later LR reduction
    cfg.SOLVER.GAMMA = 0.5  # Gentler LR decay
    cfg.SOLVER.WARMUP_ITERS = 800
    cfg.SOLVER.WARMUP_FACTOR = 0.1
    cfg.MODEL_WEIGHTS = os.path.join(model_dir, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # confidence threshold

    model_path = str(MODELS_DIR / "rgb-testing" / model_name / "model_final.pth")
    cfg.MODEL.WEIGHTS = model_path  # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set a custom testing threshold

    return cfg


"""
this returns the detectron2 predictor using the cfg 

args:
    img_dir (str): full directory for one image 
    predictor: detectron2 predictor based on the cfg 
    cfg: this is the cfg for our detectron2 model

"""


def return_predictor(cfg):
    return DefaultPredictor(cfg)


"""
this is a dummy function for the dataset 
"""


def get_dataset_dicts():
    return []


"""
this will take the imahe and run the prediction based on the predictor and cfg type 

will save the visualization to a dir 
"""


def visualize_image(im_dir, predictor, cfg, output_path, name, type):
    im = cv2.imread(im_dir)
    print("Processesing:", im_dir)
    outputs = predictor(im)

    # --- NEW, CLEANER FIX ---
    # Create a brand new, completely separate metadata object
    # just for visualization.
    vis_metadata = Metadata()
    vis_metadata.thing_classes = ["Tumor"]  # Class name for index 0
    vis_metadata.thing_colors = [(255, 0, 0)]  # (R, G, B) for Red
    # -------------------------

    # Create a visualizer object for the predicted image
    v_pred = Visualizer(
        im[:, :, ::-1],
        scale=1.0,
        instance_mode=ColorMode.IMAGE_BW,
        metadata=vis_metadata,  # <-- Pass the new object here
    )
    out_pred = v_pred.draw_instance_predictions(outputs["instances"].to("cpu"))
    annotated_image = out_pred.get_image()[:, :, ::-1]

    if type == ImageType.RGB:
        out_raw = os.path.join(output_path, f"Raw_RGB.png")
        out_annotated = os.path.join(output_path, f"RGB_{name}")
    elif type == ImageType.DEPTH:
        out_raw = os.path.join(output_path, f"Raw_Depth.png")
        out_annotated = os.path.join(output_path, f"DEPTH_{name}")
    else:
        out_raw = os.path.join(output_path, f"Raw_RGD.png")
        out_annotated = os.path.join(output_path, f"RGD_{name}")

    cv2.imwrite(out_annotated, annotated_image)
    cv2.imwrite(out_raw, im)


"""
puts everything together, takes the name dicts and saves the images to the dict 

agrs: 
    names_dict (Dict): this is a dict that stores all the filenames from before
"""


def create_images(names_dict, output_dir):

    rgb_cfg = return_cfg(model_type=ImageType.RGB)
    depth_cfg = return_cfg(model_type=ImageType.DEPTH)
    rgd_cfg = return_cfg(model_type=ImageType.RGD)

    rgb_predictor = return_predictor(rgb_cfg)
    depth_predictor = return_predictor(depth_cfg)
    rgd_predictor = return_predictor(rgd_cfg)

    predictors = {
        ImageType.RGB: rgb_predictor,
        ImageType.DEPTH: depth_predictor,
        ImageType.RGD: rgd_predictor,
    }
    configs = {
        ImageType.RGB: rgb_cfg,
        ImageType.DEPTH: depth_cfg,
        ImageType.RGD: rgd_cfg,
    }

    # these are the paths to be saved
    paths = create_paths(output_dir, names_dict)

    # save image
    for split in names_dict.keys():
        for basename in names_dict[split].keys():

            # current output path for images
            base_path = paths[basename]

            for image_type in names_dict[split][basename].keys():
                visualize_image(
                    im_dir=names_dict[split][basename][image_type],
                    predictor=predictors[image_type],
                    cfg=configs[image_type],
                    output_path=base_path,
                    name=basename,
                    type=image_type,
                )

    return names_dict


"""
this creates paths and returns the list of paths 
"""


def create_paths(root_dir, names_dict):
    paths = defaultdict()

    for split in names_dict.keys():
        for basename in names_dict[split].keys():
            # basename here includes the file ext
            dir_name = os.path.splitext(basename)[0]
            # path removed without the file ext
            path = os.path.join(root_dir, dir_name)
            os.makedirs(path, exist_ok=True)
            paths[basename] = path

    return paths
