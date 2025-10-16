"""
this file has util functions that pulls from the processed_data dir 
"""
from model_types import Type
from detectron2.layers import batched_nms
from typing import List, Dict
import os, cv2

"""
returns an array of file names based on the dir 

args: 
    dir (str): image directory 
"""     
def get_arr_filenames(dir) -> List[str]:
    image_extensions = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".tiff"}
    
    image_filenames = []

    if not os.path.isdir(dir):
        print(f"Error: Directory not found at '{dir}'")
        return image_filenames

    for filename in os.listdir(dir):
        full_path = os.path.join(dir, filename)

        if os.path.isfile(full_path):
            _, extension = os.path.splitext(filename)
            if extension.lower() in image_extensions:
                image_filenames.append(full_path)
                
    return image_filenames

"""
returns a dict wth all the filenames for 3 different datasets 
"""
def get_files(type: Type) -> Dict:
    match type: 
        case Type.RGB: 
            root_dir = "../../../data/processed_data/rgb"
        case Type.DEPTH: 
            root_dir = "../../../data/processed_data/depth"
        case Type.RGD: 
            root_dir = "../../../data/processed_data/rgd"

    names_dict = dict()
    train_dirs = get_arr_filenames(
        os.path.join(root_dir, "train", "images"))
    val_dirs = get_arr_filenames(
        os.path.join(root_dir, "val", "images"))
    test_dirs = get_arr_filenames(
        os.path.join(root_dir, "test", "images"))
    
    # names_dict["train_dirs"] = train_dirs
    # names_dict["val_dirs"] = val_dirs
    names_dict["test_dirs"] = test_dirs

    return names_dict

"""
returns a dict with image names and their amt of annos 
"""
def get_annotations(predictor, names_dict):

    # Nested for loop to process each image directory and file
    for key in names_dict.keys():
        for dir in names_dict[key]:
            im = cv2.imread(dir)
            if im is None:
                print(f"Warning: Could not read image at {dir}. Skipping.")
                continue

            outputs = predictor(im)
            # Move instances to CPU for processing
            instances = outputs["instances"].to("cpu")

            # First, select only the instances with a score above a threshold.
            # You had 0.5, which is a good starting point.
            score_threshold = 0.1
            high_conf_instances = instances[instances.scores > score_threshold]
            
            # If no instances pass the score threshold, we can stop here.
            if len(high_conf_instances) == 0:
                print(0)
                continue

            # This filters out the redundant, overlapping predictions for the same object.
            # You can tune the iou_threshold. A lower value is stricter.
            iou_threshold = 0.5 
            
            # Prepare the inputs for the NMS algorithm
            boxes_to_filter = high_conf_instances.pred_boxes.tensor
            scores_to_filter = high_conf_instances.scores
            classes_to_filter = high_conf_instances.pred_classes

            # Get the indices of the instances to keep
            keep_indices = batched_nms(boxes_to_filter, scores_to_filter, classes_to_filter, iou_threshold)
            
            # Create the final Instances object containing only the unique detections
            final_instances = high_conf_instances[keep_indices]

            # This will now print the number of unique objects (e.g., 1 or 2)
            print(len(final_instances))

    
