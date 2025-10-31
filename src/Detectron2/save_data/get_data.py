"""
functions to get filenames and create a dictionary 
"""

import os 
from typing import Dict, List
from collections import defaultdict 
from visualize_images import ImageType

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
split - this refers to a split between train/test/val 
basename - the name of the image in the pipeline (ex. text_101.jpg)
image_type - this refers to the type of data used (rgb, depth, rbd)
"""
def get_files() -> Dict: 
    types = [ImageType.RGB, ImageType.DEPTH, ImageType.RGD]
    splits = ["train", "val", "test"]

    base_root = "../../../data/processed_data/"
    names_dict = defaultdict(lambda: defaultdict(dict))
    for image_type in types: 
        for split in splits:
            file_dir = os.path.join(base_root,
                                    image_type.name.lower(), split, "images")
            filenames = get_arr_filenames(file_dir)
            for name in filenames: 
                basename = os.path.basename(name)
                names_dict[split][basename][image_type] = name

    return names_dict
            


