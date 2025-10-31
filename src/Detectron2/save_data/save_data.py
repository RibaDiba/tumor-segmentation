"""
the goal of this feature is to allow us to save all 5 image types while also preserving the 
file names. This is for the google sheet 

Image file names are lost when fed into our preprocessing functions, so a slightly different approach
is used here.

the reason that I am running the image through the model is so that you can see the image with the 
processing functions AND the segmentation at the same time 
"""

import argparse, os
from get_data import get_files 
from visualize_images import create_images


# method that takes both and saves it to a dir


"""
main mehtod call 

args: 
    save_path (str): path to save the folders of data 
"""
def main(save_path: str):
    names_dict = get_files()

    create_images(names_dict, save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments for saving data")

    parser.add_argument("save_path", type=str, help="Specifies saving path")
    args = parser.parse_args()
    
    save_path = args.save_path

    main(save_path)