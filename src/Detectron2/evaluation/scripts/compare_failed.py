"""
this file compares all the failed scans across all 3 models and 
saves graphs that show the comparisons 
"""

import argparse



def main(): 
    pass

def _get_json_data():
    pass

def _get_failed_images():
    pass

if __name__ == "__main__": 

    parser = argparse.ArgumentParser(description="Arguments for evaluating data")

    # now we get the 3 paths for the 3 models we want to compare 
    parser.add_argument("rgb_model", type=str, help="Model path for rgb")
    parser.add_argument("depth_model", type=str, help="Model path for depth")
    parser.add_argument("rgd_model", type=str, help="Model path for rgd")

    args = parser.parse_args()

    rgb_path = parser.rgb_model
    depth_path = parser.depth_model
    

    main()