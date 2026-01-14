"""
this file is going to "flag" our data within the training/validation/testing set 
if there are multiple annotations per file, this woild mean that there is something wrong 
with that image 

some methods are going to be copied from the evaluation classes
"""

# TODO: save every image, put into google sheets and do some flagging mechanism

import argparse
from enum import Enum, auto
from get_annotations import get_annotations, get_files
from model_setup import return_cfg, return_predictor
from model_types import Type


def main():

    cfg = return_cfg("rgb-5000-1", "../../../models/rgb-testing")
    predictor = return_predictor(cfg)

    names_dict = get_files(Type.RGB)
    get_annotations(predictor, names_dict=names_dict)


if __name__ == "__main__":
    main()
