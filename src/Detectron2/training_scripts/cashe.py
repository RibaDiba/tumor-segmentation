"""
this file is to run/test the cashing pipeline optionally, 
before training the model 
"""

import argparse, sys, os

# some custom classes
# Add the project root to the path to make imports system-independent
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../.."))
util_dir = os.path.join(project_root, "src", "util")
src_dir = os.path.join(project_root, "src")
if util_dir not in sys.path:
    sys.path.insert(0, util_dir)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from preprocessing.TumorDataset.tumor_dataset import Dataset
from Detectron2.trainer.TrainerClass import Trainer

# helper function for parsing - might add to another module later
def str2bool(v):
    if isinstance(v, bool):
        return v
    elif v.lower() in ("true", "True"):
        return True
    elif v.lower() in ("false", "False"):
        return False
    else:
        raise argparse.ArgumentError("Boolean value expected")

parser = argparse.ArgumentParser(description="Arguments for cashing data")

# test arguments
parser.add_argument("--skip-tests", type=str2bool, default=False, help="Skip pytest before cashing")

# augmentation arguments
parser.add_argument("--augmentations", type=str2bool, default=False, help="Setting for doing augmentations")
parser.add_argument("--flip_prob", type=float, help="probability for flipping")
parser.add_argument("--rotate_prob", type=float, help="Probability for rotating an image")
parser.add_argument("--rotate_degrees", type=int, help="-/+ range for how much to rotate an image")
parser.add_argument("--target", type=int, help="Target amount of images post augmentation")

args = parser.parse_args()

# collect augmentation arguments 
is_augment = args.augmentations 
flip_prob = args.flip_prob
rotate_prob = args.rotate_prob
rotate_degrees = args.rotate_degrees
target = args.target


d = Dataset(data_path=os.path.join(project_root, "data/huggingface-repo/useable_data"))

if is_augment == True:
    d.preprocess_augs(
        target_size=target,
        flip_prob=flip_prob,
        rotate_prob=rotate_prob,
        rotate_degrees=rotate_degrees,
    )
else: 
    d.preprocess_images()
d.split_train_val_test(70, 10, 20)
d.cashe_data()


