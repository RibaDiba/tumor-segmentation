```"""
this file is to run/test the caching pipeline optionally,
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

parser = argparse.ArgumentParser(description="Arguments for caching data")

# test arguments
parser.add_argument("--skip-tests", action="store_true", help="Skip pytest before caching")

# augmentation arguments
parser.add_argument("--augmentations", action="store_true", help="Enable augmentation pipeline")
parser.add_argument("--rotate_degrees", type=int, default=0, help="-/+ range for how much to rotate an image")
args = parser.parse_args()

d = Dataset(data_path=os.path.join(project_root, "data/huggingface-repo/useable_data"))

if args.augmentations:
    d.preprocess_augs(rotate_degrees=args.rotate_degrees)
else:
    d.preprocess_images()
d.split_train_val_test(70, 10, 20)
d.cache_data()
