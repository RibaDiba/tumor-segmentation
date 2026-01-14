import os, sys, argparse
from pathlib import Path

# set the parent dir to import dataset class 
parent_dir = str(Path(__file__).parent.parent.parent)
if parent_dir not in sys.path: 
    sys.path.append(parent_dir)

from util.preprocessing.tumor_dataset import Dataset
from data.data_scripts.get_usable_data import get_usable_data
from data.data_scripts.remove_data import remove_data

# init argparse 
parser = argparse.ArgumentParser(description="Arguments for data upload")

# add args 
parser.add_argument("--dataset-dir", metavar="dataset_dir")

# init vars
args = parser.parse_args()
dataset_dir = args.dataset_dir

# file names that have to be removed
r_filenames = [
    "032224 MCF7 EdPIT-Control-1-17-01",
    "032224 MCF7 EdPIT-Control-1-24-24",
    "032224 MCF7 EdPIT-iv only-1-10-09",
    "032224 MCF7 EdPIT-iv only-1-13-05",
    "013023 4T1 EpP + aPD1-Control-1-2-212-3" # this one has an extra annotation
]

def create_dataset(datatset_dir: str):
    """
    this will split the data into valid sets  
    also will throw it into a useable data 
    """

    useable_data_dir = "../../data/huggingface-repo/useable_data"

    get_usable_data(src_dir=datatset_dir, dest_dir=useable_data_dir)
    remove_data(dir=useable_data_dir, r_filenames=r_filenames) 

def main(): 
    create_dataset(datatset_dir=dataset_dir)

if __name__ == "__main__": 
    main()




    


