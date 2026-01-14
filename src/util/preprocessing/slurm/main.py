import sys
import os

# 1. Get the directory where main.py is located (.../util/preprocessing/slurm)
current_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Go up 3 levels to find the project root (the folder containing 'util')
#    slurm (1) -> preprocessing (2) -> util (3) -> ROOT
project_root = os.path.abspath(os.path.join(current_dir, "../../.."))

# 3. Add this root to the Python path so "import util" works
if project_root not in sys.path:
    sys.path.append(project_root)

# Debug print to be sure
print(f"Project root added to path: {project_root}")

# NOW you can import
from util.preprocessing.tumor_dataset import Dataset

d = Dataset(
    data_path="/projects/PUCHALLA/LLP2024/tumor-segmentation/data/huggingface-repo/useable_data"
)
d.preprocess_images()
d.split_train_val_test(70, 0, 30)
d.cashe_data()
d.register_instances(rgb=True)
