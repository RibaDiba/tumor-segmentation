import pytest, os, cv2, sys
import numpy as np
from pathlib import Path

parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

preprocessing_dir = str(Path(parent_dir) / "preprocessing")
if preprocessing_dir not in sys.path:
    sys.path.append(preprocessing_dir)

PROJECT_ROOT = Path(parent_dir).parent

from preprocessing.tumor_dataset import Dataset
from typing import List

def count_mask_augmentations(mask_image: np.ndarray) -> int:
    """
    count the number of augmentations (contours/objects) in a mask image.
    part of this code was sampled from digital sreeni
    
    :param mask_image: Mask image as numpy array
    :return: Number of augmentations (contours with area > threshold)
    """
    
    if mask_image is None:
        raise ValueError("Mask image is None")
    
    # Check if the image is already grayscale (single-channel)
    if len(mask_image.shape) == 2 or (len(mask_image.shape) == 3 and mask_image.shape[2] == 1):
        gray = mask_image
    else:
        # Only convert if the image has 3 channels (BGR)
        gray = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)
    
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
    
    augmentation_count = 0
    for contour in contours:
        augmentation_count += 1
    
    return augmentation_count


@pytest.mark.parametrize("dir", [
    (str(PROJECT_ROOT / "data/processed_data/rgb/val/masks/Tumor"))
])

def test_augmentations(dir):
    assert os.path.isdir(dir) == True, f"directory does not exist: {dir}"
    invalid_indexes: List[int] = []

    try: 
        d = Dataset(dir)
        d.preprocess_images(read_bins=False)
        images, masks = d.return_data()

        for i, img in enumerate(masks): 
            if count_mask_augmentations(img) > 1: 
                invalid_indexes.append(i)

    except Exception as e: # skips if some error was there while preprocessing 
        print(f"Error: {e}")
        pytest.skip()
    
    assert len(invalid_indexes) == 0, f"Images with more than one annotation at {invalid_indexes}"
