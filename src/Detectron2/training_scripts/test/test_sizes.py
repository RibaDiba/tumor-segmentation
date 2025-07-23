import pytest, os, cv2 
from typing import List, Dict
from pathlib import Path

"""
these will be tests on each individual directory 
this makes sure the following 
- makes sure that all the images are a certain size 

the expected_sizes are the current supported sizes in our preprocessing functions 
"""

# Get project root directory
project_root = Path(__file__).parent.parent.parent.parent.parent

expected_sizes = {
    # [invotive, MC_data]
    "mask_sizes": [[577, 615], [492, 495]],
    "raw_image_sizes": [[640, 480], [640, 480]]
}

@pytest.mark.parametrize("dir, expected_sizes", [
    (project_root / "data/raw_data/useable_data", expected_sizes)
])

def test_image_size_raw(dir, expected_sizes: Dict):
    # Convert to string if it's a Path object for compatibility with os functions
    dir_str = str(dir)
    assert os.path.isdir(dir_str) == True, "directory does not exist"
    new_sizes = []

    for filename in os.listdir(dir_str):
        img = cv2.imread(os.path.join(dir_str, filename))
        match_found = False

        if img is None: 
            continue

        basename, ext = os.path.splitext(filename)
        height, width = img.shape[:2]

        # we are only looking at raw images for this test
        if ext == '.jpg' and basename.endswith("_texture"):
            for size in expected_sizes["raw_image_sizes"]:
                expected_width, expected_height = size

                if expected_width == width and expected_height == height: 
                    match_found = True
                    break
            
            if not match_found: 
                img_size = [width, height]
                if img_size not in new_sizes:
                    new_sizes.append(img_size)
        
    
    assert len(new_sizes) == 0, f"{dir} has invalid dimensions of {new_sizes}"

@pytest.mark.parametrize("dir, expected_sizes", [
    (project_root / "data/raw_data/useable_data", expected_sizes)
])

def test_image_size_mask(dir, expected_sizes: Dict):
    # Convert to string if it's a Path object for compatibility with os functions
    dir_str = str(dir)
    assert os.path.isdir(dir_str) == True, "directory does not exist"
    new_sizes = []

    for filename in os.listdir(dir_str):
        img = cv2.imread(os.path.join(dir_str, filename))
        match_found = False

        if img is None: 
            continue

        basename, ext = os.path.splitext(filename)
        height, width = img.shape[:2]

        # we are only looking at raw images for this test
        if ext == '.jpg' and not basename.endswith("_texture"):
            for size in expected_sizes["mask_sizes"]:
                expected_width, expected_height = size

                if expected_width == width and expected_height == height: 
                    match_found = True
                    break
            
            if not match_found: 
                img_size = [width, height]
                if img_size not in new_sizes:
                    new_sizes.append(img_size)
        
    
    assert len(new_sizes) == 0, f"{dir} has invalid dimensions of {new_sizes}"