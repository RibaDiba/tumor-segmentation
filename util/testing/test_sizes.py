import pytest, os, cv2 
from typing import List, Dict

"""
these will be tests on each individual directory 
this makes sure the following 
- makes sure that all the images are a certain size 

the expected_sizes are the current supported sizes in our preprocessing functions 
"""

expected_sizes = {
    # [invotive, MC_data]
    "mask_sizes": [[577, 615], [492, 495]],
    "raw_image_sizes": [[640, 480], [640, 480]]
}

@pytest.mark.parametrize("dir, expected_sizes", [
    # ("../../data/raw_data/useable_data", expected_sizes), # sanity 
    # ("../../data/raw_data/J&j-1", expected_sizes),
    # ("../../data/raw_data/J&j-2", expected_sizes),
    # ("../../data/raw_data/J&j-3", expected_sizes),
    # ("../../data/raw_data/J&j-4", expected_sizes),
    # ("../../data/raw_data/J&j-5", expected_sizes),
    # ("../../data/raw_data/J&j-6", expected_sizes),
    # ("../../data/raw_data/J&j-7", expected_sizes),
    # ("../../data/raw_data/J&j-8", expected_sizes),
    # ("../../data/raw_data/J&j-9", expected_sizes),
    # ("../../data/raw_data/J&j-10", expected_sizes),
    # ("../../data/raw_data/J&j-10", expected_sizes),
    # ("../../data/raw_data/J&j-11", expected_sizes),
    # ("../../data/raw_data/J&j-12", expected_sizes),
    # ("../../data/raw_data/J&j-13", expected_sizes),
    # ("../../data/raw_data/J&j-14", expected_sizes),
    # ("../../data/raw_data/DTC #326 Scan Images", expected_sizes),
    # ("../../data/raw_data/DTC #347 Scan Images", expected_sizes),
    # ("../../data/raw_data/DTC #357", expected_sizes),
    ("../../data/raw_data/S-065-006", expected_sizes),
    ("../../data/raw_data/S-069-012", expected_sizes),
    ("../../data/raw_data/Test Nude - 073-009", expected_sizes),
])

def test_image_size_raw(dir: str, expected_sizes: Dict):
    assert os.path.isdir(dir) == True, "directory does not exist"
    new_sizes = []

    for filename in os.listdir(dir):
        img = cv2.imread(os.path.join(dir, filename))
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
    ("../../data/raw_data/useable_data", expected_sizes), # sanity 
    ("../../data/raw_data/J&j-1", expected_sizes),
    ("../../data/raw_data/J&j-2", expected_sizes),
    ("../../data/raw_data/J&j-3", expected_sizes),
    ("../../data/raw_data/J&j-4", expected_sizes),
    ("../../data/raw_data/J&j-5", expected_sizes),
    ("../../data/raw_data/J&j-6", expected_sizes),
    ("../../data/raw_data/J&j-7", expected_sizes),
    ("../../data/raw_data/J&j-8", expected_sizes),
    ("../../data/raw_data/J&j-9", expected_sizes),
    ("../../data/raw_data/J&j-10", expected_sizes),
    ("../../data/raw_data/J&j-10", expected_sizes),
    ("../../data/raw_data/J&j-11", expected_sizes),
    ("../../data/raw_data/J&j-12", expected_sizes),
    ("../../data/raw_data/J&j-13", expected_sizes),
    ("../../data/raw_data/J&j-14", expected_sizes),
    ("../../data/raw_data/DTC #326 Scan Images", expected_sizes),
    ("../../data/raw_data/DTC #347 Scan Images", expected_sizes),
    ("../../data/raw_data/DTC #357", expected_sizes),
    ("../../data/raw_data/S-065-006", expected_sizes),
    ("../../data/raw_data/S-069-012", expected_sizes),
    ("../../data/raw_data/Test Nude - 073-009", expected_sizes),
])

def test_image_size_mask(dir: str, expected_sizes: Dict):
    assert os.path.isdir(dir) == True, "directory does not exist"
    new_sizes = []

    for filename in os.listdir(dir):
        img = cv2.imread(os.path.join(dir, filename))
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