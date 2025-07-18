import pytest, os, cv2

"""
this test is to just show how many sets there are 
prints the number of sets found and number of basenames that are invalid 
"""

@pytest.mark.parametrize("dir", [
    ("../../data/raw_data/useable_data"), # sanity, not really required
    ("../../data/raw_data/invotive"),
    ("../../data/raw_data/MC_Data")
])

def test_sets(dir):
    assert os.path.isdir(dir) == True, "directory does not exist"

    num_images_total = 0
    num_images_valid = 0

    total_num_bins = 0

    for file_name in (os.listdir(dir)):
        base_name, ext = os.path.splitext(file_name)

        # sanity 
        if ext.lower() == ".bin":
            total_num_bins += 1
        
        # this is to make sure that we only count one type of file 
        if ext.lower() == ".jpg" and not base_name.endswith("_texture"):
            num_images_total += 1
            
            jpg_file = f"{base_name}.jpg"
            texture_file = f"{base_name}_texture.jpg"
            bin_file = f"{base_name}.bin"

            required_files = [jpg_file, texture_file, bin_file]
            if all(os.path.exists(os.path.join(dir, f)) for f in required_files):
                num_images_valid += 1
    

    assert num_images_total == num_images_valid, f"""Here is the report of the failed test
    Number of total images: {num_images_total}
    Number of valid images: {num_images_valid}
    Number of invalid images: {num_images_total - num_images_valid}
    """
    
            
