import os, shutil
from tqdm import tqdm
from typing import List

def get_usable_data(src_dir, dest_dir):
    os.makedirs(dest_dir, exist_ok=True)

    moved_files_count = 0
    
    for file_name in tqdm(os.listdir(src_dir), desc=f'Reading {src_dir}'):
        base_name, ext = os.path.splitext(file_name)
        
        jpg_file = f"{base_name}.jpg"
        texture_file = f"{base_name}_texture.jpg"
        bin_file = f"{base_name}.bin"

        required_files = [jpg_file, texture_file, bin_file]
        if all(os.path.exists(os.path.join(src_dir, f)) for f in required_files):
            if not all(os.path.exists(os.path.join(dest_dir, f)) for f in required_files):
                for f in required_files:
                    src_file = os.path.join(src_dir, f)
                    dest_file = os.path.join(dest_dir, f)
                    if os.path.exists(src_file):
                        shutil.copy(src_file, dest_file)
                        moved_files_count += 1
    # TODO: maybe put a better print statement here, this doesnt tell much 
    # print(f"\nTotal images found from {src_dir}: {moved_files_count}")

"""
this is just useful to manually view all the masks and get rid of bad ones for testing 
I usually will delete the directory that this function creates 
"""
def move_files_mask(src_dir: str, new_dir: str):
    os.makedirs(new_dir, exist_ok=True)

    for file_name in os.listdir(src_dir):
        name, ext = os.path.splitext(file_name)
        if ext == ".jpg" and not name.endswith("_texture"): 
            src_file = os.path.join(src_dir, file_name)
            dest_file = os.path.join(new_dir, file_name)
            if os.path.exists(src_file):
                shutil.copy(src_file, dest_file)
    
    print("moved images")

def get_total_images(src_dir):
    triplet_count = 0

    for file_name in tqdm(os.listdir(src_dir), desc=f'Counting images in {src_dir}'):
        base_name, ext = os.path.splitext(file_name)
        
        # only count one type of file 
        if ext.lower() != ".jpg" or base_name.endswith("_texture"):
            continue
        
        jpg_file = f"{base_name}.jpg"
        texture_file = f"{base_name}_texture.jpg"
        bin_file = f"{base_name}.bin"

        required_files = [jpg_file, texture_file, bin_file]
        if all(os.path.exists(os.path.join(src_dir, f)) for f in required_files):
            triplet_count += 1

    print(f"\nTotal Images in {src_dir}: {triplet_count}")
    return triplet_count

def get_num_images(dirs: List[str]):
    num_images_total = 0
    num_images_valid = 0

    total_num_bins = 0

    for i, dir in enumerate(dirs):
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

    # print for logging 
    print(f"Number of total images: {num_images_total}")
    print(f"Number of valid images: {num_images_valid}")
    print(f"Number of invalid images: {num_images_total - num_images_valid}")
    print(f"Number of bin files: {total_num_bins}")


# get_usable_data('data/raw_data/MC_data', 'data/raw_data/useable_data')
# get_usable_data('data/raw_data/invotive', 'data/raw_data/useable_data')
# # get_total_images('data/useable_data')
# get_usable_data('data/raw_data/032224 MCF7 EdPIT', 'data/raw_data/useable_data')
# # move_files_mask("data/raw_data/032224 MCF7 EdPIT", "data/test")