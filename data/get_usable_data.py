import os, shutil
from tqdm import tqdm


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
        
    print(f"\nTotal usable images from {src_dir}: {moved_files_count}")

get_usable_data('data/MC_data', 'data/useable_data')
get_usable_data('data/invotive', 'data/useable_data')
