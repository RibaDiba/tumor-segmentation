import numpy as np
import cv2, os, re, shutil
from typing import List


# this is made for individual directories
def filter_subset_in_folder(self, arr: List[int], folder_path):
    exts = [".jpg", ".png"]
    files = [
        f
        for f in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, f))
        and os.path.splitext(f)[1].lower() in exts
    ]

    image_array = []

    for f in files:
        file_num = int(re.search(r"\d+", f).group())
        if file_num in arr:
            print(f"Image {file_num} is removed")
        else:
            img = cv2.imread(os.path.join(folder_path, f))
            image_array.append(img)

    return image_array


def remove_files_in_dir(self, folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.remove(file_path)  # remove file or symbolic link
            elif os.path.isdir(file_path):
                # this is bassically if for some reason there is a folder
                continue
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")


# saves the new image array
def save_subset_array(self, folder_path: str, image_array: List[np.ndarray], type: str):
    for i, img in enumerate(image_array):
        # could use original names here but max
        cv2.imwrite(os.path.join(folder_path, f"{type}_{i}.jpg"), img)


def subset_automation(self, dir_array, arr):
    for dir in dir_array:
        # First, get the filtered array of images to keep
        image_array = self.filter_subset_in_folder(arr=arr, folder_path=dir)

        # Create a unique backup directory for each original directory
        dir_name = os.path.basename(dir)
        backup_dir = os.path.join(os.path.dirname(dir), f"temp_backup_{dir_name}")
        os.makedirs(backup_dir, exist_ok=True)

        # Save the filtered images to the backup directory
        img_type = "image" if "images" in dir else "mask"
        self.save_subset_array(
            folder_path=backup_dir, image_array=image_array, type=img_type
        )

        # Now that we have a backup, it's safe to remove all files from the original directory
        self.remove_files_in_dir(folder_path=dir)

        # Move the files from the backup to the original directory
        for file in os.listdir(backup_dir):
            src = os.path.join(backup_dir, file)
            dst = os.path.join(dir, file)
            shutil.move(src, dst)

        # Remove the backup directory for this directory
        shutil.rmtree(backup_dir)
