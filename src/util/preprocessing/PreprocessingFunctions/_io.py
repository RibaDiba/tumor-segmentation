import json
import numpy as np
import cv2, os
from scipy.interpolate import griddata
from tqdm import tqdm
from typing import Tuple, List


def read_images_to_array(
    self, folder_path: str, read_bins: bool = True
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[str]]:
    """
    This assumes a clean dataset. First reads the directory and then returns sorted
    images into 3 categories ->

    Base images
    Segmented images from the scanner
    Binary file information for depth images

    Parameters
    ----------
    folder_path : str
        this is the folder path that will contain all the processed data
    read_bins : boolean
        if we want to get the depth information for our models, then this
        should be set to true. The reason this is a boolean because reading
        the point cloud information takes a lot of computational overhead

    Returns
    -------
    segmented_images : List[np.ndarray]
        this is an array of images of the mice that have been segmented with
        the scanner
    base_images : List[np.ndarray]
        this is an array of raw images of the tumor with no scanner involved
    depth_info : List[np.ndarray]
        this currently has the point cloud information from the binary file
    valid_filenames : List[str]
        list of strings that preserve the filenames of all the tumors, which
        was mainly helpful for the tumor flagging webapp
    """

    segmented_images = []
    base_images = []
    depth_info = []
    # take the filenames to save later
    valid_filenames = []

    # load the depth cache index if it exists, so .bin reads can be skipped in O(1)
    cache_dir = os.path.join(folder_path, "depth_cache")
    cache_index_path = os.path.join(cache_dir, "depth_cache.json")
    cache_index = {}
    if os.path.exists(cache_index_path):
        with open(cache_index_path, "r") as f:
            cache_index = json.load(f)
    cache_updated = False

    filenames = sorted(os.listdir(folder_path))
    for filename in tqdm(filenames, desc="Reading files"):
        full_path = os.path.join(folder_path, filename)

        if filename.endswith(".jpg") and not filename.endswith("_texture.jpg"):
            img = cv2.imread(full_path)
            if img is not None:
                segmented_images.append(img)
                valid_filenames.append(filename)

        elif filename.endswith("_texture.jpg"):
            img = cv2.imread(full_path)
            if img is not None:
                base_images.append(img)

        if read_bins and filename.endswith(".bin"):
            try:
                if filename in cache_index:
                    npy_path = os.path.join(cache_dir, cache_index[filename])
                    data = np.load(npy_path)
                    grid_x, grid_y, grid_z = data[0], data[1], data[2]
                else:
                    grid_x, grid_y, grid_z = read_bin(full_path)
                    os.makedirs(cache_dir, exist_ok=True)
                    stem = os.path.splitext(filename)[0]
                    npy_name = f"{stem}.npy"
                    np.save(os.path.join(cache_dir, npy_name), np.stack([grid_x, grid_y, grid_z]))
                    cache_index[filename] = npy_name
                    cache_updated = True
                depth_info.append((grid_x, grid_y, grid_z, filename))
            except Exception as e:
                print(f"Failed to read binary file {filename}: {e}")

    if cache_updated:
        with open(cache_index_path, "w") as f:
            json.dump(cache_index, f, indent=2)

    return segmented_images, base_images, depth_info, valid_filenames


def read_neg_images(self, folder_path: str) -> List[np.ndarray]:
    """
    this reads a path that specifically reads images that have "no tumor"
    we are not sure that these images don't have tumors, but we made
    that assumption when filtering our data

    Parameters
    ----------
    folder_path : str
        path where all the negative images are stored, this was done
        because the negative images were stored seperate

    Returns
    -------
    neg_images : List[np.ndarray]
        an array of images that do not contain a tumor

    """

    filenames = sorted(os.listdir(folder_path))
    neg_images = []
    for filename in tqdm(filenames, desc="Reading negative images"):
        full_path = os.path.join(folder_path, filename)
        img = cv2.imread(full_path)
        neg_images.append(img)

    return neg_images


def read_bin(file_path):
    """
    opens a binary file in a specific path and extracts the point cloud
    information from that. Returns lists of x, y, and z values

    Parameters
    ----------
    file_path : str
        file path to binary file

    Returns
    -------
    grid_x, grid_y, grid_z
        lists of floats that represent the point cloud of the tumor
    """

    with open(file_path, "rb") as fid:
        data = np.fromfile(fid, dtype=">f8")

    points = data.reshape(-1, 3)

    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    grid_x, grid_y = np.meshgrid(
        np.linspace(min(x), max(x), 256), np.linspace(min(y), max(y), 256)
    )

    grid_z = griddata((x, y), z, (grid_x, grid_y), method="linear")

    return grid_x, grid_y, grid_z


def read_folder_to_array(self, folder_path: str) -> List[np.ndarray]:
    """
    This function reads a folder and returns an array of images that is
    in the folder. This is a helper function that is used to load the data
    that has been stored and processed in the /processed_data directory

    Parameters:
        folder_path (str): directory location of where you want to pull
        from

    Returns:
        image_array: array of images from that directory, ignores other
        files
    """
    image_array = []

    exts = [".jpg", ".png"]
    files = [
        f
        for f in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, f))
        and os.path.splitext(f)[1].lower() in exts
    ]

    # sort the files in the same way
    files.sort()

    for f in files:
        full_path = os.path.join(folder_path, f)
        img = cv2.imread(full_path)
        if img is None:
            print("error with file read")
            continue
        image_array.append(img)

    return image_array


def read_to_array_post(self, root_path: str) -> List[np.ndarray]:
    """
    This function reads all of the subdirectories in the processed_data
    directory and returns their images.

    Parameters:
        root_path (str): processed_data directory
    Returns:
        returns the data for each split
    """

    train_images = self.read_folder_to_array(
        folder_path=os.path.join(root_path, "train/images")
    )
    train_masks = self.read_folder_to_array(
        folder_path=os.path.join(root_path, "train/masks/Tumor")
    )

    val_images = self.read_folder_to_array(
        folder_path=os.path.join(root_path, "val/images")
    )
    val_masks = self.read_folder_to_array(
        folder_path=os.path.join(root_path, "val/masks/Tumor")
    )

    test_images = self.read_folder_to_array(
        folder_path=os.path.join(root_path, "test/images")
    )
    test_masks = self.read_folder_to_array(
        folder_path=os.path.join(root_path, "test/masks/Tumor")
    )

    return train_images, train_masks, val_images, val_masks, test_images, test_masks
