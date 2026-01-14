import pytest, os, cv2, sys
import numpy as np
from pathlib import Path

parent_dir = str(Path(__file__).parent.parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

PROJECT_ROOT = Path(parent_dir).parent
util_dir = str(PROJECT_ROOT / "util")
if util_dir not in sys.path:
    sys.path.append(util_dir)

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
    if len(mask_image.shape) == 2 or (
        len(mask_image.shape) == 3 and mask_image.shape[2] == 1
    ):
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


@pytest.mark.parametrize(
    "dir",
    [
        (str(PROJECT_ROOT / "data/huggingface-repo/useable_data")),
    ],
)
def test_annotations(dir):
    mask_fnames: List[str] = []
    masks: List[np.ndarray] = []
    image_fnames: List[str] = []
    images: List[np.ndarray] = []

    # first get images/masks and their filenames
    for filename in os.listdir(dir):
        basename, ext = os.path.splitext(filename)
        if ext.lower() != ".jpg":
            continue

        img = cv2.imread(os.path.join(dir, filename))
        assert img is not None, f"{filename}: failed to load"

        if basename.endswith("_texture"):
            image_fnames.append(basename)
            images.append(img)
        else:
            mask_fnames.append(basename)
            masks.append(img)

    d = Dataset(dir)

    # manually apply preprocessing code
    masks = d.crop_masks(masks)
    images, masks = d.add_padding(images, masks)
    masks = d.zoom_at(masks, 1.333, coord=None)
    masks = d.create_binary_masks(masks)
    masks = d.crop_images_offset(masks, x_offset=-25)
    masks = d.correct_binary_masks(masks)

    # check annotations for those masks
    for fname, mask in zip(mask_fnames, masks):
        count = count_mask_augmentations(mask)
        assert count == 1, f"{fname}: expected 1 annotation, found {count}"
