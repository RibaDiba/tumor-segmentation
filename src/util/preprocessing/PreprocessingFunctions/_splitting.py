import numpy as np
from typing import List


def split_train_val_test(
    images: List[np.ndarray],
    masks: List[np.ndarray],
    per_train: int,
    per_val: int,
    per_test: int,
):
    """
    takes images and masks and splits them into training, validation, and tests

    Parameters
    ----------
    images : List[np.ndarray]
        at this point, this should be the annotated images for the tumor
    masks : List[np.ndarray]
        the masks that correspond to each image
    per_train, per_val, per_test : int
        these are the percentages that we want to split our data amoung

    Returns
    -------
    train_images, val_images, test_images : List[np.ndarray]
        image arrays that correspond to train, val, test
    train_masks, val_masks, test_masks
        mask arrays that correspond to train, val, test

    """

    # returns error if they don't add up
    assert (
        per_train + per_val + per_test
    ) == 100, "The percentages must sum up to 100."

    train_num = int(len(images) * (per_train / 100))
    val_num = int(len(images) * (per_val / 100))
    test_num = len(images) - train_num - val_num

    indices = np.arange(len(images))

    train_indices = indices[:train_num]
    val_indices = indices[train_num : train_num + val_num]
    test_indices = indices[train_num + val_num :]

    train_images = [images[i] for i in train_indices]
    train_masks = [masks[i] for i in train_indices]
    val_images = [images[i] for i in val_indices]
    val_masks = [masks[i] for i in val_indices]
    test_images = [images[i] for i in test_indices]
    test_masks = [masks[i] for i in test_indices]

    return train_images, train_masks, val_images, val_masks, test_images, test_masks
