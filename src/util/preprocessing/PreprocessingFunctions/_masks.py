import numpy as np
import cv2
from typing import List


def create_binary_masks(self, image_array: List[np.ndarray]) -> List[np.ndarray]:
    """
    creates binary masks of the dataset to be used to create the COCO JSON annotations.
    We did a simple chroma key of the red region of the pre segmented images.

    Parameters
    ----------
    image_array : List[np.ndarray]
        array of masks

    Return
    ------
    binary_masks : List[np.ndarray]
        these are binary masks created

    """

    binary_masks = []

    for image in image_array:
        if image.ndim == 2:
            image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] != 3:
            raise ValueError("Input image must have 3 channels (BGR format).")
        else:
            image_color = image

        hsv = cv2.cvtColor(image_color, cv2.COLOR_BGR2HSV)

        lower_red1 = np.array([0, 150, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 150, 50])
        upper_red2 = np.array([180, 255, 255])

        # Create masks for the red color
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

        # Combine the two masks
        mask = cv2.bitwise_or(mask1, mask2)

        binary_masks.append(mask)

    return binary_masks


def create_neg_masks(self, length: float) -> List[np.ndarray]:
    """
    this function creates a mask for the negative images, whcih
    is just an array of 0s (since the image does not have a tumor)

    Parameters
    ----------
    length : float
        the amount of masks we need for negative images

    Returns
    -------
        negative_masks : List[np.ndarray]
            Essentially an array of 0s that represent an empty image

    Note
    ----
    We don't currently use this feature, so the size of the mask is
    inaccurate here and must be added as a parameter if this function
    is used
    """

    negative_masks = []
    for i in range(length):
        negative_mask = np.ones((495, 492), dtype=np.uint8) * 0
        negative_masks.append(negative_mask)

    return negative_masks


# this is to correct the chroma key error with the previous function
def correct_binary_masks(self, mask_array: List[np.ndarray]) -> List[np.ndarray]:
    fixed_images = []
    for i, img in enumerate(mask_array):
        # apprently they are saved as 3 channel color images 
        binary = cv2.cvtColor(img, cv2.COLOR_BAYER_BG2GRAY)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        binary = binary.astype(np.uint8)

        filled = np.zeros_like(binary)
        cv2.fillPoly(filled, contours, 255)

        fixed_images.append(filled)
    
    return fixed_images
