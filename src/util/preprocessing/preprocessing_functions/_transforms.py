import numpy as np
import cv2
from typing import List


def crop_raw_images(self, image_array: List[np.ndarray]):
    """
    crops the raw images by doing a circle crop. These images are part of the set that are just
    pictures of the tumor

    Parameters
    ----------
    image_array : List[np.ndarray]
        list of images from the raw picture dataset (unsegmented)

    Return
    ------
    cropped_images : List[np.ndarray]
        returns the images after they have been cropped
    """

    cropped_images = []

    for i in range(len(image_array)):

        image = image_array[i]

        mask = np.zeros(image.shape, dtype=np.uint8)
        mask = cv2.circle(mask, (320, 240), 180, (255, 255, 255), -1)

        res = cv2.bitwise_and(image, mask)
        res[mask == 0] = 255

        cropped_images.append(res)

    return cropped_images


def crop_masks(self, image_array: List[np.ndarray]):
    """
    before any mask processing is done, the images with the
    red segmented region have a circle crop

    Parameters
    ----------
    image_array : List[np.ndarray]
        these are the "masks" before any processing has been done

    Return
    ------
    cropped_images : List[np.ndarray]
        returns the images after they have been cropped
    """

    cropped_images = []

    for i in range(len(image_array)):
        image = image_array[i]

        mask = np.zeros(image.shape, dtype=np.uint8)
        mask = cv2.circle(mask, (288, 307), 200, (255, 255, 255), -1)

        res = cv2.bitwise_and(image, mask)
        res[mask == 0] = 255

        cropped_images.append(res)

    return cropped_images


def add_padding(
    self, image_array: List[np.ndarray], mask_array: List[np.ndarray]
):
    """
    the goal for our preprocessing code is to make sure that the masks and raw image
    line up exactly on top of each other, so adding padding to both images will make
    them the same size.

    because the size of the image really matters here, we can only support specific
    size images.

    See Also
    --------
    Our pytests for supported sizes shows what image sizes we support

    Parameters
    ----------
    image_array : List[np.ndarray]
        this is the array of images without the segmentation
    mask_array : List[np.ndarray]
        at this point we have not chroma keyed them, but these are the images for the
        ground truth

    Return
    ------
    padded_images, padded_masks : List[np.ndarray]
        both the image and mask array are changed
    """

    padded_images = []
    padded_masks = []

    for i in range(len(image_array)):

        image = image_array[i]
        mask = mask_array[i]  # Assuming mask_array is a list of masks

        # Check mask dimensions using len()
        if len(mask[0]) == 492:  # Check the number of columns (width)
            # MC_data
            padded_image = cv2.copyMakeBorder(
                image, 7, 7, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255)
            )

            padded_mask = cv2.copyMakeBorder(
                mask, 0, 0, 74, 74, cv2.BORDER_CONSTANT, value=(255, 255, 255)
            )

            padded_images.append(padded_image)
            padded_masks.append(padded_mask)

        elif len(mask[0]) == 577:  # Check the number of columns (width)
            # invotive data
            padded_image = cv2.copyMakeBorder(
                image, 67, 67, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255)
            )

            padded_mask = cv2.copyMakeBorder(
                mask, 0, 0, 31, 31, cv2.BORDER_CONSTANT, value=(255, 255, 255)
            )

            padded_images.append(padded_image)
            padded_masks.append(padded_mask)

        else:
            print(f"Error: Mask dimensions {len(mask)} not recognized")

    return padded_images, padded_masks


def zoom_at(
    self, image_array: List[np.ndarray], zoom: float, coord: float = None
) -> List[np.ndarray]:
    """
    takes the image and "zooms" in at a specific coords. We use this to make sure that
    the "circles" of the two images are the same size. This is key in making sure that
    the tumors lay on top of each other

    Parameters
    ----------
    image_array : List[np.ndarray]
        list of images that we want to zoom
    zoom : float
        this is the factor by which we are applying the zoom
    coord : float
        this is where we are applying the zoom, by default it is in the center

    Return
    ------
    zoomed_array : List[np.ndarray]
        image array after the zooming process is done

    """

    zoomed_array = []

    for img in image_array:

        h, w, _ = [zoom * i for i in img.shape]

        if coord is None:
            cx, cy = w / 2, h / 2
        else:
            cx, cy = [zoom * c for c in coord]

        img = cv2.resize(img, (0, 0), fx=zoom, fy=zoom)
        img = img[
            int(round(cy - h / zoom * 0.5)) : int(round(cy + h / zoom * 0.5)),
            int(round(cx - w / zoom * 0.5)) : int(round(cx + w / zoom * 0.5)),
            :,
        ]
        zoomed_array.append(img)

    return zoomed_array


def crop_images(self, image_array: List[np.ndarray]) -> List[np.ndarray]:
    """
    crops images to a standard 256 by 256 in the center for training, this
    is done for both the images and the masks

    Parameters
    ----------
    image_array : List[np.ndarray]
        image array to crop

    Return
    ------
    cropped_images : List[np.ndarray]
        cropped set of images from the param array
    """

    cropped_images = []

    for i in range(len(image_array)):

        image = image_array[i]

        image_height, image_width = image.shape[:2]

        # Bounding box dimensions
        box_width, box_height = 256, 256

        x_top_left = (image_width - box_width) // 2
        y_top_left = (image_height - box_height) // 2
        x_bottom_right = x_top_left + box_width
        y_bottom_right = y_top_left + box_height

        cropped_image = image[y_top_left:y_bottom_right, x_top_left:x_bottom_right]
        cropped_images.append(cropped_image)

    return cropped_images


def crop_images_offset(
    self, image_array: List[np.ndarray], x_offset: float = 0, y_offset: float = 0
) -> List[np.ndarray]:
    """
    this is a utility function that is used in the case that some formats of images
    could need to be offset in order to line up. Takes in x and y values to determine
    how much to offset

    It also applies a crop to the image at the same time. Unclear why this is a
    separate method.

    Parameters
    ------
    image_array : List[np.ndarray]
        image array to offset
    x_offset : float
        x value to offset the image horizontally
    y_offset : float
        y value to offset the image vertically

    Returns
    -------
    cropped_images : List[np.ndarray]
        List of images after they have been cropped and offset
    """

    cropped_images = []

    for image in image_array:
        image_height, image_width = image.shape[:2]

        # Bounding box dimensions
        box_width, box_height = 256, 256

        # Ensure the image is large enough to crop
        if image_width < box_width or image_height < box_height:
            print(
                f"Skipping image with dimensions {image_width}x{image_height}, too small for cropping."
            )
            continue

        # Calculate the top-left corner with offsets
        x_top_left = (image_width - box_width) // 2 + x_offset
        y_top_left = (image_height - box_height) // 2 + y_offset

        # Ensure the crop doesn't go out of bounds
        x_top_left = max(0, min(x_top_left, image_width - box_width))
        y_top_left = max(0, min(y_top_left, image_height - box_height))

        # Calculate bottom-right coordinates
        x_bottom_right = x_top_left + box_width
        y_bottom_right = y_top_left + box_height

        # Crop the image
        cropped_image = image[y_top_left:y_bottom_right, x_top_left:x_bottom_right]
        cropped_images.append(cropped_image)

    return cropped_images


def translate_images(
    self, images: List[np.ndarray], x_offset: float, y_offset: float = 0
):
    """
    this is a utility function that is used in the case that some formats of images
    could need to be offset in order to line up. Takes in x and y values to determine
    how much to offset

    Parameters
    ------
    image_array : List[np.ndarray]
        image array to offset
    x_offset : float
        x value to offset the image horizontally
    y_offset : float
        y value to offset the image vertically
    """

    translated_images = []

    for img_np in images:
        height, width, channels = img_np.shape

        translated_img_np = np.ones((height, width, channels), dtype=np.uint8) * 255

        x_start = max(0, x_offset)
        x_end = min(width, width + x_offset)
        y_start = max(0, y_offset)
        y_end = min(height, height + y_offset)

        src_x_start = max(0, -x_offset)
        src_x_end = width - max(0, x_offset)
        src_y_start = max(0, -y_offset)
        src_y_end = height - max(0, y_offset)

        translated_img_np[y_start:y_end, x_start:x_end] = img_np[
            src_y_start:src_y_end, src_x_start:src_x_end
        ]

        translated_images.append(translated_img_np)

    return translated_images
