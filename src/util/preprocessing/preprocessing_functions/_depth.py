import matplotlib.pyplot as plt
import numpy as np
import cv2, io, os
from tqdm import tqdm
from PIL import Image
from typing import List


def read_contours_with_grid(self, data_array):
    """
    Returns the raw 256x256 interpolated height grid for each sample (the
    "rawgrid" depth source), WITHOUT rendering through matplotlib/contourf.

    Lossless-render counterpart to read_contours_array_depth: both consume the
    same depth_info tuples from read_images_to_array, each element being
    (grid_x, grid_y, grid_z, filename) where grid_z is the 256x256
    griddata(method="linear") interpolation computed in read_bin (_io.py). The
    contour path screenshots a contourf plot (100-level quantization + colormap
    round-trip); this path returns the height grid directly, avoiding that loss.

    NOTE on alignment: the returned array is in grid-index space (256x256), NOT
    the ~640x480 camera frame that crop_raw_images/add_padding/crop_images were
    tuned for, so it is not yet pixel-aligned with the RGB image. Alignment is
    handled separately for the rgbd_rawgrid modality.

    Parameters
    ----------
    data_array : List[Tuple[np.ndarray, np.ndarray, np.ndarray, str]]
        depth_info tuples (grid_x, grid_y, grid_z, filename). Only grid_z is
        used; grid_x/grid_y/filename are kept for parity with the contour path.

    Returns
    -------
    grid_array : List[np.ndarray]
        One float32 array of shape (256, 256) per sample. NaNs outside the
        convex hull are filled with the minimum finite height, then values are
        inverted and min-max normalized to [0, 255] to match the RGD/contour
        convention (255 - normalize), so contour vs rawgrid differ only by the
        rendering step.
    """

    grid_array = []

    for data in tqdm(data_array, desc="Building raw depth grids"):
        _, _, z, _ = data
        z = np.asarray(z, dtype=np.float32)

        # A2: griddata(linear) leaves NaN outside the convex hull of the points.
        # contourf hides this; the raw grid does not. Fill with the minimum
        # finite height so background sits at the "far" end of the range.
        if np.isnan(z).any():
            finite = z[np.isfinite(z)]
            fill = float(finite.min()) if finite.size else 0.0
            z = np.nan_to_num(z, nan=fill)

        # A3: match RGD/contour normalization (invert + per-image min-max) so the
        # ONLY difference between contour and rawgrid is the render step.
        z_norm = cv2.normalize(z, None, 0, 255, cv2.NORM_MINMAX)
        z_norm = 255.0 - z_norm

        grid_array.append(z_norm.astype(np.float32))

    return grid_array

def read_contours_array_depth(self, data_array):
    """
    takes the point cloud information from the data_array and generates
    contour plots that represent the point cloud.

    After generating contour plots in the "Grays" color, we take a picture
    of them from a topview angle and use that as our "depth" image

    Parameters
    ----------
    data_array : Tuple
        point cloud information to be turned into contour plots

    Returns
    -------
    image_array : List[np.ndarray]
        image array of contour plots
    """

    image_array = []

    for data in tqdm(data_array, desc="Saving Contour Plots"):
        x, y, z, original_filename = data
        base_file_name = os.path.splitext(original_filename)[0]
        file_name = f"{base_file_name}.png"

        plt.contourf(x, y, z, levels=100, cmap="Grays")
        plt.gca().set_aspect("equal")
        plt.axis("off")

        buf = io.BytesIO()
        plt.savefig(buf, format="jpg")
        buf.seek(0)

        image = Image.open(buf)
        image = np.array(image)
        image_array.append(image)

        buf.close()
        plt.close()

    return image_array


def infuse_depth_into_blue_channel(
    self, image_array: List[np.ndarray], depth_array: List[np.ndarray]
) -> List[np.ndarray]:
    """
    Infuses depth map information into the blue channel of BGR images
    by fully replacing the blue channel with the normalized depth map.

    Parameters:
        image_array (List[np.ndarray]): List of BGR images (each shape: H x W x 3)
        depth_array (List[np.ndarray]): List of grayscale or BGR depth maps (each shape: H x W or H x W x 3)

    Returns:
        List[np.ndarray]: List of BGR images with depth fully infused into the blue channel
    """
    if len(image_array) != len(depth_array):
        raise ValueError("image_array and depth_array must have the same length")

    image_array_infused = []

    for i in tqdm(range(len(image_array)), desc="Infusing Images"):
        image = image_array[i]
        depth_map = depth_array[i]

        # Validate image and depth map
        if image is None or depth_map is None:
            raise ValueError(f"Missing image or depth map at index {i}")

        if len(image.shape) != 3 or image.shape[2] != 3:
            raise ValueError(f"Image at index {i} is not 3-channel (BGR)")

        # Resize depth map to match image dimensions
        depth_map_resized = cv2.resize(depth_map, (image.shape[1], image.shape[0]))

        # Convert depth map to grayscale if it's BGR
        if len(depth_map_resized.shape) == 3 and depth_map_resized.shape[2] == 3:
            depth_map_resized = cv2.cvtColor(depth_map_resized, cv2.COLOR_BGR2GRAY)

        # Normalize and invert depth map to 0-255 (uint8)
        depth_map_normalized = 255 - cv2.normalize(
            depth_map_resized, None, 0, 255, cv2.NORM_MINMAX
        ).astype(np.uint8)

        # Split image channels
        _, g, r = cv2.split(image)

        # Replace the blue channel with the depth map
        infused_blue = depth_map_normalized

        # Merge back the channels
        infused_image = cv2.merge((infused_blue, g, r))

        image_array_infused.append(infused_image)

    return image_array_infused
