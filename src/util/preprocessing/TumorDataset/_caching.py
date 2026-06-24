import os
import cv2
from typing import List, Tuple
import numpy as np


class CachingMixin:

    # this will load data from the cache
    # returns train_images for each type (rgb, depth, rgd) and shared masks
    def load_data(
        self,
        rgb: bool = False,
        depth: bool = False,
        rgd: bool = False,
    ) -> Tuple[
        List[np.ndarray],
        List[np.ndarray],
        List[np.ndarray],
        List[np.ndarray],
        List[np.ndarray],
        List[np.ndarray],
    ]:
        if rgb:
            rgb_root = os.path.join(self.processed_root, "rgb")
            return self.read_to_array_post(rgb_root)
        if depth:
            depth_root = os.path.join(self.processed_root, "depth")
            return self.read_to_array_post(depth_root)
        if rgd:
            rgd_root = os.path.join(self.processed_root, "rgd")
            return self.read_to_array_post(rgd_root)

    # this is only for inline tests (only used for rgb)
    def return_data(self):
        return self.images_rgb, self.masks

    @staticmethod
    def _contour_to_depth_channel(contour_img):
        """Contour render -> single-channel depth, matching RGD's blue-channel
        convention (inverted + per-image min-max), as float32."""
        gray = (
            cv2.cvtColor(contour_img, cv2.COLOR_BGR2GRAY)
            if getattr(contour_img, "ndim", 2) == 3
            else contour_img
        )
        norm = cv2.normalize(gray.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX)
        return (255.0 - norm).astype(np.float32)

    def _cache_rgbd_variant(self, variant, splits) -> None:
        """Persist an RGBD variant as paired files per sample:
           images/<stem>.jpg, images/<stem>_depth.npy, masks/Tumor/<stem>.png.

        RGB is saved as .jpg so it matches the COCO file_name (process_coco_json
        ORIGINAL_EXT='jpg') and the rgb baseline; the depth channel is a sibling
        .npy that the future 4-channel mapper stacks on to form (H, W, 4).

        splits: {split_name: (rgb_imgs, depth_arrays, masks, filenames)}.
        """
        for split, (rgb_imgs, depth_arrays, masks, filenames) in splits.items():
            img_dir = os.path.join(self.processed_root, variant, split, "images")
            mask_dir = os.path.join(self.processed_root, variant, split, "masks/Tumor")
            os.makedirs(img_dir, exist_ok=True)
            os.makedirs(mask_dir, exist_ok=True)

            for rgb, depth, name in zip(rgb_imgs, depth_arrays, filenames):
                stem = os.path.splitext(name)[0]
                cv2.imwrite(os.path.join(img_dir, f"{stem}.jpg"), rgb)
                np.save(
                    os.path.join(img_dir, f"{stem}_depth.npy"),
                    np.asarray(depth, dtype=np.float32),
                )

            for mask, name in zip(masks, filenames):
                stem = os.path.splitext(name)[0]
                cv2.imwrite(os.path.join(mask_dir, f"{stem}.png"), mask)

    # this will save our data into directories for each image type, with shared masks
    def cache_data(self) -> None:
        # Create RGB image and mask directories
        train_img_dir_rgb = os.path.join(self.processed_root, "rgb/train/images")
        val_img_dir_rgb = os.path.join(self.processed_root, "rgb/val/images")
        test_img_dir_rgb = os.path.join(self.processed_root, "rgb/test/images")

        train_mask_dir_rgb = os.path.join(self.processed_root, "rgb/train/masks/Tumor")
        val_mask_dir_rgb = os.path.join(self.processed_root, "rgb/val/masks/Tumor")
        test_mask_dir_rgb = os.path.join(self.processed_root, "rgb/test/masks/Tumor")

        # Create Depth image and mask directories
        train_img_dir_depth = os.path.join(self.processed_root, "depth/train/images")
        val_img_dir_depth = os.path.join(self.processed_root, "depth/val/images")
        test_img_dir_depth = os.path.join(self.processed_root, "depth/test/images")

        train_mask_dir_depth = os.path.join(
            self.processed_root, "depth/train/masks/Tumor"
        )
        val_mask_dir_depth = os.path.join(self.processed_root, "depth/val/masks/Tumor")
        test_mask_dir_depth = os.path.join(
            self.processed_root, "depth/test/masks/Tumor"
        )

        # Create RGD image and mask directories
        train_img_dir_rgd = os.path.join(self.processed_root, "rgd/train/images")
        val_img_dir_rgd = os.path.join(self.processed_root, "rgd/val/images")
        test_img_dir_rgd = os.path.join(self.processed_root, "rgd/test/images")

        train_mask_dir_rgd = os.path.join(self.processed_root, "rgd/train/masks/Tumor")
        val_mask_dir_rgd = os.path.join(self.processed_root, "rgd/val/masks/Tumor")
        test_mask_dir_rgd = os.path.join(self.processed_root, "rgd/test/masks/Tumor")

        # Create all directories
        os.makedirs(train_img_dir_rgb, exist_ok=True)
        os.makedirs(val_img_dir_rgb, exist_ok=True)
        os.makedirs(test_img_dir_rgb, exist_ok=True)
        os.makedirs(train_mask_dir_rgb, exist_ok=True)
        os.makedirs(val_mask_dir_rgb, exist_ok=True)
        os.makedirs(test_mask_dir_rgb, exist_ok=True)

        os.makedirs(train_img_dir_depth, exist_ok=True)
        os.makedirs(val_img_dir_depth, exist_ok=True)
        os.makedirs(test_img_dir_depth, exist_ok=True)
        os.makedirs(train_mask_dir_depth, exist_ok=True)
        os.makedirs(val_mask_dir_depth, exist_ok=True)
        os.makedirs(test_mask_dir_depth, exist_ok=True)

        os.makedirs(train_img_dir_rgd, exist_ok=True)
        os.makedirs(val_img_dir_rgd, exist_ok=True)
        os.makedirs(test_img_dir_rgd, exist_ok=True)
        os.makedirs(train_mask_dir_rgd, exist_ok=True)
        os.makedirs(val_mask_dir_rgd, exist_ok=True)
        os.makedirs(test_mask_dir_rgd, exist_ok=True)

        # Save RGB images to cache directories

        for img, name in zip(self.train_images_rgb, self.train_filenames):
            cv2.imwrite(os.path.join(train_img_dir_rgb, name), img)

        for img, name in zip(self.val_images_rgb, self.val_filenames):
            cv2.imwrite(os.path.join(val_img_dir_rgb, name), img)

        for img, name in zip(self.test_images_rgb, self.test_filenames):
            cv2.imwrite(os.path.join(test_img_dir_rgb, name), img)

        # Save Depth images to cache directories (if they exist)
        if hasattr(self, "train_images_depth") and self.train_images_depth:
            for img, name in zip(self.train_images_depth, self.train_filenames):
                cv2.imwrite(os.path.join(train_img_dir_depth, name), img)

            for img, name in zip(self.val_images_depth, self.val_filenames):
                cv2.imwrite(os.path.join(val_img_dir_depth, name), img)

            for img, name in zip(self.test_images_depth, self.test_filenames):
                cv2.imwrite(os.path.join(test_img_dir_depth, name), img)

        # Save RGD images to cache directories (if they exist)
        if hasattr(self, "train_images_rgd") and self.train_images_rgd:
            for img, name in zip(self.train_images_rgd, self.train_filenames):
                cv2.imwrite(os.path.join(train_img_dir_rgd, name), img)

            for img, name in zip(self.val_images_rgd, self.val_filenames):
                cv2.imwrite(os.path.join(val_img_dir_rgd, name), img)

            for img, name in zip(self.test_images_rgd, self.test_filenames):
                cv2.imwrite(os.path.join(test_img_dir_rgd, name), img)

        # Save masks to all cache directories (same masks for all image types)
        # RGB masks
        for mask, name in zip(self.train_masks, self.train_filenames):
            clean_name = os.path.splitext(name)[0] + ".png"
            cv2.imwrite(os.path.join(train_mask_dir_rgb, clean_name), mask)

        for mask, name in zip(self.val_masks, self.val_filenames):
            clean_name = os.path.splitext(name)[0] + ".png"
            cv2.imwrite(os.path.join(val_mask_dir_rgb, clean_name), mask)

        for mask, name in zip(self.test_masks, self.test_filenames):
            clean_name = os.path.splitext(name)[0] + ".png"
            cv2.imwrite(os.path.join(test_mask_dir_rgb, clean_name), mask)

        # Depth masks (same as RGB masks)
        for mask, name in zip(self.train_masks, self.train_filenames):
            clean_name = os.path.splitext(name)[0] + ".png"
            cv2.imwrite(os.path.join(train_mask_dir_depth, clean_name), mask)

        for mask, name in zip(self.val_masks, self.val_filenames):
            clean_name = os.path.splitext(name)[0] + ".png"
            cv2.imwrite(os.path.join(val_mask_dir_depth, clean_name), mask)

        for mask, name in zip(self.test_masks, self.test_filenames):
            clean_name = os.path.splitext(name)[0] + ".png"
            cv2.imwrite(os.path.join(test_mask_dir_depth, clean_name), mask)

        # RGD masks (same as RGB masks)
        for mask, name in zip(self.train_masks, self.train_filenames):
            clean_name = os.path.splitext(name)[0] + ".png"
            cv2.imwrite(os.path.join(train_mask_dir_rgd, clean_name), mask)

        for mask, name in zip(self.val_masks, self.val_filenames):
            clean_name = os.path.splitext(name)[0] + ".png"
            cv2.imwrite(os.path.join(val_mask_dir_rgd, clean_name), mask)

        for mask, name in zip(self.test_masks, self.test_filenames):
            clean_name = os.path.splitext(name)[0] + ".png"
            cv2.imwrite(os.path.join(test_mask_dir_rgd, clean_name), mask)

        # ---- RGBD early-fusion variant (paired RGB .jpg + depth .npy) ----
        # rgbd_early: depth = inverted-normalized grayscale of the contour render
        if hasattr(self, "train_images_rgbd_early") and self.train_images_rgbd_early:
            self._cache_rgbd_variant(
                "rgbd_early",
                {
                    "train": (
                        self.train_images_rgbd_rgb,
                        [
                            self._contour_to_depth_channel(d)
                            for d in self.train_images_rgbd_early
                        ],
                        self.train_masks,
                        self.train_filenames,
                    ),
                    "val": (
                        self.val_images_rgbd_rgb,
                        [
                            self._contour_to_depth_channel(d)
                            for d in self.val_images_rgbd_early
                        ],
                        self.val_masks,
                        self.val_filenames,
                    ),
                    "test": (
                        self.test_images_rgbd_rgb,
                        [
                            self._contour_to_depth_channel(d)
                            for d in self.test_images_rgbd_early
                        ],
                        self.test_masks,
                        self.test_filenames,
                    ),
                },
            )

        print("Data cached successfully.")
