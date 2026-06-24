"""
creating custom augmentations
"""

import os, cv2
from itertools import product
from typing import List, Tuple
import numpy as np
import albumentations as A

class AugmentationClass:

    # canonical order of the image sets carried through every transform; the two
    # rgbd_* sets must stay in sync with each other (and the mask) so the 4-channel
    # stack built downstream remains valid
    #   [rgb, depth, rgd, rgbd_rgb, rgbd_early]

    def __init__(
            self,
            tumor_dataset,
            rotate_degrees: int,
            test_only: bool = False # TODO: implement this feature
    ):
        """
        Stores a reference to the tumor dataset structure, accesses its masks and images
        directly in order to augment the images to add more to our data

        Args:
            tumor_dataset: reference to the tumor_dataset class
            test_only: TODO, but this is to indicate if only the testing dataset should
            be altered
        """

        self.dataset = tumor_dataset
        self.test_only = test_only
        self.rotate_degrees = rotate_degrees

        # some defined constants
        self.dataset_size = len(self.dataset.images_rgb)

    def augment_images(self) -> None:

        """
        we apply the following transformations to the dataset. Note that each transformation
        is applied and all possible combinations are applied to produce the most amount of
        unqiue images

        Vertical Flips
        Horizontal Flips
        Randomized Rotation

        """

        self.new_rgb, self.new_depth, self.new_rgd = [], [], []
        self.new_rgbd_rgb, self.new_rgbd_early = [], []
        self.new_masks = []
        self.new_filenames = []

        for idx in range(self.dataset_size):
            # canonical order: rgb, depth, rgd, rgbd_rgb, rgbd_early
            images = [
                self.dataset.images_rgb[idx].copy(),
                self.dataset.images_depth_maps[idx].copy(),
                self.dataset.images_rgd[idx].copy(),
                self.dataset.images_rgbd_rgb[idx].copy(),
                self.dataset.images_rgbd_early[idx].copy(),
            ]
            mask        = self.dataset.masks[idx].copy()
            original_name = self.dataset.filenames[idx]
            base, ext = os.path.splitext(original_name)

            # populates the new arrays
            self._return_combinations(images, mask, base, ext)

        # merges all datasets togther
        self.combined_rgb         = self.dataset.images_rgb         + self.new_rgb
        self.combined_depth       = self.dataset.images_depth_maps  + self.new_depth
        self.combined_rgd         = self.dataset.images_rgd         + self.new_rgd
        self.combined_rgbd_rgb    = self.dataset.images_rgbd_rgb    + self.new_rgbd_rgb
        self.combined_rgbd_early  = self.dataset.images_rgbd_early  + self.new_rgbd_early
        self.combined_masks       = self.dataset.masks              + self.new_masks
        self.combined_filenames   = self.dataset.filenames          + self.new_filenames

        # shuffle all parallel arrays with a single permutation to guarantee sync
        perm = np.random.permutation(len(self.combined_rgb))
        self.combined_rgb         = [self.combined_rgb[p]         for p in perm]
        self.combined_depth       = [self.combined_depth[p]       for p in perm]
        self.combined_rgd         = [self.combined_rgd[p]         for p in perm]
        self.combined_rgbd_rgb    = [self.combined_rgbd_rgb[p]    for p in perm]
        self.combined_rgbd_early  = [self.combined_rgbd_early[p]  for p in perm]
        self.combined_masks       = [self.combined_masks[p]       for p in perm]
        self.combined_filenames   = [self.combined_filenames[p]   for p in perm]

    def _return_combinations(
        self,
        images,
        mask,
        base,
        ext
    ):
        """
        returns images/masks that have gone under all possible augmentations.

        `images` is a list in the canonical order
        [rgb, depth, rgd, rgbd_rgb, rgbd_early]; every transform is applied
        to the whole list at once so all sets stay spatially aligned with the mask.
        """

        transforms = [
            (self._flip_horizontally, "h"),
            (self._flip_vertically,   "v"),
            (self._rotate,            "r"),
        ]

        """
        essentially this piece of code applies every combination of transformation
        using the itertool. This ensures the maximum amount of images generated for
        the dataset
        """
        for combo in product([False, True], repeat=len(transforms)):
            if not any(combo):
                continue  # skip identity (no transform applied)

            imgs = [im.copy() for im in images]
            m = mask.copy()
            suffix = ""

            for apply, (fn, tag) in zip(combo, transforms):
                if apply:
                    imgs, m = fn(imgs, m)
                    suffix += tag

            self.new_rgb.append(imgs[0])
            self.new_depth.append(imgs[1])
            self.new_rgd.append(imgs[2])
            self.new_rgbd_rgb.append(imgs[3])
            self.new_rgbd_early.append(imgs[4])
            self.new_masks.append(m)
            self.new_filenames.append(f"{base}_aug_{suffix}{ext}")

    def _flip_vertically(
        self,
        images,
        mask
    ):
        """
        flips the image veritcally, since we are doing offline augmentations
        there is no probability on which image gets to be flipped or not
        """
        images = [cv2.flip(im, 0) for im in images]
        mask = cv2.flip(mask, 0)

        return images, mask

    def _elastic_transform(
        self,
        images,
        mask
    ):
        """
        uses the albumentations package to perform an elastic transform on the
        tumors

        TODO: chnage these constant values to variables to be passed through the
        pipeline

        TODO: finish implmenting this function -> not worth the time right now,
        plus it might add too much noise to the images anyway
        """
        raise NotImplementedError("_elastic_transform is not yet implemented")

        transform = A.ElasticTransform(
            alpha=100,
            sigma=10,
            p=1.0  # always apply since you're controlling this offline
        )

        images = [transform(image=im)["image"] for im in images]

        return images, mask


    def _flip_horizontally(
        self,
        images,
        mask
    ):
        """
        flips hte image horizontally, since we are doing offline augmentations
        there is no porbability on which an image gets flipped
        """
        images = [cv2.flip(im, 1) for im in images]
        mask = cv2.flip(mask, 1)

        return images, mask

    def _rotate(
        self,
        images,
        mask
    ):
        # single random angle shared across every image set (and the mask) so they
        # stay aligned; the rotation matrix is built per image to be robust to any
        # per-set H x W differences
        angle = np.random.uniform(-self.rotate_degrees, self.rotate_degrees)

        def _warp(img):
            h, w = img.shape[:2]
            M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

        images = [_warp(im) for im in images]
        mask = _warp(mask)

        return images, mask

    def return_augmentations(self):
        """
        returns all the directories, but now with combined augmenation
        """
        return (
            self.combined_rgb,
            self.combined_masks.copy(),
            self.combined_depth,
            self.combined_masks.copy(),
            self.combined_rgd,
            self.combined_masks.copy(),
            self.combined_rgbd_rgb,
            self.combined_rgbd_early,
            self.combined_masks.copy(),
            self.combined_filenames,
        )
