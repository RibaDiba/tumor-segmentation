"""
creating custom augmentations 
"""

import os, cv2
from itertools import product
from typing import List, Tuple
import numpy as np
import albumentations as A

class AugmentationClass: 

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
        self.new_masks = []
        self.new_filenames = []

        for idx in range(self.dataset_size):
            image_rgb   = self.dataset.images_rgb[idx].copy()
            image_depth = self.dataset.images_depth_maps[idx].copy()
            image_rgd   = self.dataset.images_rgd[idx].copy()
            mask        = self.dataset.masks[idx].copy()
            original_name = self.dataset.filenames[idx]
            base, ext = os.path.splitext(original_name)

            # populates the new arrays 
            self._return_combinations(
                image_rgb, 
                image_depth, 
                image_rgd, 
                mask, 
                base,
                ext
            )

        # merges all datasets togther
        self.combined_rgb       = self.dataset.images_rgb        + self.new_rgb
        self.combined_depth     = self.dataset.images_depth_maps + self.new_depth
        self.combined_rgd       = self.dataset.images_rgd        + self.new_rgd
        self.combined_masks     = self.dataset.masks              + self.new_masks
        self.combined_filenames = self.dataset.filenames          + self.new_filenames

        # shuffle all parallel arrays with a single permutation to guarantee sync
        perm = np.random.permutation(len(self.combined_rgb))
        self.combined_rgb       = [self.combined_rgb[p]       for p in perm]
        self.combined_depth     = [self.combined_depth[p]     for p in perm]
        self.combined_rgd       = [self.combined_rgd[p]       for p in perm]
        self.combined_masks     = [self.combined_masks[p]     for p in perm]
        self.combined_filenames = [self.combined_filenames[p] for p in perm]

    def _return_combinations(
        self, 
        image_rgb, 
        image_depth, 
        image_rgd, 
        mask,
        base,
        ext
    ): 
        """
        returns images/masks that have gone under all possible augmentations
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

            r, d, g, m = image_rgb.copy(), image_depth.copy(), image_rgd.copy(), mask.copy()
            suffix = ""

            for apply, (fn, tag) in zip(combo, transforms):
                if apply:
                    r, d, g, m = fn(r, d, g, m)
                    suffix += tag

            self.new_rgb.append(r)
            self.new_depth.append(d)
            self.new_rgd.append(g)
            self.new_masks.append(m)
            self.new_filenames.append(f"{base}_aug_{suffix}{ext}")

    def _flip_vertically(
        self, 
        rgb_image, 
        depth_image, 
        rgd_image, 
        mask
    ):
        """
        flips the image veritcally, since we are doing offline augmentations 
        there is no probability on which image gets to be flipped or not
        """
        rgb_image = cv2.flip(rgb_image, 0)
        depth_image = cv2.flip(depth_image, 0)
        rgd_image = cv2.flip(rgd_image, 0)
        mask = cv2.flip(mask, 0)

        return rgb_image, depth_image, rgd_image, mask
    
    def _elastic_transform(
        self, 
        rgb_image, 
        depth_image, 
        rgd_image, 
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

        rgb_result = transform(image=rgb_image)
        depth_result = transform(image=depth_image)
        rgd_result = transform(image=rgd_image)

        rgb_image = rgb_result['image']
        depth_image = depth_result['image']
        rgd_image = rgd_result['image']
        
        return rgb_image, depth_image, rgd_image, mask


    def _flip_horizontally(
        self, 
        rgb_image, 
        depth_image, 
        rgd_image, 
        mask
    ): 
        """
        flips hte image horizontally, since we are doing offline augmentations
        there is no porbability on which an image gets flipped 
        """
        rgb_image = cv2.flip(rgb_image, 1)
        depth_image = cv2.flip(depth_image, 1)
        rgd_image = cv2.flip(rgd_image, 1)
        mask = cv2.flip(mask, 1)

        return rgb_image, depth_image, rgd_image, mask

    def _rotate(
        self, 
        rgb_image, 
        depth_image, 
        rgd_image, 
        mask
    ): 

        h, w = rgb_image.shape[:2]
        angle = np.random.uniform(-self.rotate_degrees, self.rotate_degrees)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        rgb_image = cv2.warpAffine(rgb_image, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        depth_image = cv2.warpAffine(depth_image, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        rgd_image = cv2.warpAffine(rgd_image, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        mask = cv2.warpAffine(mask, M, (w, h), borderMode=cv2.BORDER_REFLECT)

        return rgb_image, depth_image, rgd_image, mask

    def return_augmentations(self):
        """
        returns all the directories, but now with combined augmenation
        """
        return self.combined_rgb, self.combined_masks.copy(), self.combined_depth, self.combined_masks.copy(), self.combined_rgd, self.combined_masks.copy(), self.combined_filenames