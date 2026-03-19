"""
creating custom augmentations 
"""

import os, cv2
from typing import List, Tuple 
import numpy as np 

class AugmentationClass: 

    def __init__(
            self, 
            tumor_dataset, 
            flip_prob: float, 
            rotate_prob: float,
            rotate_degrees: int, 
            target_size: int,
            test_only: bool = False # TODO: implement this feature  
    ): 
        """
        Stores a reference to the tumor dataset structure, accesses its masks and images 
        directly in order to augment the images to add more to our data 

        Args:
            tumor_dataset: reference to the tumor_dataset class
            target_size: this is the size that our data is selected to be 
            test_only: TODO, but this is to indicate if only the testing dataset should 
            be altered    
        """

        self.dataset = tumor_dataset
        self.test_only = test_only
        self.flip_prob = flip_prob
        self.rotate_prob = rotate_prob
        self.rotate_degrees = rotate_degrees
        self.target_size = target_size
        
        # some defined constants 
        self.dataset_size = len(self.dataset.images_rgb)

    def augment_images(self) -> None:

        if (self.target_size <= self.dataset_size): 
            print(f"Target size of {self.target_size} <= dataset size of {self.dataset_size}, therefore no augmentation needed")
            return 

        new_rgb, new_depth, new_rgd = [], [], []
        new_masks = []
        new_filenames = []

        i = 0
        while (len(new_rgb) + self.dataset_size) < self.target_size:
            idx = i % self.dataset_size
            image_rgb = self.dataset.images_rgb[idx].copy()
            image_depth = self.dataset.images_depth_maps[idx].copy()
            image_rgd = self.dataset.images_rgd[idx].copy()

            mask = self.dataset.masks[idx].copy()

            original_name = self.dataset.filenames[idx]

            # now apply the transformation
            image_rgb, image_depth, image_rgd, mask = self._flip_horizontally(image_rgb, image_depth, image_rgd, mask)
            image_rgb, image_depth, image_rgd, mask = self._flip_vertically(image_rgb, image_depth, image_rgd, mask)
            image_rgb, image_depth, image_rgd, mask = self._rotate(image_rgb, image_depth, image_rgd, mask)

            # now save them to the array 
            new_rgb.append(image_rgb)
            new_depth.append(image_depth)
            new_rgd.append(image_rgd)
            new_masks.append(mask)
            # change the filename as well 
            base, ext = os.path.splitext(original_name)
            new_filenames.append(f"{base}_aug_{i}{ext}")

            i += 1

        self.combined_rgb       = self.dataset.images_rgb        + new_rgb
        self.combined_depth     = self.dataset.images_depth_maps + new_depth
        self.combined_rgd       = self.dataset.images_rgd        + new_rgd
        self.combined_masks     = self.dataset.masks              + new_masks
        self.combined_filenames = self.dataset.filenames          + new_filenames

        # shuffle all parallel arrays with a single permutation to guarantee sync
        perm = np.random.permutation(len(self.combined_rgb))
        self.combined_rgb       = [self.combined_rgb[p]       for p in perm]
        self.combined_depth     = [self.combined_depth[p]     for p in perm]
        self.combined_rgd       = [self.combined_rgd[p]       for p in perm]
        self.combined_masks     = [self.combined_masks[p]     for p in perm]
        self.combined_filenames = [self.combined_filenames[p] for p in perm]


    def _flip_vertically(
        self, 
        rgb_image, 
        depth_image, 
        rgd_image, 
        mask
        ):
        if np.random.random() < self.flip_prob:
            rgb_image = cv2.flip(rgb_image, 0)
            depth_image = cv2.flip(depth_image, 0)
            rgd_image = cv2.flip(rgd_image, 0)
            mask = cv2.flip(mask, 0)
        return rgb_image, depth_image, rgd_image, mask

    def _flip_horizontally(
        self, 
        rgb_image, 
        depth_image, 
        rgd_image, 
        mask
        ): 
        if np.random.random() < self.flip_prob:
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
        if np.random.random() < self.rotate_prob:
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