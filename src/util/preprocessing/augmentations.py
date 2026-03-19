"""
this class is for creating custom augmentations that go into the workflow
BEFORE key image cropping, therefore preserving any information that would
get lost in the process
"""

import os
import cv2
import numpy as np
from typing import List, Tuple


class AugmentationClass:

    def __init__(
            self,
            tumor_dataset,
            flip_prob,
            rotate_prob,
            rotate_degrees,
            target_size: int,
            test_only: bool = False
        ):
        """
        Stores a reference to the Dataset instance and accesses its image/mask
        lists directly. Augmented pairs are appended in-place so originals are
        preserved.

        Args:
            tumor_dataset: instance of Dataset (tumor_dataset.py)
            target_size:   total dataset size after augmentation (originals +
                           augmented). The class generates (target_size - n_original)
                           augmented samples, then combines and shuffles all modalities
                           with a single shared permutation to keep them in sync.
            test_only:     if True, only the first image is processed —
                           useful for smoke-testing that augmentations apply
                           correctly before running on the full set
        """
        self.dataset = tumor_dataset
        self.test_only = test_only
        self.flip_prob = flip_prob
        self.rotate_prob = rotate_prob
        self.rotate_degrees = rotate_degrees
        self.target_size = target_size

    def augment_images(self) -> None:
        """
        Loops through the dataset (cycling with modulo) until the augmented
        bucket contains (target_size - n_original) new samples, then combines
        the originals and the bucket and shuffles all modalities with a single
        shared permutation so RGB, Depth, and RGD always stay in sync.

        If test_only is True, only the first image is processed.
        """
        has_depth = hasattr(self.dataset, "images_depth_maps") and self.dataset.images_depth_maps
        has_rgd = hasattr(self.dataset, "images_rgd") and self.dataset.images_rgd

        n = 1 if self.test_only else len(self.dataset.images_rgb)
        aug_needed = self.target_size - n  # how many augmented samples to generate

        if aug_needed <= 0:
            print(f"target_size ({self.target_size}) <= dataset size ({n}). No augmentation needed.")
            return

        new_rgb, new_depth, new_rgd = [], [], []
        new_masks = []
        new_filenames = []

        i = 0
        while len(new_rgb) < aug_needed:
            idx = i % n

            # bundle all modalities at this index so they share the same transform
            imgs = [self.dataset.images_rgb[idx]]
            if has_depth:
                imgs.append(self.dataset.images_depth_maps[idx])
            if has_rgd:
                imgs.append(self.dataset.images_rgd[idx])

            mask = self.dataset.masks[idx]

            imgs, mask = self._flip(imgs, mask, self.flip_prob)
            imgs, mask = self.__rotate(imgs, mask, self.rotate_prob, self.rotate_degrees)

            new_rgb.append(imgs[0])
            new_masks.append(mask)

            # derive augmented filename from the original
            orig_name = self.dataset.filenames[idx]
            base, ext = os.path.splitext(orig_name)
            new_filenames.append(f"{base}_aug_{i}{ext}")

            if has_depth:
                new_depth.append(imgs[1])
            if has_rgd:
                # rgd is always last; index is 1 (depth absent) or 2 (depth present)
                new_rgd.append(imgs[-1])

            i += 1

        # truncate to exactly aug_needed (loop may generate one extra on final pass)
        new_rgb   = new_rgb[:aug_needed]
        new_masks = new_masks[:aug_needed]
        new_filenames = new_filenames[:aug_needed]
        if has_depth:
            new_depth = new_depth[:aug_needed]
        if has_rgd:
            new_rgd = new_rgd[:aug_needed]

        # combine originals + augmented bucket
        combined_rgb       = self.dataset.images_rgb + new_rgb
        combined_masks     = self.dataset.masks      + new_masks
        combined_filenames = self.dataset.filenames   + new_filenames
        if has_depth:
            combined_depth = self.dataset.images_depth_maps + new_depth
        if has_rgd:
            combined_rgd = self.dataset.images_rgd + new_rgd
        # len(combined_rgb) == self.target_size

        # single permutation applied identically to all modalities — guarantees sync
        perm = np.random.permutation(len(combined_rgb))

        self.dataset.images_rgb = [combined_rgb[p]   for p in perm]
        self.dataset.masks      = [combined_masks[p] for p in perm]
        self.dataset.filenames  = [combined_filenames[p] for p in perm]

        if has_depth:
            self.dataset.images_depth_maps = [combined_depth[p] for p in perm]
            self.dataset.masks_clone_depth = [combined_masks[p] for p in perm]

        if has_rgd:
            self.dataset.images_rgd      = [combined_rgd[p]   for p in perm]
            self.dataset.masks_clone_rgd = [combined_masks[p] for p in perm]

        print(f"Augmentation done. Generated {aug_needed} augmented samples.")
        print(f"Total dataset size after augmentation: {len(self.dataset.images_rgb)} (target: {self.target_size})")

    def _flip(
        self, images: List[np.ndarray], mask: np.ndarray, prob: float
    ) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        Horizontally flips all images and the mask with probability `prob`.
        One random decision is shared across all image types.
        """
        if np.random.random() < prob:
            images = [cv2.flip(img, 1) for img in images]
            mask = cv2.flip(mask, 1)
        return images, mask

    def __rotate(
        self, images: List[np.ndarray], mask: np.ndarray, prob: float, degrees: float
    ) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        Rotates all images and the mask by a random angle in [-degrees, degrees]
        with probability `prob`. One rotation matrix is shared across all image
        types. Uses BORDER_REFLECT to avoid black border artifacts that would
        corrupt binary mask edges.
        """
        if np.random.random() < prob:
            h, w = images[0].shape[:2]
            angle = np.random.uniform(-degrees, degrees)
            M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            images = [cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT) for img in images]
            mask = cv2.warpAffine(mask, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        return images, mask


    def return_augmentations(self) -> Tuple[list, list, list, list, list, list]:
        """
        Returns a tuple of 6 arrays:
            (rgb_images, rgb_masks, depth_images, depth_masks, rgd_images, rgd_masks)
        Missing modalities are returned as empty lists.
        """
        rgb_images = self.dataset.images_rgb
        rgb_masks  = self.dataset.masks

        depth_images = self.dataset.images_depth_maps if hasattr(self.dataset, "images_depth_maps") and self.dataset.images_depth_maps else []
        depth_masks  = self.dataset.masks_clone_depth  if hasattr(self.dataset, "masks_clone_depth")  and self.dataset.masks_clone_depth  else []

        rgd_images = self.dataset.images_rgd        if hasattr(self.dataset, "images_rgd")        and self.dataset.images_rgd        else []
        rgd_masks  = self.dataset.masks_clone_rgd   if hasattr(self.dataset, "masks_clone_rgd")   and self.dataset.masks_clone_rgd   else []

        return rgb_images, rgb_masks, depth_images, depth_masks, rgd_images, rgd_masks
