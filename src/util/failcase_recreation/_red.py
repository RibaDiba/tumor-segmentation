import os
import random

import cv2
import numpy as np
import yaml


class RedRegionMixin:
    """Mixin class that adds red-region augmentation capability to FailureRecreation.

    Generates synthetic patches of red/pink discoloration at the border of the tumor
    mask. Each patch is composed of a primary circle placed on the tumor border plus
    a random number of smaller adjacent sub-circles, producing organic "blob"-shaped
    regions. The red channel of the underlying image is increased by a random
    percentage within each region, with overlapping regions stacking additively.

    This simulates skin-irritation or vascular discoloration that could mislead an
    RGB-only segmentation model, while leaving the blue channel (depth information in
    RGD images) entirely unmodified.

    This class is not intended to be instantiated directly. It is used as a mixin
    via multiple inheritance in ``FailureRecreation``. The following instance
    attributes must be present on ``self`` before any method is called:

    Attributes:
        image (numpy.ndarray): Current BGR image being processed, shape (H, W, 3),
            dtype uint8. Modified in-place by ``_fill``.
        mask (numpy.ndarray): Binary tumor mask, shape (H, W), dtype uint8.
            Pixel value 255 = tumor, 0 = background.
        aug_config (AugConfig): Dataclass holding the size-tier-specific parameters
            loaded from ``config.yaml``.
        current_filename (str): Stem of the current image filename (no extension).
            Used as the top-level key in ``circle_data`` and as the YAML log name.
        output_path (str): Root directory where augmented outputs are written.
            YAML logs are saved to ``<output_path>/logs/``.
        circle_data (dict): Accumulator for randomly generated circle parameters.
            Reset at the start of each image's processing pass. Structure::

                {
                    "<filename>": {
                        "circles": [
                            {
                                "center": (cx, cy),
                                "offset": (ox, oy),
                                "radius": int,
                                "red_effect": int,
                                "sub_circles": [
                                    {"center": (x, y), "radius": int},
                                    ...
                                ]
                            },
                            ...
                        ]
                    }
                }
    """

    def __init__(self, image, mask, config):
        """Initialise the red-region datastructure.

        In practice this is never called directly; ``FailureRecreation.__init__``
        sets up all shared attributes. The signature is preserved to document the
        expected per-image state.

        Args:
            image (numpy.ndarray): BGR image, shape (H, W, 3), dtype uint8.
            mask (numpy.ndarray): Binary mask, shape (H, W), dtype uint8.
            config (AugConfig): Size-tier-specific configuration dataclass.
        """
        # Initialise the datastructure that stores all randomly generated
        # numbers for this image. The structure is keyed by filename so that
        # multiple images can accumulate without collision.
        self.circle_data = {}

    # ------------------------------------------------------------------
    # Public-facing orchestrator
    # ------------------------------------------------------------------

    def _create_red_regions(self):
        """Orchestrate all steps needed to apply red-region augmentation.

        This is the "main" function that runs the full pipeline for one image:
        circle generation → image fill → YAML log. It resets the per-image
        datastructure before running so it is safe to call on multiple images
        sequentially.

        Side effects:
            - Populates ``self.circle_data[self.current_filename]``.
            - Modifies ``self.image`` in-place (red channel increased in patch areas).
            - Writes a YAML log file to ``<output_path>/logs/<filename>_red.yaml``.
        """
        # Reset the per-image datastructure
        self.circle_data[self.current_filename] = {"circles": []}

        # Run each step in order
        self._generate_circles()
        self._fill()
        self._generate_yml_file()

    # ------------------------------------------------------------------
    # Step 1: generate the circle datastructure
    # ------------------------------------------------------------------

    def _generate_circles(self):
        """Build the full circle datastructure for the current image.

        For each primary circle this method:
        1. Extracts all border pixels from ``self.mask`` using contour detection.
        2. Samples a random number of non-conflicting border points.
        3. Applies a random (x, y) offset to each chosen border point to position
           the circle centre.
        4. Assigns a random radius and a random red-channel increase percentage.
        5. Delegates sub-circle generation to ``_generate_addtional_circles``.

        The results are appended to ``self.circle_data[self.current_filename]``.

        Raises:
            ValueError: If the mask contains no contour (i.e. no tumor region
                found). This should not happen on valid preprocessed data.
        """
        cfg = self.aug_config.red_region

        # ------------------------------------------------------------------
        # Get the border of the tumor
        # ------------------------------------------------------------------
        contours, _ = cv2.findContours(
            self.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        if not contours:
            raise ValueError(
                f"No contour found in mask for image '{self.current_filename}'. "
                "Ensure the mask is a valid binary image with a tumour region."
            )
        # Use the largest contour (primary tumour outline)
        contour = max(contours, key=cv2.contourArea)
        # Flatten from (N, 1, 2) → list of (x, y) tuples
        border_pixels = [tuple(pt[0]) for pt in contour]
        n_border = len(border_pixels)

        # ------------------------------------------------------------------
        # Generate number of circles
        # ------------------------------------------------------------------
        n_circles = random.randint(1, cfg["interest_points"])

        # ------------------------------------------------------------------
        # Generate location (on the border of the tumor)
        # Configured to not have conflicting points: enforce a minimum
        # separation of 2 * radius_max pixels between chosen indices.
        # ------------------------------------------------------------------
        min_separation = max(cfg["radius_max"] * 2, 20)
        chosen_indices = []
        max_attempts = n_border * 3  # upper bound on retries
        attempts = 0

        while len(chosen_indices) < n_circles and attempts < max_attempts:
            candidate = random.randint(0, n_border - 1)
            # Check distance from every already-chosen index (circular wrap)
            conflict = any(
                min(abs(candidate - idx), n_border - abs(candidate - idx))
                < min_separation
                for idx in chosen_indices
            )
            if not conflict:
                chosen_indices.append(candidate)
            attempts += 1

        # ------------------------------------------------------------------
        # For each chosen border point, build the full circle definition
        # ------------------------------------------------------------------
        for idx in chosen_indices:
            bx, by = border_pixels[idx]

            # Generate offset using the yaml configs
            ox = random.randint(-cfg["point_offset_x"], cfg["point_offset_x"])
            oy = random.randint(-cfg["point_offset_y"], cfg["point_offset_y"])
            center = (bx + ox, by + oy)

            # Generate a random radius
            radius = random.randint(cfg["radius_min"], cfg["radius_max"])

            # Generate percentage to increase the red channel by within that region
            red_effect = random.randint(cfg["red_effect_min"], cfg["red_effect_max"])

            # Now loop through to generate additional circles (sub-blobs)
            sub_circles = self._generate_addtional_circles(center, radius)

            self.circle_data[self.current_filename]["circles"].append(
                {
                    "center": center,
                    "offset": (ox, oy),
                    "radius": radius,
                    "red_effect": red_effect,
                    "sub_circles": sub_circles,
                }
            )

    def _generate_addtional_circles(self, parent_center, parent_radius):
        """Generate sub-circles adjacent to a primary circle to form realistic blobs.

        For each primary circle, a random number of smaller satellite circles are
        generated. Their centres are chosen from within the parent circle's disk,
        plus a small additional random offset, so they overlap with the parent and
        with each other to produce an organic, irregular shape.

        Args:
            parent_center (tuple[int, int]): (x, y) pixel coordinate of the parent
                circle's centre.
            parent_radius (int): Radius of the parent circle in pixels.

        Returns:
            list[dict]: A list of sub-circle descriptors, each with keys:

                - ``"center"`` (tuple[int, int]): (x, y) pixel coordinate.
                - ``"radius"`` (int): Sub-circle radius in pixels.

            Returns an empty list if the random draw produces zero sub-circles.
        """
        cfg = self.aug_config.red_region
        add_cfg = cfg["addtional_points"]

        # Generate additional points
        n_sub = random.randint(0, add_cfg["range"])
        sub_circles = []

        for _ in range(n_sub):
            # Pick a random pixel inside the parent circle (uniform disk sampling)
            angle = random.uniform(0, 2 * np.pi)
            r_frac = random.uniform(0, parent_radius)
            px = int(parent_center[0] + r_frac * np.cos(angle))
            py = int(parent_center[1] + r_frac * np.sin(angle))

            # Generate offset and radius
            ox = random.randint(-add_cfg["offset_x"], add_cfg["offset_x"])
            oy = random.randint(-add_cfg["offset_y"], add_cfg["offset_y"])
            sub_center = (px + ox, py + oy)
            sub_radius = random.randint(
                max(1, cfg["radius_min"] // 2), parent_radius
            )

            sub_circles.append({"center": sub_center, "radius": sub_radius})

        return sub_circles

    # ------------------------------------------------------------------
    # Step 2: apply the circles to the image
    # ------------------------------------------------------------------

    def _fill(self):
        """Apply the red-channel increase to all circles in the datastructure.

        Iterates over every primary circle and sub-circle stored in
        ``self.circle_data[self.current_filename]`` and increases the red channel
        (channel index 2 in BGR) of the pixels within each circle's disk by the
        circle's ``red_effect`` percentage. Overlapping pixels receive the
        cumulative effect of all circles that cover them. Final values are clipped
        to the uint8 maximum of 255.

        Side effects:
            Modifies ``self.image`` in-place.
        """
        h, w = self.mask.shape[:2]

        # Work in float32 to allow safe additive accumulation before clipping
        image_float = self.image.astype(np.float32)

        # Alter the image: iterate all primary circles
        for circle in self.circle_data[self.current_filename]["circles"]:
            all_circles = [
                {"center": circle["center"], "radius": circle["radius"], "red_effect": circle["red_effect"]}
            ]
            # Sub-circles inherit the parent's red_effect
            for sub in circle["sub_circles"]:
                all_circles.append(
                    {"center": sub["center"], "radius": sub["radius"], "red_effect": circle["red_effect"]}
                )

            for c in all_circles:
                cx, cy = c["center"]
                r = c["radius"]
                effect_pct = c["red_effect"] / 100.0

                # Build a binary mask for this circle
                circle_mask = np.zeros((h, w), dtype=np.uint8)
                cv2.circle(circle_mask, (cx, cy), r, 255, -1)

                # Increase red channel (index 2 in BGR) by the effect percentage
                region = circle_mask > 0
                image_float[:, :, 2][region] = (
                    image_float[:, :, 2][region] * (1.0 + effect_pct)
                )

        # Clip and convert back to uint8
        self.image = np.clip(image_float, 0, 255).astype(np.uint8)

    # ------------------------------------------------------------------
    # Step 3: save the YAML log
    # ------------------------------------------------------------------

    def _generate_yml_file(self):
        """Save all randomly generated circle parameters to a YAML log file.

        Creates a YAML file at ``<output_path>/logs/<filename>_red.yaml`` that
        records every random value used during the augmentation of the current
        image. This log can be used to reproduce any specific augmented image
        exactly by re-seeding the random calls with the stored values.

        The output directory is created automatically if it does not exist.

        Side effects:
            Writes a YAML file to disk.
        """
        log_dir = os.path.join(self.output_path, "logs")
        os.makedirs(log_dir, exist_ok=True)

        log_path = os.path.join(log_dir, f"{self.current_filename}_red.yaml")

        # Build a YAML-serialisable copy (tuples → lists for safe dumping)
        serialisable = {"circles": []}
        for circle in self.circle_data[self.current_filename]["circles"]:
            serialisable["circles"].append(
                {
                    "center": list(circle["center"]),
                    "offset": list(circle["offset"]),
                    "radius": circle["radius"],
                    "red_effect": circle["red_effect"],
                    "sub_circles": [
                        {"center": list(s["center"]), "radius": s["radius"]}
                        for s in circle["sub_circles"]
                    ],
                }
            )

        with open(log_path, "w") as f:
            yaml.dump(
                {self.current_filename: serialisable},
                f,
                default_flow_style=False,
                sort_keys=False,
            )
