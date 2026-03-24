import os
import random

import cv2
import numpy as np
import yaml


class ShadowMixin:
    """Mixin class that adds shadow augmentation capability to FailureRecreation.

    Generates synthetic shadow arcs along the border of the tumour mask.
    Each shadow region is a contiguous arc of border pixels whose surrounding
    area (extending outward into the background by a random number of pixels)
    is darkened by reducing the red and green colour channels.

    The blue channel is never modified. In RGD images the blue channel encodes
    depth information, so preserving it means the RGD model retains its depth
    signal even when shadows are applied — exactly the condition we want to
    evaluate.

    This class is not intended to be instantiated directly. It is used as a
    mixin via multiple inheritance in ``FailureRecreation``. The following
    instance attributes must be present on ``self`` before any method is called:

    Attributes:
        image (numpy.ndarray): Current BGR image being processed, shape (H, W, 3),
            dtype uint8. Modified in-place by ``_apply_shadows``.
        mask (numpy.ndarray): Binary tumour mask, shape (H, W), dtype uint8.
            Pixel value 255 = tumour, 0 = background.
        aug_config (AugConfig): Dataclass holding the size-tier-specific parameters
            loaded from ``config.yaml``.
        current_filename (str): Stem of the current image filename (no extension).
            Used as the top-level key in ``shadow_data`` and as the YAML log name.
        output_path (str): Root directory where augmented outputs are written.
            YAML logs are saved to ``<output_path>/logs/``.
        shadow_data (dict): Accumulator for randomly generated shadow parameters.
            Reset at the start of each image's processing pass. Structure::

                {
                    "<filename>": {
                        "shadows": [
                            {
                                "start_idx": int,
                                "length":    int,
                                "offset":    int,
                                "decrease":  int
                            },
                            ...
                        ]
                    }
                }

            Where:
            - ``start_idx`` is the index into the ordered contour array.
            - ``length`` is the number of consecutive contour pixels in the arc.
            - ``offset`` is the shadow band width in pixels extending outward
              from the border.
            - ``decrease`` is the percentage reduction applied to the R and G
              colour channels (0–100).
    """

    def __init__(self, image, mask, config):
        """Initialise the shadow datastructure.

        In practice this is never called directly; ``FailureRecreation.__init__``
        sets up all shared attributes. The signature is preserved to document the
        expected per-image state.

        Args:
            image (numpy.ndarray): BGR image, shape (H, W, 3), dtype uint8.
            mask (numpy.ndarray): Binary mask, shape (H, W), dtype uint8.
            config (AugConfig): Size-tier-specific configuration dataclass.
        """
        # Initialise the datastructure that stores all random values for this
        # image, keyed by filename to allow accumulation across multiple images.
        self.shadow_data = {}

    # ------------------------------------------------------------------
    # Public-facing orchestrator
    # ------------------------------------------------------------------

    def generate_shadows(self):
        """Orchestrate all steps needed to apply shadow augmentation.

        This is the "main" function that calls everything for one image:
        shadow-region generation → image application → YAML log. It resets the
        per-image datastructure before running so it is safe to call on multiple
        images sequentially.

        Side effects:
            - Populates ``self.shadow_data[self.current_filename]``.
            - Modifies ``self.image`` in-place (R and G channels reduced in
              shadow areas; B channel unchanged).
            - Writes a YAML log file to
              ``<output_path>/logs/<filename>_shadow.yaml``.
        """
        # Reset the per-image datastructure
        self.shadow_data[self.current_filename] = {"shadows": []}

        self._generate_shadow_regions()
        self._apply_shadows()
        self._save_yml()

    # ------------------------------------------------------------------
    # Step 1: generate the shadow datastructure
    # ------------------------------------------------------------------

    def _generate_shadow_regions(self):
        """Build the shadow-region datastructure for the current image.

        For each shadow arc this method:
        1. Extracts the full ordered border of the tumour using contour detection.
        2. Randomly samples a number of non-overlapping arc regions.
        3. Assigns each region a random length, outward band width (offset), and
           colour-channel decrease percentage.

        The constraint that ``borders cannot overlap`` (as specified in the design
        doc) is enforced by tracking all occupied contour indices and retrying
        placement up to ``max_retries`` times per region.

        The results are appended to ``self.shadow_data[self.current_filename]``.

        Raises:
            ValueError: If no contour is found in the mask.
        """
        cfg = self.aug_config.shadow

        # ------------------------------------------------------------------
        # Extract the full ordered contour (every border pixel)
        # ------------------------------------------------------------------
        contours, _ = cv2.findContours(
            self.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        if not contours:
            raise ValueError(
                f"No contour found in mask for image '{self.current_filename}'. "
                "Ensure the mask is a valid binary image with a tumour region."
            )
        contour = max(contours, key=cv2.contourArea)
        n_border = len(contour)

        # Store the contour on self so _apply_shadows can reuse it without
        # re-running findContours
        self._shadow_contour = contour

        # ------------------------------------------------------------------
        # Generate a random number of shadow regions
        # ------------------------------------------------------------------
        n_regions = random.randint(1, cfg["number_regions"])

        occupied_indices = set()  # tracks all contour indices already covered
        max_retries = 20

        for _ in range(n_regions):
            # Generate a random length for each region
            length = random.randint(cfg["length_min"], cfg["length_max"])
            # Clamp length to at most the full contour minus already-used pixels
            length = min(length, n_border - len(occupied_indices))
            if length <= 0:
                break

            # Generate sections of the border — validation: borders cannot overlap
            placed = False
            for _attempt in range(max_retries):
                start_idx = random.randint(0, n_border - 1)
                arc_indices = set(
                    (start_idx + i) % n_border for i in range(length)
                )
                if not arc_indices & occupied_indices:
                    occupied_indices |= arc_indices
                    placed = True
                    break

            if not placed:
                continue  # skip this region if no valid position found

            # Generate percentage to decrease from the green and red channels
            offset = random.randint(cfg["offset_min"], cfg["offset_max"])
            decrease = random.randint(cfg["decrease_min"], cfg["decrease_max"])

            self.shadow_data[self.current_filename]["shadows"].append(
                {
                    "start_idx": start_idx,
                    "length": length,
                    "offset": offset,
                    "decrease": decrease,
                }
            )

    # ------------------------------------------------------------------
    # Step 2: apply the shadows to the image
    # ------------------------------------------------------------------

    def _apply_shadows(self):
        """Apply the shadow band to the image for each generated shadow region.

        For each shadow region stored in ``self.shadow_data``:

        1. The arc's pixel coordinates are extracted from the stored contour.
        2. Those border pixels are drawn onto a blank canvas.
        3. The canvas is dilated outward by ``offset`` pixels using a circular
           structuring element, creating a band around the arc.
        4. The band is masked to the background only (pixels outside the tumour)
           so the tumour interior is never darkened.
        5. The red (channel 2) and green (channel 1) pixel values in the shadow
           band are multiplied by ``(1 - decrease / 100)``. The blue channel
           (channel 0, which carries depth in RGD images) is left untouched.

        Side effects:
            Modifies ``self.image`` in-place.
        """
        h, w = self.mask.shape[:2]
        contour = self._shadow_contour
        n_border = len(contour)

        image_float = self.image.astype(np.float32)

        for region in self.shadow_data[self.current_filename]["shadows"]:
            start_idx = region["start_idx"]
            length = region["length"]
            offset = region["offset"]
            decrease = region["decrease"]
            factor = 1.0 - decrease / 100.0

            # Extract the arc's pixel coordinates
            arc_pts = [
                tuple(contour[(start_idx + i) % n_border][0])
                for i in range(length)
            ]

            # Draw arc pixels onto a blank canvas
            arc_mask = np.zeros((h, w), dtype=np.uint8)
            for px, py in arc_pts:
                if 0 <= py < h and 0 <= px < w:
                    arc_mask[py, px] = 255

            # Dilate outward by offset pixels using a circular kernel
            kernel_size = 2 * offset + 1
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
            )
            dilated = cv2.dilate(arc_mask, kernel)

            # Intersect with background (outside the tumour) so that the tumour
            # interior is never darkened
            background = (self.mask == 0).astype(np.uint8) * 255
            shadow_band = cv2.bitwise_and(dilated, background)

            region_pixels = shadow_band > 0

            # Apply decrease to R and G channels only (B channel = depth, never modified)
            image_float[:, :, 1][region_pixels] = np.clip(
                image_float[:, :, 1][region_pixels] * factor, 0, 255
            )
            image_float[:, :, 2][region_pixels] = np.clip(
                image_float[:, :, 2][region_pixels] * factor, 0, 255
            )

        self.image = np.clip(image_float, 0, 255).astype(np.uint8)

    # ------------------------------------------------------------------
    # Step 3: save the YAML log
    # ------------------------------------------------------------------

    def _save_yml(self):
        """Save all randomly generated shadow parameters to a YAML log file.

        Creates a YAML file at ``<output_path>/logs/<filename>_shadow.yaml``
        that records every random value used during the shadow augmentation of
        the current image. This allows any specific augmented image to be
        reproduced exactly.

        The output directory is created automatically if it does not exist.

        Side effects:
            Writes a YAML file to disk.
        """
        log_dir = os.path.join(self.output_path, "logs")
        os.makedirs(log_dir, exist_ok=True)

        log_path = os.path.join(log_dir, f"{self.current_filename}_shadow.yaml")

        with open(log_path, "w") as f:
            yaml.dump(
                {self.current_filename: self.shadow_data[self.current_filename]},
                f,
                default_flow_style=False,
                sort_keys=False,
            )
