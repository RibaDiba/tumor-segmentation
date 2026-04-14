import os
import random

import cv2
import numpy as np
import yaml


class NecroticMixin:
    """Mixin class that adds necrotic-region augmentation capability to FailureRecreation.

    Generates a synthetic necrotic patch near the tumour centroid by darkening a
    blob-shaped region. Necrotic tissue (dead or hypoxic tumour core) appears as a
    dark, irregular patch in the centre of a tumour — a real imaging feature that can
    mislead RGB models by removing the colour contrast that identifies the tumour.

    Each patch is composed of a single primary circle placed near the centroid plus a
    random number of smaller adjacent sub-circles (same organic blob format as
    RedRegionMixin), producing an irregular shape. All pixels within the blob are
    darkened by multiplying channel values by (1 - darken_effect / 100), where
    darken_effect is drawn randomly per image from the config range.

    The blue channel behaviour depends on ``model_type``:

    - ``'rgb'``: blue is a colour channel, so all three channels (B, G, R) are
      darkened, producing a realistic dark patch.
    - ``'rgd'``: blue encodes depth from the 3D scanner. It is **left untouched**,
      preserving the depth signal that identifies the raised tumour surface even when
      the colour channels are darkened.

    This class is not intended to be instantiated directly. It is used as a mixin
    via multiple inheritance in ``FailureRecreation``. The following instance
    attributes must be present on ``self`` before any method is called:

    Attributes:
        image (numpy.ndarray): Current BGR image being processed, shape (H, W, 3),
            dtype uint8. Modified in-place by ``_apply_necrotic_darkening``.
        mask (numpy.ndarray): Binary tumour mask, shape (H, W), dtype uint8.
            Pixel value 255 = tumour, 0 = background.
        aug_config (AugConfig): Dataclass holding the size-tier-specific parameters
            loaded from ``config.yaml``.
        current_filename (str): Stem of the current image filename (no extension).
            Used as the top-level key in ``necrotic_data`` and as the YAML log name.
        output_path (str): Root directory where augmented outputs are written.
            YAML logs are saved to ``<output_path>/logs/``.
        model_type (str): Either ``'rgb'`` or ``'rgd'``. Controls whether the blue
            channel is darkened (``'rgb'``) or preserved (``'rgd'``).
        necrotic_data (dict): Accumulator for randomly generated circle parameters.
            Reset at the start of each image's processing pass. Structure::

                {
                    "<filename>": {
                        "circles": [
                            {
                                "center": (cx, cy),
                                "offset": (ox, oy),
                                "radius": int,
                                "darken_effect": int,
                                "sub_circles": [
                                    {"center": (x, y), "radius": int},
                                    ...
                                ]
                            },
                            ...
                        ]
                    }
                }

            Where ``darken_effect`` is the percentage by which channels are reduced
            (applied as ``pixel * (1 - darken_effect / 100)``).
    """

    def __init__(self, image, mask, config):
        """Initialise the necrotic-region datastructure.

        In practice this is never called directly; ``FailureRecreation.__init__``
        sets up all shared attributes. The signature is preserved to document the
        expected per-image state.

        Args:
            image (numpy.ndarray): BGR image, shape (H, W, 3), dtype uint8.
            mask (numpy.ndarray): Binary mask, shape (H, W), dtype uint8.
            config (AugConfig): Size-tier-specific configuration dataclass.
        """
        self.necrotic_data = {}

    # ------------------------------------------------------------------
    # Public-facing orchestrator
    # ------------------------------------------------------------------

    def generate_necrotic_regions(self):
        """Orchestrate all steps needed to apply necrotic-region augmentation.

        This is the "main" function that runs the full pipeline for one image:
        circle generation → image darkening → YAML log. It resets the per-image
        datastructure before running so it is safe to call on multiple images
        sequentially.

        Side effects:
            - Populates ``self.necrotic_data[self.current_filename]``.
            - Modifies ``self.image`` in-place (R and G channels — and optionally B —
              darkened in necrotic patch areas).
            - Writes a YAML log file to
              ``<output_path>/logs/<filename>_necrotic.yaml``.
        """
        self.necrotic_data[self.current_filename] = {"circles": []}

        self._generate_necrotic_circles()
        self._apply_necrotic_darkening()
        self._save_necrotic_yml()

    # ------------------------------------------------------------------
    # Step 1: generate the circle datastructure
    # ------------------------------------------------------------------

    def _generate_necrotic_circles(self):
        """Build the necrotic circle datastructure for the current image.

        For each primary circle (controlled by ``n_blobs`` in config) this method:

        1. Computes the tumour centroid via ``cv2.moments`` on ``self.mask``, falling
           back to the image centre if the mask is empty.
        2. Applies a random (x, y) offset from the centroid to simulate the
           necrotic core not always being precisely centred.
        3. Clamps the resulting centre to image bounds.
        4. Assigns a random radius and a random darken-percentage.
        5. Delegates sub-circle generation to ``_generate_necrotic_sub_circles``.

        The results are appended to ``self.necrotic_data[self.current_filename]``.
        """
        cfg = self.aug_config.necrotic
        h, w = self.mask.shape[:2]

        # ------------------------------------------------------------------
        # Compute tumour centroid (identical to _specular.py pattern)
        # ------------------------------------------------------------------
        M = cv2.moments(self.mask)
        if M["m00"] == 0:
            centroid_x, centroid_y = w // 2, h // 2
        else:
            centroid_x = int(M["m10"] / M["m00"])
            centroid_y = int(M["m01"] / M["m00"])

        # ------------------------------------------------------------------
        # Generate n_blobs primary circles (always 1 in current config)
        # ------------------------------------------------------------------
        for _ in range(cfg["n_blobs"]):
            ox = random.randint(-cfg["offset_x"], cfg["offset_x"])
            oy = random.randint(-cfg["offset_y"], cfg["offset_y"])

            # Clamp to image bounds so the circle centre is always visible
            cx = max(0, min(w - 1, centroid_x + ox))
            cy = max(0, min(h - 1, centroid_y + oy))
            center = (cx, cy)

            radius = random.randint(cfg["radius_min"], cfg["radius_max"])
            darken_effect = random.randint(cfg["darken_min"], cfg["darken_max"])

            sub_circles = self._generate_necrotic_sub_circles(center, radius)

            self.necrotic_data[self.current_filename]["circles"].append(
                {
                    "center": center,
                    "offset": (ox, oy),
                    "radius": radius,
                    "darken_effect": darken_effect,
                    "sub_circles": sub_circles,
                }
            )

    def _generate_necrotic_sub_circles(self, parent_center, parent_radius):
        """Generate sub-circles adjacent to the primary necrotic circle.

        Produces an organic, irregular blob shape by placing smaller satellite
        circles whose centres are sampled from within the parent disk plus a small
        random offset. This mirrors ``RedRegionMixin._generate_addtional_circles``
        in structure and algorithm, but reads from ``aug_config.necrotic["sub_circles"]``
        rather than ``aug_config.red_region["addtional_points"]``.

        Args:
            parent_center (tuple[int, int]): (x, y) pixel coordinate of the primary
                circle's centre.
            parent_radius (int): Radius of the primary circle in pixels.

        Returns:
            list[dict]: A list of sub-circle descriptors, each with keys:

                - ``"center"`` (tuple[int, int]): (x, y) pixel coordinate.
                - ``"radius"`` (int): Sub-circle radius in pixels.

            Returns an empty list if the random draw produces zero sub-circles.
        """
        cfg = self.aug_config.necrotic
        sub_cfg = cfg["sub_circles"]

        n_sub = random.randint(0, sub_cfg["range"])
        sub_circles = []

        for _ in range(n_sub):
            # Uniform disk sampling inside the parent circle
            angle = random.uniform(0, 2 * np.pi)
            r_frac = random.uniform(0, parent_radius)
            px = int(parent_center[0] + r_frac * np.cos(angle))
            py = int(parent_center[1] + r_frac * np.sin(angle))

            ox = random.randint(-sub_cfg["offset_x"], sub_cfg["offset_x"])
            oy = random.randint(-sub_cfg["offset_y"], sub_cfg["offset_y"])
            sub_center = (px + ox, py + oy)
            sub_radius = random.randint(
                max(1, cfg["radius_min"] // 2), parent_radius
            )

            sub_circles.append({"center": sub_center, "radius": sub_radius})

        return sub_circles

    # ------------------------------------------------------------------
    # Step 2: apply the darkening to the image
    # ------------------------------------------------------------------

    def _apply_necrotic_darkening(self):
        """Apply necrotic darkening to all circles in the datastructure.

        Iterates over every primary circle and sub-circle stored in
        ``self.necrotic_data[self.current_filename]`` and darkens the pixels
        within each circle's disk using:

            pixel' = pixel * (1 - darken_effect / 100)

        Channels modified:
        - R (channel 2) and G (channel 1): always darkened.
        - B (channel 0): darkened when ``model_type == 'rgb'`` (B is colour);
          left untouched when ``model_type == 'rgd'`` (B is depth).

        Overlapping pixels receive the cumulative darkening of all circles that
        cover them (each multiplication further reduces the value). Final values
        are clipped to the uint8 range [0, 255].

        Side effects:
            Modifies ``self.image`` in-place.
        """
        h, w = self.mask.shape[:2]
        image_float = self.image.astype(np.float32)

        for circle in self.necrotic_data[self.current_filename]["circles"]:
            all_circles = [
                {
                    "center": circle["center"],
                    "radius": circle["radius"],
                    "darken_effect": circle["darken_effect"],
                }
            ]
            # Sub-circles inherit the parent's darken_effect
            for sub in circle["sub_circles"]:
                all_circles.append(
                    {
                        "center": sub["center"],
                        "radius": sub["radius"],
                        "darken_effect": circle["darken_effect"],
                    }
                )

            for c in all_circles:
                cx, cy = c["center"]
                r = c["radius"]
                factor = 1.0 - c["darken_effect"] / 100.0

                # Build a binary mask for this circle
                circle_mask = np.zeros((h, w), dtype=np.uint8)
                cv2.circle(circle_mask, (cx, cy), r, 255, -1)
                region = circle_mask > 0

                # Always darken R (ch2) and G (ch1)
                image_float[:, :, 2][region] = image_float[:, :, 2][region] * factor
                image_float[:, :, 1][region] = image_float[:, :, 1][region] * factor

                # B (ch0): darken for RGB images (B is colour);
                # leave untouched for RGD images (B is depth)
                if self.model_type == "rgb":
                    image_float[:, :, 0][region] = (
                        image_float[:, :, 0][region] * factor
                    )

        self.image = np.clip(image_float, 0, 255).astype(np.uint8)

    # ------------------------------------------------------------------
    # Step 3: save the YAML log
    # ------------------------------------------------------------------

    def _save_necrotic_yml(self):
        """Save all randomly generated necrotic circle parameters to a YAML log file.

        Creates a YAML file at ``<output_path>/logs/<filename>_necrotic.yaml``
        that records every random value used during necrotic augmentation of the
        current image. This allows any specific augmented image to be reproduced
        exactly by replaying the stored values.

        Tuples are converted to lists before dumping so the output contains plain
        YAML sequences rather than ``!!python/tuple`` tags (matching _red.py and
        _specular.py conventions).

        The output directory is created automatically if it does not exist.

        Side effects:
            Writes a YAML file to disk.
        """
        log_dir = os.path.join(self.output_path, "logs")
        os.makedirs(log_dir, exist_ok=True)

        log_path = os.path.join(log_dir, f"{self.current_filename}_necrotic.yaml")

        serialisable = {"circles": []}
        for circle in self.necrotic_data[self.current_filename]["circles"]:
            serialisable["circles"].append(
                {
                    "center": list(circle["center"]),
                    "offset": list(circle["offset"]),
                    "radius": circle["radius"],
                    "darken_effect": circle["darken_effect"],
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
