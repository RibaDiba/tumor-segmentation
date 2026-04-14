import os
import random

import cv2
import numpy as np
import yaml


class SpecularMixin:
    """Mixin class that adds specular highlight augmentation capability to FailureRecreation.

    Generates a synthetic specular highlight — a bright, white-saturated Gaussian
    blob — centred near the tumour centroid. This simulates a real imaging artefact:
    the 3D scanner's overhead light source reflecting directly off the raised skin
    surface above the tumour. The peak of a subcutaneous lump is the highest point
    facing the scanner, so it receives the most direct specular reflection.

    The highlight is modelled as a soft Gaussian falloff from the centre toward white
    (255 in each affected channel):

        alpha(x, y) = (intensity / 100) * exp(-((x - cx)² + (y - cy)²) / (2 * σ²))
        pixel'      = pixel * (1 - alpha) + 255 * alpha

    Multiple overlapping highlights are applied sequentially; their effects stack
    additively because each step blends the running result further toward white.

    The blue channel behaviour depends on ``model_type``:

    - ``'rgb'``: blue is a colour channel, so it is also blended toward white.
    - ``'rgd'``: blue encodes depth from the 3D scanner. It is **left untouched**,
      preserving the depth signal that identifies the raised tumour even when colour
      is saturated.

    This class is not intended to be instantiated directly. It is used as a mixin
    via multiple inheritance in ``FailureRecreation``. The following instance
    attributes must be present on ``self`` before any method is called:

    Attributes:
        image (numpy.ndarray): Current BGR image being processed, shape (H, W, 3),
            dtype uint8. Modified in-place by ``_apply_specular``.
        mask (numpy.ndarray): Binary tumour mask, shape (H, W), dtype uint8.
            Pixel value 255 = tumour, 0 = background.
        aug_config (AugConfig): Dataclass holding the size-tier-specific parameters
            loaded from ``config.yaml``.
        current_filename (str): Stem of the current image filename (no extension).
            Used as the top-level key in ``specular_data`` and as the YAML log name.
        output_path (str): Root directory where augmented outputs are written.
            YAML logs are saved to ``<output_path>/logs/``.
        model_type (str): Either ``'rgb'`` or ``'rgd'``. Controls whether the blue
            channel is saturated (``'rgb'``) or preserved (``'rgd'``).
        specular_data (dict): Accumulator for randomly generated highlight parameters.
            Reset at the start of each image's processing pass. Structure::

                {
                    "<filename>": {
                        "specular_regions": [
                            {
                                "center":    (cx, cy),
                                "offset":    (ox, oy),
                                "sigma":     int,
                                "intensity": int
                            },
                            ...
                        ]
                    }
                }

            Where:
            - ``center`` is the absolute (x, y) pixel coordinate of the highlight peak.
            - ``offset`` is the (x, y) displacement from the tumour centroid before
              clamping to image bounds.
            - ``sigma`` is the Gaussian standard deviation in pixels (controls spread).
            - ``intensity`` is the maximum blend percentage toward white (0–100).
    """

    def __init__(self, image, mask, config):
        """Initialise the specular highlight datastructure.

        In practice this is never called directly; ``FailureRecreation.__init__``
        sets up all shared attributes. The signature is preserved to document the
        expected per-image state.

        Args:
            image (numpy.ndarray): BGR image, shape (H, W, 3), dtype uint8.
            mask (numpy.ndarray): Binary mask, shape (H, W), dtype uint8.
            config (AugConfig): Size-tier-specific configuration dataclass.
        """
        self.specular_data = {}

    # ------------------------------------------------------------------
    # Public-facing orchestrator
    # ------------------------------------------------------------------

    def generate_specular_highlights(self):
        """Orchestrate all steps needed to apply specular highlight augmentation.

        This is the "main" function that calls everything for one image:
        highlight-region generation → image application → YAML log. It resets the
        per-image datastructure before running so it is safe to call on multiple
        images sequentially.

        Side effects:
            - Populates ``self.specular_data[self.current_filename]``.
            - Modifies ``self.image`` in-place (R and G channels — and optionally B —
              blended toward 255 in highlight areas).
            - Writes a YAML log file to
              ``<output_path>/logs/<filename>_specular.yaml``.
        """
        self.specular_data[self.current_filename] = {"specular_regions": []}

        self._generate_specular_regions()
        self._apply_specular()
        self._save_specular_yml()

    # ------------------------------------------------------------------
    # Step 1: generate the specular highlight datastructure
    # ------------------------------------------------------------------

    def _generate_specular_regions(self):
        """Build the specular-region datastructure for the current image.

        For each highlight this method:

        1. Computes the tumour centroid via ``cv2.moments`` on ``self.mask``.
        2. Draws a random number of highlights in [1, ``n_highlights``].
        3. Assigns each highlight a random (x, y) offset from the centroid,
           a random Gaussian sigma, and a random intensity.

        Center coordinates are clamped to ``[0, w-1] × [0, h-1]`` so that every
        highlight produces some visible effect within the image. The stored
        ``offset`` is the pre-clamp random displacement; ``center`` is the
        post-clamp absolute coordinate.

        If the mask is empty (``m00 == 0``), the centroid falls back to the image
        centre so augmentation can still proceed.

        The results are appended to
        ``self.specular_data[self.current_filename]["specular_regions"]``.
        """
        cfg = self.aug_config.specular
        h, w = self.mask.shape[:2]

        # ------------------------------------------------------------------
        # Compute tumour centroid
        # ------------------------------------------------------------------
        M = cv2.moments(self.mask)
        if M["m00"] == 0:
            centroid_x, centroid_y = w // 2, h // 2
        else:
            centroid_x = int(M["m10"] / M["m00"])
            centroid_y = int(M["m01"] / M["m00"])

        # ------------------------------------------------------------------
        # Generate a random number of highlights
        # ------------------------------------------------------------------
        n = random.randint(1, cfg["n_highlights"])

        for _ in range(n):
            ox = random.randint(-cfg["offset_x"], cfg["offset_x"])
            oy = random.randint(-cfg["offset_y"], cfg["offset_y"])

            # Clamp to image bounds so the highlight is always at least partially visible
            cx = max(0, min(w - 1, centroid_x + ox))
            cy = max(0, min(h - 1, centroid_y + oy))

            # Sigma guard: ensure at least 1px width even if config is misconfigured
            sigma = max(1, random.randint(cfg["sigma_min"], cfg["sigma_max"]))
            intensity = random.randint(cfg["intensity_min"], cfg["intensity_max"])

            self.specular_data[self.current_filename]["specular_regions"].append(
                {
                    "center": (cx, cy),
                    "offset": (ox, oy),
                    "sigma": sigma,
                    "intensity": intensity,
                }
            )

    # ------------------------------------------------------------------
    # Step 2: apply the highlights to the image
    # ------------------------------------------------------------------

    def _apply_specular(self):
        """Apply each specular highlight to the image.

        For each highlight stored in ``self.specular_data``:

        1. A per-pixel Gaussian alpha map is computed from the highlight centre and
           sigma. Values lie in ``[0, intensity / 100]``.
        2. R (channel 2) and G (channel 1) are linearly blended toward 255 using
           ``pixel' = pixel * (1 - alpha) + 255 * alpha``.
        3. B (channel 0) is blended toward 255 only when ``model_type == 'rgb'``
           (where B is a colour channel). When ``model_type == 'rgd'``, B encodes
           depth and is left untouched.

        Multiple highlights stack by applying the blend sequentially on
        ``image_float``; overlapping regions converge further toward white with
        each additional highlight, which is physically correct.

        The coordinate grids are built once outside the per-highlight loop to
        avoid redundant allocation.

        Side effects:
            Modifies ``self.image`` in-place.
        """
        h, w = self.mask.shape[:2]
        image_float = self.image.astype(np.float32)

        # Build coordinate grids once for all highlights
        Y, X = np.mgrid[0:h, 0:w]

        for region in self.specular_data[self.current_filename]["specular_regions"]:
            cx, cy = region["center"]
            sigma = region["sigma"]
            intensity = region["intensity"]

            # Gaussian alpha map: shape (H, W), values in [0, intensity/100]
            alpha = (intensity / 100.0) * np.exp(
                -((X - cx) ** 2 + (Y - cy) ** 2) / (2.0 * sigma ** 2)
            )

            # Blend R (ch2) and G (ch1) toward white — always
            image_float[:, :, 1] = image_float[:, :, 1] * (1 - alpha) + 255 * alpha
            image_float[:, :, 2] = image_float[:, :, 2] * (1 - alpha) + 255 * alpha

            # B (ch0): blend toward white for RGB images (B is colour);
            # leave untouched for RGD images (B is depth)
            if self.model_type == "rgb":
                image_float[:, :, 0] = (
                    image_float[:, :, 0] * (1 - alpha) + 255 * alpha
                )

        self.image = np.clip(image_float, 0, 255).astype(np.uint8)

    # ------------------------------------------------------------------
    # Step 3: save the YAML log
    # ------------------------------------------------------------------

    def _save_specular_yml(self):
        """Save all randomly generated specular parameters to a YAML log file.

        Creates a YAML file at ``<output_path>/logs/<filename>_specular.yaml``
        that records every random value used during the specular augmentation of
        the current image. This allows any specific augmented image to be
        reproduced exactly.

        Tuples are converted to lists before dumping so the output contains plain
        YAML sequences rather than ``!!python/tuple`` tags.

        The output directory is created automatically if it does not exist.

        Side effects:
            Writes a YAML file to disk.
        """
        log_dir = os.path.join(self.output_path, "logs")
        os.makedirs(log_dir, exist_ok=True)

        log_path = os.path.join(log_dir, f"{self.current_filename}_specular.yaml")

        # Build a YAML-serialisable copy: tuples → lists (matches _red.py pattern)
        serialisable = {
            "specular_regions": [
                {
                    "center": list(r["center"]),
                    "offset": list(r["offset"]),
                    "sigma": r["sigma"],
                    "intensity": r["intensity"],
                }
                for r in self.specular_data[self.current_filename]["specular_regions"]
            ]
        }

        with open(log_path, "w") as f:
            yaml.dump(
                {self.current_filename: serialisable},
                f,
                default_flow_style=False,
                sort_keys=False,
            )
