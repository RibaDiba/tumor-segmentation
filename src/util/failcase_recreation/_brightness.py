import os
import random

import cv2
import numpy as np
import yaml


class BrightnessMixin:
    """Mixin class that adds brightness variation augmentation to FailureRecreation.

    Applies a uniform brightness reduction image-wide by multiplying the colour
    channels by a random scalar factor drawn from a configurable range. This
    simulates real-world dim or inconsistent lighting conditions — a genuine
    confound for RGB-only models, because the model loses colour contrast across
    the entire image.

    Unlike the other augmentation types in this framework (red regions, shadows,
    specular highlights), brightness variation is **scene-level**, not
    tumour-size-dependent. The factor is drawn from a flat config block with no
    small/medium/large tiers.

    The key argument for depth: the 3D scanner measures surface geometry using
    structured light or time-of-flight — both are independent of ambient
    lighting. So the depth channel (B in RGD images) retains full signal even
    when the room is dark, while the RGB colour channels degrade proportionally.

    The blue channel behaviour depends on ``model_type``:

    - ``'rgb'``: blue is a colour channel, so it is multiplied by the same
      factor as R and G — all three channels are uniformly dimmed.
    - ``'rgd'``: blue encodes depth from the 3D scanner. It is **left
      untouched**, preserving the depth signal under simulated low-light.

    This class is not intended to be instantiated directly. It is used as a
    mixin via multiple inheritance in ``FailureRecreation``. The following
    instance attributes must be present on ``self`` before any method is called:

    Attributes:
        image (numpy.ndarray): Current BGR image being processed, shape (H, W, 3),
            dtype uint8. Modified in-place by ``_apply_brightness``.
        aug_config (AugConfig): Dataclass holding the (flat, non-tiered)
            brightness parameters loaded from ``config.yaml``.
        current_filename (str): Stem of the current image filename (no extension).
            Used as the top-level key in ``brightness_data`` and as the YAML log
            name.
        output_path (str): Root directory where augmented outputs are written.
            YAML logs are saved to ``<output_path>/logs/``.
        model_type (str): Either ``'rgb'`` or ``'rgd'``. Controls whether the
            blue channel is dimmed (``'rgb'``) or preserved (``'rgd'``).
        brightness_data (dict): Accumulator for the randomly generated factor.
            Reset at the start of each image's processing pass. Structure::

                {
                    "<filename>": {
                        "factor": float   # multiplier applied to colour channels
                    }
                }

            Where ``factor`` is in ``[factor_min, factor_max]`` from the config.
    """

    def __init__(self, image, mask, config):
        """Initialise the brightness datastructure.

        In practice this is never called directly; ``FailureRecreation.__init__``
        sets up all shared attributes. The signature is preserved to document the
        expected per-image state.

        Args:
            image (numpy.ndarray): BGR image, shape (H, W, 3), dtype uint8.
            mask (numpy.ndarray): Binary mask, shape (H, W), dtype uint8.
            config (AugConfig): Configuration dataclass (brightness block is flat).
        """
        self.brightness_data = {}

    # ------------------------------------------------------------------
    # Public-facing orchestrator
    # ------------------------------------------------------------------

    def apply_brightness_variation(self):
        """Orchestrate all steps needed to apply brightness variation.

        This is the "main" function that calls everything for one image:
        factor generation → image application → YAML log. It resets the
        per-image datastructure before running so it is safe to call on
        multiple images sequentially.

        Side effects:
            - Populates ``self.brightness_data[self.current_filename]``.
            - Modifies ``self.image`` in-place (R and G channels — and
              optionally B — multiplied by the brightness factor).
            - Writes a YAML log file to
              ``<output_path>/logs/<filename>_brightness.yaml``.
        """
        self.brightness_data[self.current_filename] = {}

        self._generate_brightness_params()
        self._apply_brightness()
        self._save_brightness_yml()

    # ------------------------------------------------------------------
    # Step 1: generate the brightness factor
    # ------------------------------------------------------------------

    def _generate_brightness_params(self):
        """Draw a random brightness multiplier from the config range.

        Reads ``factor_min`` and ``factor_max`` from ``self.aug_config.brightness``
        and samples a single float uniformly at random. The value is rounded to
        four decimal places so the YAML log is human-readable.

        Unlike the other augmentation types, no mask or contour is needed —
        brightness is applied uniformly across the entire image.

        The result is stored in
        ``self.brightness_data[self.current_filename]["factor"]``.
        """
        cfg = self.aug_config.brightness
        factor = round(random.uniform(cfg["factor_min"], cfg["factor_max"]), 4)
        self.brightness_data[self.current_filename]["factor"] = factor

    # ------------------------------------------------------------------
    # Step 2: apply brightness reduction to the image
    # ------------------------------------------------------------------

    def _apply_brightness(self):
        """Apply the brightness factor to the image channels.

        Multiplies R (channel 2) and G (channel 1) by the stored factor.
        B (channel 0) is also multiplied when ``model_type == 'rgb'`` (where B
        is a colour channel); it is left untouched when ``model_type == 'rgd'``
        (where B is depth).

        No loop is required — the factor is a scalar applied to all pixels at
        once via numpy broadcasting.

        Side effects:
            Modifies ``self.image`` in-place.
        """
        factor = self.brightness_data[self.current_filename]["factor"]
        image_float = self.image.astype(np.float32)

        # Always dim R (ch2) and G (ch1) — these are always colour channels
        image_float[:, :, 2] *= factor
        image_float[:, :, 1] *= factor

        # B (ch0): dim for RGB images (B is colour); preserve for RGD (B is depth)
        if self.model_type == "rgb":
            image_float[:, :, 0] *= factor

        self.image = np.clip(image_float, 0, 255).astype(np.uint8)

    # ------------------------------------------------------------------
    # Step 3: save the YAML log
    # ------------------------------------------------------------------

    def _save_brightness_yml(self):
        """Save the brightness factor to a YAML log file.

        Creates a YAML file at ``<output_path>/logs/<filename>_brightness.yaml``
        recording the factor used for this image. This allows the exact
        augmentation to be reproduced. The factor is a plain Python float so
        no tuple-to-list conversion is needed.

        The output directory is created automatically if it does not exist.

        Side effects:
            Writes a YAML file to disk.
        """
        log_dir = os.path.join(self.output_path, "logs")
        os.makedirs(log_dir, exist_ok=True)

        log_path = os.path.join(log_dir, f"{self.current_filename}_brightness.yaml")

        with open(log_path, "w") as f:
            yaml.dump(
                {self.current_filename: self.brightness_data[self.current_filename]},
                f,
                default_flow_style=False,
                sort_keys=False,
            )
