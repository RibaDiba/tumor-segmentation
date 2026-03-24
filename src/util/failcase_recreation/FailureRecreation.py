import json
import os

import cv2
import numpy as np
from dataclasses import dataclass
from detectron2.data import DatasetCatalog
from detectron2.structures import BoxMode

from _red import RedRegionMixin
from _shadow import ShadowMixin


@dataclass
class AugConfig:
    """Holds the size-tier-specific augmentation parameters for one image.

    This dataclass is produced by ``FailureRecreation._extract_values`` and
    is passed implicitly to ``RedRegionMixin`` and ``ShadowMixin`` via
    ``self.aug_config``.

    Attributes:
        tier (str): The size tier assigned to this image: ``'small'``,
            ``'medium'``, or ``'large'``. Determined by comparing the tumour
            bounding-box width against the thresholds in ``config.yaml``.
        options (dict): The ``options`` block from ``config.yaml``, containing
            flags such as ``red_region``, ``shadows``, ``test_mode``, and
            ``output_path``.
        red_region (dict): The ``red_region.<tier>`` block from ``config.yaml``,
            containing all circle-generation parameters for this tier.
        shadow (dict): The ``shadow.<tier>`` block from ``config.yaml``,
            containing all shadow-generation parameters for this tier.
    """

    tier: str
    options: dict
    red_region: dict
    shadow: dict


class FailureRecreation(RedRegionMixin, ShadowMixin):
    """Generates synthetic adversarial augmentations of the test dataset.

    The overall goal is to produce a new COCO JSON dataset whose images have
    been modified to include colour anomalies at the tumour border. Two types
    of augmentation are supported:

    - **Red regions**: Patches of increased red-channel intensity at the
      tumour border, simulating vascular or skin-irritation discoloration.
    - **Shadows**: Arcs of darkened pixels (R and G channels reduced) at the
      tumour border, simulating shadow artefacts from lighting conditions.

    Both augmentation types leave the blue channel untouched. In RGD images the
    blue channel encodes depth information, so the RGD model retains its depth
    signal and should continue to segment correctly — whereas an RGB-only model
    may be misled by the colour changes.

    The class inherits from ``RedRegionMixin`` and ``ShadowMixin``. All shared
    per-image state (``self.image``, ``self.mask``, ``self.aug_config``,
    ``self.current_filename``) is set here in ``image_recreation`` or
    ``image_recreation_test`` before any mixin method is called.

    Args:
        cfg: Detectron2 configuration object (``CfgNode``). Used to locate the
            registered test dataset via ``DatasetCatalog``.
        output_path (str): Path to the directory where augmented images, the
            output COCO JSON, and per-image YAML logs will be written.
        yaml_config (dict): Parsed contents of ``config.yaml`` (typically loaded
            with ``yaml.safe_load``).
        image (numpy.ndarray, optional): Single BGR image for test-mode runs.
            Shape (H, W, 3), dtype uint8. Ignored when ``test_mode`` is False.
        mask (numpy.ndarray, optional): Single binary mask for test-mode runs.
            Shape (H, W), dtype uint8 with values 0 or 255. Ignored when
            ``test_mode`` is False.

    Attributes:
        cfg: Detectron2 config node.
        output_path (str): Root output directory.
        yaml_config (dict): Full parsed YAML config.
        options (dict): ``yaml_config['options']`` shortcut.
        size_thresholds (dict): ``yaml_config['size_thresholds']`` shortcut.
        image (numpy.ndarray or None): Current image being processed.
        mask (numpy.ndarray or None): Current binary mask.
        aug_config (AugConfig or None): Size-tier config for the current image.
        circle_data (dict): Accumulated red-region parameters keyed by filename.
        shadow_data (dict): Accumulated shadow parameters keyed by filename.
        current_filename (str or None): Stem of the filename being processed.
    """

    def __init__(
        self,
        cfg,
        output_path,
        yaml_config,
        image=None,        # optional argument
        mask=None,         # optional argument
        dataset_name="my_dataset_test",
    ):
        """Initialise FailureRecreation and all shared mixin state.

        Stores all constructor arguments, extracts the size-independent config
        sections from ``yaml_config``, initialises the per-image state variables
        used by the mixin methods, and ensures the output directory exists.

        Note:
            The mixin ``__init__`` methods (``RedRegionMixin.__init__`` and
            ``ShadowMixin.__init__``) are intentionally not called here via MRO
            because their signatures require ``image``, ``mask``, and ``config``
            arguments that are only meaningful per-image. The data structures
            they would initialise (``circle_data``, ``shadow_data``) are instead
            initialised directly below.

        Args:
            cfg: Detectron2 ``CfgNode`` configuration object.
            output_path (str): Directory for all output files.
            yaml_config (dict): Parsed ``config.yaml`` contents.
            image (numpy.ndarray, optional): BGR image for single-image test mode.
            mask (numpy.ndarray, optional): Binary mask for single-image test mode.
        """
        self.cfg = cfg
        self.output_path = output_path
        self.yaml_config = yaml_config
        self.dataset_name = dataset_name

        # Extract some values from the yaml that are not determined by image size
        # (e.g. the options flags and size thresholds)
        self.options = yaml_config["options"]
        self.size_thresholds = yaml_config["size_thresholds"]

        # Make sure the yml info is in a datastruct so that it can be passed down.
        # Per-image AugConfig is built by _extract_values and stored here.
        self.aug_config = None

        # Optional image / mask for single-image test mode
        self.image = image
        self.mask = mask

        # Shared state initialised here so mixin methods can access them
        self.circle_data = {}
        self.shadow_data = {}
        self.current_filename = None

        # Ensure the output tree exists
        os.makedirs(output_path, exist_ok=True)
        os.makedirs(os.path.join(output_path, "images"), exist_ok=True)
        os.makedirs(os.path.join(output_path, "logs"), exist_ok=True)

    # ------------------------------------------------------------------
    # Configuration extraction
    # ------------------------------------------------------------------

    def _extract_values(self, image, mask):
        """Determine augmentation parameters for a single image–mask pair.

        Reads size-independent values from the YAML config (``self.options``)
        and then derives size-dependent values by inspecting the mask geometry:
        the bounding-box width of the tumour contour is compared against the
        ``size_thresholds`` from the config to select the correct parameter
        tier (``'small'``, ``'medium'``, or ``'large'``).

        Note:
            The result must be kept in a dataclass (``AugConfig``) so that it
            can be passed down cleanly to the mixin methods via ``self.aug_config``.

        Args:
            image (numpy.ndarray): BGR image, shape (H, W, 3), dtype uint8.
                Not used in this method but included for symmetry with the
                mask-inspection step.
            mask (numpy.ndarray): Binary tumour mask, shape (H, W), dtype uint8
                with values 0 (background) or 255 (tumour).

        Returns:
            AugConfig: Dataclass containing the tier label, options block, and
            the tier-specific ``red_region`` and ``shadow`` parameter dicts.

        Raises:
            ValueError: If no contour is found in the mask (empty mask).
        """
        # Get the properties of the mask and set the values
        # Ensure mask is single-channel
        if mask.ndim == 3:
            mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        else:
            mask_gray = mask

        contours, _ = cv2.findContours(
            mask_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            raise ValueError(
                "Mask contains no contour. Cannot determine tumour size tier."
            )

        largest = max(contours, key=cv2.contourArea)
        _x, _y, bbox_width, _h = cv2.boundingRect(largest)

        # Determine the tier based on bbox_width
        small_thresh = self.size_thresholds["small"]
        medium_thresh = self.size_thresholds["medium"]

        if bbox_width <= small_thresh:
            tier = "small"
        elif bbox_width <= medium_thresh:
            tier = "medium"
        else:
            tier = "large"

        return AugConfig(
            tier=tier,
            options=self.options,
            red_region=self.yaml_config["red_region"][tier],
            shadow=self.yaml_config["shadow"][tier],
        )

    # ------------------------------------------------------------------
    # Main entry points
    # ------------------------------------------------------------------

    def image_recreation(self):
        """Run augmentation over the full test dataset or a single test image.

        Checks whether ``test_mode`` is active in the config options. If so,
        delegates to ``image_recreation_test`` (single-image debugging path).
        Otherwise, iterates every image registered under ``"my_dataset_test"``
        in Detectron2's ``DatasetCatalog``, applies the configured augmentations,
        and writes all outputs to ``self.output_path``.

        The output includes:

        - Augmented JPEG images in ``<output_path>/images/``.
        - Per-image YAML logs in ``<output_path>/logs/``.
        - A COCO JSON file at ``<output_path>/augmented_test.json`` whose
          annotation entries are identical to the originals (the tumour boundary
          is unchanged) but whose ``file_name`` fields point to the augmented
          images.

        Raises:
            RuntimeError: If a test image or its corresponding mask cannot be
                read from disk.
        """
        # If test mode, do the other thing (single-image debug path)
        if self.options["test_mode"]:
            return self.image_recreation_test()

        # Extract test dataset from the registered Detectron2 catalog
        test_dicts = DatasetCatalog.get(self.dataset_name)

        # Build the COCO JSON skeleton for the augmented dataset
        coco_output = {
            "info": {},
            "licenses": [],
            "categories": [{"id": 1, "name": "Tumor", "supercategory": "Tumor"}],
            "images": [],
            "annotations": [],
        }

        # Do the loop
        ann_id = 0
        for dataset_dict in test_dicts:
            image_path = dataset_dict["file_name"]

            # Derive the mask path from the image path:
            # .../test/images/<name>.jpg  →  .../test/masks/Tumor/<name>.png
            mask_path = image_path.replace("/images/", "/masks/Tumor/")
            mask_path = os.path.splitext(mask_path)[0] + ".png"

            image = cv2.imread(image_path)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            if image is None:
                raise RuntimeError(f"Could not read image: {image_path}")
            if mask is None:
                raise RuntimeError(f"Could not read mask: {mask_path}")

            # Determine size tier and load appropriate config block
            self.aug_config = self._extract_values(image, mask)
            self.image = image.copy()
            self.mask = mask
            self.current_filename = os.path.splitext(
                os.path.basename(image_path)
            )[0]

            # Apply augmentations according to the options flags
            if self.options["red_region"]:
                self._create_red_regions()
            if self.options["shadows"]:
                self.generate_shadows()

            # Save the augmented image
            out_img_path = os.path.join(
                self.output_path, "images", os.path.basename(image_path)
            )
            cv2.imwrite(out_img_path, self.image)

            # Add image entry to COCO JSON (same id/dimensions, updated file_name)
            coco_output["images"].append(
                {
                    "id": dataset_dict["image_id"],
                    "width": dataset_dict["width"],
                    "height": dataset_dict["height"],
                    "file_name": os.path.basename(image_path),
                }
            )

            # Copy all annotation entries unchanged — the tumour boundary is
            # not modified by the augmentations, only the pixel colours.
            for ann in dataset_dict.get("annotations", []):
                bbox = ann["bbox"]
                if ann.get("bbox_mode") == BoxMode.XYXY_ABS:
                    x1, y1, x2, y2 = bbox
                    bbox = [x1, y1, x2 - x1, y2 - y1]

                coco_output["annotations"].append(
                    {
                        "id": ann_id,
                        "image_id": dataset_dict["image_id"],
                        "category_id": 1,  # D2 stores 0-indexed; COCO JSON needs original id (Tumor=1)
                        "bbox": bbox,
                        "area": ann.get("area", bbox[2] * bbox[3]),
                        "segmentation": ann["segmentation"],
                        "iscrowd": ann.get("iscrowd", 0),
                    }
                )
                ann_id += 1

        # Write the final COCO JSON
        json_path = os.path.join(self.output_path, "augmented_test.json")
        with open(json_path, "w") as f:
            json.dump(coco_output, f)

    def image_recreation_test(self):
        """Run augmentation on the single image stored in ``self.image`` and ``self.mask``.

        This does the same thing as ``image_recreation`` but operates on only one
        image — the one passed to ``__init__`` as the optional ``image`` and
        ``mask`` arguments. This is mainly for debugging purposes: it lets you
        visually verify the augmentation output on a known sample before running
        the full test set.

        A YAML log is written for the augmented image. No COCO JSON is produced
        since there is no full dataset to annotate.

        The augmented image is written to
        ``<output_path>/images/test_<current_filename>.jpg``.

        Raises:
            ValueError: If ``self.image`` or ``self.mask`` is ``None``.
        """
        if self.image is None or self.mask is None:
            raise ValueError(
                "image_recreation_test requires both 'image' and 'mask' to be "
                "set on the FailureRecreation instance. Pass them to __init__ "
                "or assign them directly before calling this method."
            )

        # Derive a filename stem if not already set
        if self.current_filename is None:
            self.current_filename = "test_image"

        self.aug_config = self._extract_values(self.image, self.mask)

        # Work on a copy so the original self.image is preserved for re-runs
        self.image = self.image.copy()

        if self.options["red_region"]:
            self._create_red_regions()
        if self.options["shadows"]:
            self.generate_shadows()

        # Save the single augmented image
        out_img_path = os.path.join(
            self.output_path, "images", f"{self.current_filename}.jpg"
        )
        cv2.imwrite(out_img_path, self.image)
