import json
import os
import sys

import cv2
import numpy as np
from dataclasses import dataclass
from detectron2.data import DatasetCatalog
from detectron2.structures import BoxMode

from _red import RedRegionMixin
from _shadow import ShadowMixin
from _specular import SpecularMixin
from _brightness import BrightnessMixin
from _necrotic import NecroticMixin

_util_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _util_dir not in sys.path:
    sys.path.insert(0, _util_dir)
from paths import PROCESSED_DATA_DIR


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
    specular: dict
    brightness: dict
    necrotic: dict


class FailureRecreation(RedRegionMixin, ShadowMixin, SpecularMixin, BrightnessMixin, NecroticMixin):
    """Generates synthetic adversarial augmentations of the test dataset.

    The overall goal is to produce a new COCO JSON dataset whose images have
    been modified to include colour anomalies at the tumour border. Two types
    of augmentation are supported:

    - **Red regions**: Patches of increased red-channel intensity at the
      tumour border, simulating vascular or skin-irritation discoloration.
    - **Shadows**: Arcs of darkened pixels (R and G channels reduced) at the
      tumour border, simulating shadow artefacts from lighting conditions.

    The blue channel behaviour is controlled by ``options['model_type']``:
    when set to ``'rgb'``, the blue channel is always preserved so the RGD model
    retains its depth signal; when set to ``'rgd'``, shadows also darken the blue
    channel to stress-test the RGD model under full channel degradation.

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
        model_type (str): Modality being processed — ``'rgb'`` or ``'rgd'``.
            Controls which channels shadow augmentation darkens: ``'rgb'``
            darkens R and G only (B/depth preserved); ``'rgd'`` darkens all
            three channels. Defaults to ``'rgb'``.

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
        model_type="rgb",
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
        self.model_type = model_type

        # Extract some values from the yaml that are not determined by image size
        # (e.g. the options flags and size thresholds)
        self.options = yaml_config["options"]
        self.size_thresholds = yaml_config["size_thresholds"]

        # Make sure the yml info is in a datastruct so that it can be passed down.
        # Per-image AugConfig is built by _extract_values and stored here.
        self.aug_config = None

        # Load test_mode images from config if test_mode is enabled
        if self.options.get("test_mode") and "test_image_config" in self.options:
            test_config = self.options["test_image_config"]
            # Paths in config.yaml are relative to data/processed_data/ so they
            # work regardless of repo install location.
            image_path = str(PROCESSED_DATA_DIR / test_config["image_path"])
            mask_path = str(PROCESSED_DATA_DIR / test_config["mask_path"])
            output_name = test_config["output_name"]

            # Infer model_type from the image path directory structure
            # (overrides the constructor default so test mode is self-describing)
            self.model_type = FailureRecreation._infer_model_type(image_path)

            # Validate paths exist
            if not os.path.exists(image_path):
                raise ValueError(f"Image path does not exist: {image_path}")
            if not os.path.exists(mask_path):
                raise ValueError(f"Mask path does not exist: {mask_path}")

            # Load images from disk (clone them to preserve originals)
            loaded_image = cv2.imread(image_path)
            loaded_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            if loaded_image is None:
                raise ValueError(f"Could not read image: {image_path}")
            if loaded_mask is None:
                raise ValueError(f"Could not read mask: {mask_path}")

            # Store cloned images (these will be modified during augmentation)
            self.image = loaded_image.copy()
            self.mask = loaded_mask.copy()
            self.current_filename = output_name
        else:
            # Optional image / mask for single-image test mode (backwards compatibility)
            self.image = image
            self.mask = mask
            self.current_filename = None

        # Shared state initialised here so mixin methods can access them
        self.circle_data = {}
        self.shadow_data = {}
        self.specular_data = {}
        self.brightness_data = {}
        self.necrotic_data = {}
        # Note: self.current_filename is set above (either from config in test_mode,
        # or None for backwards compatibility mode)

        # Ensure the output tree exists
        os.makedirs(output_path, exist_ok=True)
        os.makedirs(os.path.join(output_path, "images"), exist_ok=True)
        os.makedirs(os.path.join(output_path, "logs"), exist_ok=True)

    # ------------------------------------------------------------------
    # Configuration extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _infer_model_type(path):
        """Infer the model modality from a file path's directory components.

        Splits ``path`` on the OS separator and looks for a component that is
        exactly ``'rgb'``, ``'rgd'``, or ``'depth'``. The check is done on
        individual path components (not a substring search) so a filename that
        happens to contain ``'rgb'`` does not produce a false match.

        Args:
            path (str): Any path containing the modality as a directory name,
                e.g. ``.../processed_data/rgd/test/images/foo.jpg``.

        Returns:
            str: One of ``'rgb'``, ``'rgd'``, or ``'depth'``.

        Raises:
            ValueError: If no recognised modality directory is found in the
                path, meaning the caller must make the modality explicit.
        """
        # Normalise separators so the split works on both posix and windows paths
        components = path.replace("\\", "/").split("/")
        for part in components:
            if part in ("rgb", "rgd", "depth"):
                return part
        raise ValueError(
            f"Cannot determine model_type from path: '{path}'.\n"
            "The path must contain a directory component named exactly "
            "'rgb', 'rgd', or 'depth' (e.g. .../processed_data/rgd/test/images/)."
        )

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
            specular=self.yaml_config["specular"][tier],
            brightness=self.yaml_config["brightness"],
            necrotic=self.yaml_config["necrotic"][tier],
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
            if self.options.get("specular"):
                self.generate_specular_highlights()
            if self.options.get("brightness"):
                self.apply_brightness_variation()
            if self.options.get("necrotic"):
                self.generate_necrotic_regions()

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

    def image_recreation_test(self, image_path=None, mask_path=None, output_path=None):
        """Run augmentation on a single image.

        When ``image_path`` and ``mask_path`` are provided, the image and mask
        are loaded from those paths and the filename stem is derived from
        ``image_path``. When omitted, the method falls back to ``self.image``
        and ``self.mask`` (set during ``__init__`` from ``config.yaml`` or
        passed directly), preserving backwards-compatible behaviour.

        When ``output_path`` is provided, augmented images and YAML logs are
        written there instead of to the instance-level ``self.output_path``.

        A YAML log is written for the augmented image. No COCO JSON is produced
        since there is no full dataset to annotate.

        Args:
            image_path (str, optional): Path to the source image to augment.
                Must be provided together with ``mask_path``. If omitted,
                ``self.image`` is used.
            mask_path (str, optional): Path to the corresponding binary mask.
                Must be provided together with ``image_path``. If omitted,
                ``self.mask`` is used.
            output_path (str, optional): Directory for the augmented image and
                YAML logs. If omitted, ``self.output_path`` is used.

        Raises:
            ValueError: If ``image_path``/``mask_path`` point to unreadable
                files, or if no image/mask is available (neither argument nor
                ``self.image``/``self.mask`` are set).
        """
        # ------------------------------------------------------------------
        # Resolve image, mask, and filename
        # ------------------------------------------------------------------
        if image_path is not None and mask_path is not None:
            loaded_image = cv2.imread(image_path)
            loaded_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if loaded_image is None:
                raise ValueError(f"Could not read image: {image_path}")
            if loaded_mask is None:
                raise ValueError(f"Could not read mask: {mask_path}")
            image = loaded_image
            mask = loaded_mask
            filename = os.path.splitext(os.path.basename(image_path))[0]
        else:
            if self.image is None or self.mask is None:
                raise ValueError(
                    "image_recreation_test requires both 'image' and 'mask'. "
                    "Pass image_path and mask_path, or set them via __init__."
                )
            image = self.image
            mask = self.mask
            filename = self.current_filename if self.current_filename is not None else "test_image"

        # ------------------------------------------------------------------
        # Resolve output path and ensure directories exist
        # ------------------------------------------------------------------
        effective_out = output_path if output_path is not None else self.output_path
        os.makedirs(os.path.join(effective_out, "images"), exist_ok=True)
        os.makedirs(os.path.join(effective_out, "logs"), exist_ok=True)

        # ------------------------------------------------------------------
        # Set shared per-image state used by the mixin methods
        # ------------------------------------------------------------------
        self.current_filename = filename
        self.aug_config = self._extract_values(image, mask)
        self.image = image.copy()
        self.mask = mask

        # Temporarily redirect self.output_path so mixin YAML logs go to the
        # correct directory, then restore it in the finally block.
        original_output_path = self.output_path
        self.output_path = effective_out
        try:
            if self.options["red_region"]:
                self._create_red_regions()
            if self.options["shadows"]:
                self.generate_shadows()
            if self.options.get("specular"):
                self.generate_specular_highlights()
            if self.options.get("brightness"):
                self.apply_brightness_variation()
            if self.options.get("necrotic"):
                self.generate_necrotic_regions()
        finally:
            self.output_path = original_output_path

        # Save the single augmented image
        out_img_path = os.path.join(effective_out, "images", f"{filename}.jpg")
        cv2.imwrite(out_img_path, self.image)

    def augment_image(self, image_path, mask_path, output_path=None):
        """Augment a single image from disk, returning it or saving to a path.

        Args:
            image_path (str): Path to the source BGR image.
            mask_path (str): Path to the corresponding binary mask (grayscale).
            output_path (str, optional): If provided, the augmented image and
                YAML logs are written there (same layout as
                ``image_recreation_test``). If omitted, no files are written
                and the augmented image is returned as a numpy array instead.

        Returns:
            numpy.ndarray or None: The augmented BGR image (H, W, 3) uint8 if
            ``output_path`` is ``None``; otherwise ``None``.

        Raises:
            ValueError: If ``image_path`` or ``mask_path`` cannot be read.
        """
        loaded_image = cv2.imread(image_path)
        loaded_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if loaded_image is None:
            raise ValueError(f"Could not read image: {image_path}")
        if loaded_mask is None:
            raise ValueError(f"Could not read mask: {mask_path}")

        filename = os.path.splitext(os.path.basename(image_path))[0]

        # Infer model_type from the image path so each call is self-describing,
        # regardless of what was set at construction time
        self.model_type = FailureRecreation._infer_model_type(image_path)

        self.current_filename = filename
        self.aug_config = self._extract_values(loaded_image, loaded_mask)
        self.image = loaded_image.copy()
        self.mask = loaded_mask

        # When saving, redirect self.output_path so mixin YAML logs go there.
        # When returning, leave self.output_path unchanged (logs go to default).
        original_output_path = self.output_path
        if output_path is not None:
            os.makedirs(os.path.join(output_path, "images"), exist_ok=True)
            os.makedirs(os.path.join(output_path, "logs"), exist_ok=True)
            self.output_path = output_path

        try:
            if self.options["red_region"]:
                self._create_red_regions()
            if self.options["shadows"]:
                self.generate_shadows()
            if self.options.get("specular"):
                self.generate_specular_highlights()
            if self.options.get("brightness"):
                self.apply_brightness_variation()
            if self.options.get("necrotic"):
                self.generate_necrotic_regions()
        finally:
            self.output_path = original_output_path

        if output_path is not None:
            out_img_path = os.path.join(output_path, "images", f"{filename}.jpg")
            cv2.imwrite(out_img_path, self.image)
            return None

        return self.image
