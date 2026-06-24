import numpy as np, re, shutil
from typing import List, Tuple
import sys
import os

_this_dir = os.path.dirname(os.path.abspath(__file__))
_util_dir = os.path.abspath(os.path.join(_this_dir, "../.."))
if _util_dir not in sys.path:
    sys.path.insert(0, _util_dir)

from preprocessing.preprocessing_functions import *
from preprocessing.process_coco_json import *
from detectron2.data import MetadataCatalog
from detectron2.data.datasets import register_coco_instances

from ._preprocessing import PreprocessingMixin
from ._splitting import SplittingMixin
from ._caching import CachingMixin
from ._coco import CocoMixin
from ._subset import SubsetMixin
from ._utils import UtilsMixin


class Dataset(
    PreprocessingMixin, SplittingMixin, CachingMixin, CocoMixin, SubsetMixin, UtilsMixin
):

    def __init__(self, data_path: str):
        self.data_path = data_path
        print("Init Dataset")

        project_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../../..")
        )

        # Root under which every modality/split directory is written. Kept as an
        # attribute (rather than hard-coded inline) so a subclass can redirect it
        # per cross-validation fold; the default reproduces the historical path.
        self.processed_root = os.path.join(project_root, "data/processed_data")
        self._set_dirs()

    def _set_dirs(self) -> None:
        """(Re)compute every modality/split directory from self.processed_root.

        Call after reassigning self.processed_root (e.g. when switching folds).
        Attribute names are unchanged so cache/coco/register logic keeps working.
        """
        # modalities that carry a binary-mask directory
        for _variant in ("rgb", "depth", "rgd"):
            for _split in ("train", "val", "test"):
                setattr(
                    self,
                    f"{_variant}_{_split}_mask_dir",
                    os.path.join(self.processed_root, _variant, _split, "masks/Tumor"),
                )

        # image dirs for every modality (rgbd_early has no mask dir attr)
        for _variant in ("rgb", "depth", "rgd", "rgbd_early"):
            for _split in ("train", "val", "test"):
                setattr(
                    self,
                    f"{_variant}_{_split}_dir",
                    os.path.join(self.processed_root, _variant, _split, "images/"),
                )

    # image processing functions
    read_images_to_array = read_images_to_array
    crop_raw_images = crop_raw_images
    crop_masks = crop_masks
    add_padding = add_padding
    zoom_at = zoom_at
    create_binary_masks = create_binary_masks
    crop_images = crop_images
    crop_images_offset = crop_images_offset
    translate_images = translate_images  # util?
    read_neg_images = read_neg_images
    create_neg_masks = create_neg_masks
    infuse_depth_into_blue_channel = (
        infuse_depth_into_blue_channel  # TODO: still has to be worked on
    )
    read_contours_array_depth = read_contours_array_depth
    read_to_array_post = read_to_array_post
    read_folder_to_array = read_folder_to_array
    correct_binary_masks = correct_binary_masks

    # coco_json methods
    images_annotations_info = images_annotations_info
    process_masks = process_masks

    # methods for data filtering
    filter_subset_in_folder = filter_subset_in_folder
    remove_files_in_dir = remove_files_in_dir
    save_subset_array = save_subset_array
    subset_automation = subset_automation
