"""
Early-fusion 4-channel RGBD utilities.

Pure building blocks (no modality branching): a DatasetMapper that loads the
paired ``<stem>.jpg`` (RGB) + ``<stem>_depth.npy`` (depth) written by
``_cache_rgbd_variant`` and stacks them into a (4, H, W) tensor, and a helper to
inflate a pretrained 3-channel stem conv1 to 4 channels.
"""

import copy
import os
import pickle

import cv2
import numpy as np
import torch

from detectron2.data import DatasetMapper, detection_utils as utils
from detectron2.data import transforms as T

# the depth channel is always the 4th index, everywhere
DEPTH_SUFFIX = "_depth.npy"


class RGBDDatasetMapper(DatasetMapper):
    """Stock ``DatasetMapper`` behaviour, but the input image is a 4-channel
    (H, W, 4) RGB+depth array instead of a 3-channel read.

    The COCO ``file_name`` points at the cached ``.jpg``; the depth sits next to
    it as ``<stem>_depth.npy``. Augmentations (resize/flip) operate on the array
    generically, so the depth channel rides along spatially aligned with RGB.
    """

    def _read_rgbd(self, file_name):
        # RGB exactly as the stock mapper would read it (honours INPUT.FORMAT)
        rgb = utils.read_image(file_name, format=self.image_format)
        depth_path = os.path.splitext(file_name)[0] + DEPTH_SUFFIX
        depth = np.load(depth_path).astype(np.float32)

        h, w = rgb.shape[:2]
        if depth.shape[:2] != (h, w):
            # defensive: rgbd_early depth already matches the RGB size (same crop
            # pipeline), so this normally no-ops; resize if a variant ever differs
            depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)

        return np.dstack([rgb.astype(np.float32), depth[..., None]])

    def __call__(self, dataset_dict):
        # mirrors detectron2's DatasetMapper.__call__, swapping only the image read
        dataset_dict = copy.deepcopy(dataset_dict)
        image = self._read_rgbd(dataset_dict["file_name"])
        utils.check_image_size(dataset_dict, image)

        aug_input = T.AugInput(image)
        transforms = self.augmentations(aug_input)
        image = aug_input.image

        image_shape = image.shape[:2]  # h, w
        dataset_dict["image"] = torch.as_tensor(
            np.ascontiguousarray(image.transpose(2, 0, 1))
        )

        if not self.is_train:
            dataset_dict.pop("annotations", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            self._transform_annotations(dataset_dict, transforms, image_shape)

        return dataset_dict


def inflate_conv1(model, weights_path):
    """Widen the backbone stem conv1 from 3 to 4 input channels in place.

    Copies the pretrained RGB filters into channels [:, :3] and initialises the
    new depth channel [:, 3:4] as the mean of those three (I3D-style inflation).
    ``model`` is expected to already have a 4-channel conv1 (built that way from a
    4-entry ``MODEL.PIXEL_MEAN``); we only fill its weights.
    """
    with open(weights_path, "rb") as f:
        ckpt = pickle.load(f, encoding="latin1")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

    key = next((k for k in state if k.endswith("stem.conv1.weight")), None)
    if key is None:
        raise KeyError(
            f"could not find a 'stem.conv1.weight' entry in {weights_path}"
        )

    pretrained = np.asarray(state[key], dtype=np.float32)  # (64, 3, 7, 7)
    conv1 = model.backbone.bottom_up.stem.conv1.weight
    if conv1.shape[1] != 4:
        raise ValueError(
            f"expected a 4-channel conv1 (got {tuple(conv1.shape)}); set a "
            "4-entry MODEL.PIXEL_MEAN/PIXEL_STD in the rgbd config"
        )

    with torch.no_grad():
        w = torch.from_numpy(pretrained).to(conv1.device, conv1.dtype)
        conv1[:, :3].copy_(w)
        conv1[:, 3:4].copy_(w.mean(dim=1, keepdim=True))

    return model
