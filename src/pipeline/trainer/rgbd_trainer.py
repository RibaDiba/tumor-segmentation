"""
Trainer for the early-fusion 4-channel RGBD modality (rgbd_early).

Subclasses the standard ``Trainer`` and overrides only the 4-channel-specific
pieces: the data loaders use ``RGBDDatasetMapper``, ``build_model`` inflates the
stem conv1 from 3 to 4 input channels, and ``_eval_mapper`` hands the same
4-channel mapper to the evaluation hooks. Everything else (hooks, evaluator,
writers) is inherited unchanged.

``train.py`` selects this class based on ``--modality``; the base ``Trainer``
stays behaviourally identical for rgb/depth/rgd. (rgbd_late currently shares this
trainer until its dedicated late-fusion implementation lands.)
"""

from detectron2.data import (
    build_detection_train_loader,
    build_detection_test_loader,
)
from detectron2.modeling import build_model

from .trainer import Trainer
from .rgbd_mapper import RGBDDatasetMapper, inflate_conv1


class RGBDTrainer(Trainer):
    @classmethod
    def build_train_loader(cls, cfg):
        mapper = RGBDDatasetMapper(cfg, is_train=True)
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        mapper = RGBDDatasetMapper(cfg, is_train=False)
        return build_detection_test_loader(cfg, dataset_name, mapper=mapper)

    @classmethod
    def build_model(cls, cfg):
        # conv1 is already 4-channel because the rgbd config sets a 4-entry
        # MODEL.PIXEL_MEAN; we only fill its weights from the pretrained backbone
        model = build_model(cfg)
        inflate_conv1(model, cfg.MODEL.WEIGHTS)
        return model

    def _eval_mapper(self, is_train=False):
        return RGBDDatasetMapper(self.cfg, is_train=is_train)
