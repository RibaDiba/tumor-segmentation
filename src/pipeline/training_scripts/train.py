"""
this file is the python exec to start training
before this file is ran there should be some checks to make sure that the data is good
there are separate tests created specifically for this workflow in tests
"""

import argparse
import datetime
import os
import sys
from pathlib import Path

# Make project imports resolvable regardless of CWD / PYTHONPATH:
# preprocessing + paths live under src/util, pipeline lives under src.
_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.abspath(os.path.join(_current_dir, "../../.."))
for _p in (os.path.join(_project_root, "src", "util"), os.path.join(_project_root, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from detectron2 import model_zoo
from detectron2.config import CfgNode, get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.logger import setup_logger

from preprocessing.TumorDataset.tumor_dataset import Dataset
from pipeline.trainer.trainer import Trainer
from paths import PROJECT_ROOT, MODELS_DIR

CONFIGS_DIR = PROJECT_ROOT / "configs"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train a Detectron2 tumor segmentation model")
    p.add_argument("--modality",
                   choices=("rgb", "depth", "rgd", "rgbd_contour", "rgbd_rawgrid"),
                   required=True,
                   help="Image modality to train on")
    p.add_argument("--model-name", dest="model_name", required=True,
                   help="Run identifier")
    p.add_argument("--model-type", dest="model_type", required=True,
                   help="Experiment group / subdirectory")
    p.add_argument("--config-dir", dest="config_dir", type=Path, default=CONFIGS_DIR,
                   help="Directory containing base.yaml and <modality>.yaml")
    p.add_argument("--split-cache", dest="split_cache", action="store_true",
                   help="Preprocess, split, and cache data (first run only)")
    p.add_argument("--augmentations", action="store_true",
                   help="Enable augmentation pipeline during preprocessing (h/v flips + rotation)")
    p.add_argument("--rotate_degrees", type=int, default=0, help="rotation degree range")
    p.add_argument(
        "opts",
        nargs=argparse.REMAINDER,
        help='Config overrides, e.g. "SOLVER.MAX_ITER 10000 SOLVER.BASE_LR 0.001"',
    )
    return p


def setup_cfg(args: argparse.Namespace) -> CfgNode:
    cfg = get_cfg()
    cfg.MODELNAME = ""
    cfg.MODELTYPE = ""
    cfg.MODALITY = ""

    cfg.merge_from_file(
        model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")
    )
    cfg.merge_from_file(str(args.config_dir / f"{args.modality}.yaml"))

    cfg.MODELNAME = args.model_name
    cfg.MODELTYPE = args.model_type
    cfg.MODALITY = args.modality

    run_id = datetime.datetime.now().strftime("run_%Y%m%d-%H%M")
    run_dir = MODELS_DIR / args.model_type / args.model_name / run_id
    cfg.OUTPUT_DIR = str(run_dir)

    if args.opts:
        cfg.merge_from_list(args.opts)

    cfg.freeze()
    return cfg


def snapshot_run(cfg: CfgNode) -> None:
    run_dir = Path(cfg.OUTPUT_DIR)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config_used.yaml").write_text(cfg.dump())
    latest = run_dir.parent / "latest"
    if latest.is_symlink() or latest.exists():
        latest.unlink()
    latest.symlink_to(run_dir.name)


def main(argv: list[str] | None = None) -> None:
    setup_logger()
    args = build_parser().parse_args(argv)

    d = Dataset(data_path=str(PROJECT_ROOT / "data/huggingface-repo/useable_data"))
    if args.split_cache:
        if args.augmentations:
            d.preprocess_augs(rotate_degrees=args.rotate_degrees)
        else:
            d.preprocess_images()
        d.split_train_val_test(70, 10, 20)
        d.cache_data()

    d.convert_binary_to_coco()
    d.register_instances(**{args.modality: True})

    MetadataCatalog.get("my_dataset_train")
    DatasetCatalog.get("my_dataset_train")
    MetadataCatalog.get("my_dataset_val")
    DatasetCatalog.get("my_dataset_val")
    MetadataCatalog.get("my_dataset_test")
    DatasetCatalog.get("my_dataset_test")

    cfg = setup_cfg(args)
    snapshot_run(cfg)

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()


if __name__ == "__main__":
    main()
