"""K-fold cross-validation training driver.

Preprocesses the dataset once, builds k folds with :class:`KFoldDataset`, then
trains a model on each fold sequentially (reusing the standard ``Trainer`` /
``RGBDTrainer`` and all evaluation hooks), and finally aggregates the per-fold
test metrics into ``models/<type>/<name>/kfold_summary.json`` plus a bar plot.

Example:
    python train_kfold.py --modality rgb --model-name cv_run --model-type CV \\
        --k 5 SOLVER.MAX_ITER 5000
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

# Resolve project imports regardless of CWD / PYTHONPATH (mirrors train.py).
_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.abspath(os.path.join(_current_dir, "../../.."))
for _p in (
    os.path.join(_project_root, "src", "util"),
    os.path.join(_project_root, "src"),
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.logger import setup_logger

from pipeline.cross_validation.kfold_dataset import KFoldDataset
from pipeline.training_scripts.train import (
    RGBD_MODALITIES,
    setup_cfg,
    snapshot_run,
)
from pipeline.trainer.trainer import Trainer
from pipeline.trainer.rgbd_trainer import RGBDTrainer
from paths import PROJECT_ROOT, MODELS_DIR

_DATASET_NAMES = ("my_dataset_train", "my_dataset_val", "my_dataset_test")
# Keys must match what AP_IOU_FinalResults.after_train() writes to ap_results.json
# and iou_results.json (see pipeline/hooks/ap_final_hook.py).
_AP_KEYS = ("AP", "AP50", "AP75", "APs", "APm", "APl")
_IOU_KEYS = ("mean_iou", "count_50", "count_75", "count_90", "count_failed")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="K-fold cross-validation training for tumor segmentation"
    )
    p.add_argument(
        "--modality",
        choices=("rgb", "depth", "rgd", "rgbd_early", "rgbd_late"),
        required=True,
        help="Image modality to train on",
    )
    p.add_argument(
        "--model-name", dest="model_name", required=True, help="Run identifier"
    )
    p.add_argument(
        "--model-type",
        dest="model_type",
        required=True,
        help="Experiment group / subdirectory",
    )
    p.add_argument(
        "--config-dir",
        dest="config_dir",
        type=Path,
        default=PROJECT_ROOT / "configs",
        help="Directory containing base.yaml and <modality>.yaml",
    )
    p.add_argument("--k", type=int, default=5, help="Number of folds (default 5)")
    p.add_argument(
        "--val-frac",
        dest="val_frac",
        type=float,
        default=0.15,
        help="Fraction of each fold's train pool held out for validation",
    )
    p.add_argument("--seed", type=int, default=42, help="Fold shuffling seed")
    p.add_argument(
        "--no-shuffle",
        dest="no_shuffle",
        action="store_true",
        help="Disable shuffling before folding (use raw order)",
    )
    p.add_argument(
        "--augmentations",
        action="store_true",
        help="Enable augmentation pipeline during preprocessing",
    )
    p.add_argument(
        "--rotate_degrees",
        type=int,
        default=0,
        help="rotation degree range (with --augmentations)",
    )
    p.add_argument(
        "opts",
        nargs=argparse.REMAINDER,
        help='Config overrides, e.g. "SOLVER.MAX_ITER 10000 SOLVER.BASE_LR 0.001"',
    )
    return p


def clear_registration() -> None:
    """Drop the standard dataset registrations so the next fold can re-register
    them under the same names (which base.yaml's DATASETS.* reference)."""
    for name in _DATASET_NAMES:
        if name in DatasetCatalog.list():
            DatasetCatalog.remove(name)
        if name in MetadataCatalog.list():
            MetadataCatalog.remove(name)


def read_fold_metrics(cfg) -> dict:
    """Load per-fold test metrics from the files written by AP_IOU_FinalResults.

    The path ``<OUTPUT_DIR>/IoU_AP_Final/json_<MODELNAME>/`` is constructed by
    ``AP_IOU_FinalResults.after_train()`` in pipeline/hooks/ap_final_hook.py.
    """
    json_dir = os.path.join(cfg.OUTPUT_DIR, "IoU_AP_Final", f"json_{cfg.MODELNAME}")
    with open(os.path.join(json_dir, "ap_results.json")) as f:
        ap = json.load(f)
    with open(os.path.join(json_dir, "iou_results.json")) as f:
        iou = json.load(f).get("dataset_metrics", {})
    return {
        "ap": {k: ap.get(k) for k in _AP_KEYS},
        "iou": {k: iou.get(k) for k in _IOU_KEYS},
    }


def _summarize(values: list) -> dict:
    """Return ``{mean, std, values}`` over numeric entries; mean/std are None if
    all values are missing (e.g. the metric was not computed for a fold)."""
    nums = [float(v) for v in values if isinstance(v, (int, float))]
    if not nums:
        return {"mean": None, "std": None, "values": values}
    return {"mean": float(np.mean(nums)), "std": float(np.std(nums)), "values": nums}


def aggregate(args, fold_metrics: list[dict]) -> dict:
    """Compute mean ± std across folds and write ``kfold_summary.json`` + bar plot."""
    summary = {
        "modality": args.modality,
        "k": args.k,
        "val_frac": args.val_frac,
        "seed": args.seed,
        "shuffle": not args.no_shuffle,
        "per_fold": fold_metrics,
        "ap": {k: _summarize([m["ap"][k] for m in fold_metrics]) for k in _AP_KEYS},
        "iou": {k: _summarize([m["iou"][k] for m in fold_metrics]) for k in _IOU_KEYS},
    }

    out_dir = MODELS_DIR / args.model_type / args.model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "kfold_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote cross-validation summary to {summary_path}")

    _plot_summary(summary, out_dir / "kfold_summary.png")
    return summary


def _plot_summary(summary: dict, out_path) -> None:
    """Bar chart of AP / AP50 / AP75 mean ± std across folds; skips missing metrics."""
    keys = [k for k in ("AP", "AP50", "AP75") if summary["ap"][k]["mean"] is not None]
    means = [summary["ap"][k]["mean"] for k in keys]
    stds = [summary["ap"][k]["std"] for k in keys]
    if not keys:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(keys, means, yerr=stds, capsize=6, color="#4C72B0")
    ax.set_ylabel("Score")
    ax.set_title(f"{summary['modality']} {summary['k']}-fold CV (mean ± std)")
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m + (s or 0) + 0.5, f"{m:.2f}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote cross-validation plot to {out_path}")


def main(argv: list[str] | None = None) -> None:
    setup_logger()
    args = build_parser().parse_args(argv)

    d = KFoldDataset(data_path=str(PROJECT_ROOT / "data/huggingface-repo/useable_data"))
    if args.augmentations:
        d.preprocess_augs(rotate_degrees=args.rotate_degrees)
    else:
        d.preprocess_images()
    d.make_folds(
        k=args.k,
        val_frac=args.val_frac,
        shuffle=not args.no_shuffle,
        seed=args.seed,
    )

    TrainerCls = RGBDTrainer if args.modality in RGBD_MODALITIES else Trainer

    fold_metrics: list[dict] = []
    for fold in range(args.k):
        print(f"\n========== Fold {fold} / {args.k - 1} ==========")
        d.prepare_fold(fold)

        clear_registration()
        d.register_instances(**{args.modality: True})
        # Force the lazy COCO dicts to be built now so any path/format errors
        # surface before training starts, not mid-epoch.
        for name in _DATASET_NAMES:
            DatasetCatalog.get(name)
            MetadataCatalog.get(name)

        cfg = setup_cfg(args, fold=fold)
        snapshot_run(cfg)

        trainer = TrainerCls(cfg)
        trainer.resume_or_load(resume=False)
        trainer.train()

        fold_metrics.append(read_fold_metrics(cfg))

    aggregate(args, fold_metrics)


if __name__ == "__main__":
    main()
