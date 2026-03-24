"""
Entry point for the failure-case inferencing experiment.

Generates synthetically augmented test images via FailureRecreation, then runs
each trained model (RGB, Depth, RGD) against its matching modality dataset.
RGB and RGD models each get a baseline run and an augmented run. The Depth
model gets a baseline run only (colour augmentations are irrelevant to depth
images). Saves raw prediction JSON files and runs AP_IOU_FinalResults to
produce AP/IoU metrics for each model on each dataset.

Usage
-----
python _run_inference.py \\
    --rgb-model   RGB-1 \\
    --depth-model depth-1 \\
    --rgd-model   rgd-1 \\
    --model-type  7030_SPLIT \\
    --yaml-config config.yaml \\
    --output-path testing_output
"""

import argparse
import json
import os
import sys
import types

import torch
import yaml
from pycocotools import mask as mask_util
import numpy as np

# ---------------------------------------------------------------------------
# Path setup — mirrors train.py so imports resolve regardless of cwd
# ---------------------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../.."))
src_dir = os.path.join(project_root, "src")
util_dir = os.path.join(project_root, "src", "util")
d2_dir = os.path.join(project_root, "src", "Detectron2")

for p in (src_dir, util_dir, d2_dir):
    if p not in sys.path:
        sys.path.insert(0, p)

# ---------------------------------------------------------------------------
# Detectron2 imports
# ---------------------------------------------------------------------------
from detectron2 import model_zoo
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import build_detection_test_loader
from detectron2.data.datasets import register_coco_instances
from detectron2.modeling import build_model
from detectron2.utils.logger import setup_logger

setup_logger()

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
from preprocessing.TumorDataset.tumor_dataset import Dataset
from Detectron2.hooks.APFinalHook import AP_IOU_FinalResults
from FailureRecreation import FailureRecreation

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _build_cfg(model_name, model_type, weights_path):
    """Build a Detectron2 CfgNode configured for inference on this project.

    Mirrors the config construction in
    ``src/Detectron2/training_scripts/train.py`` (lines 145–175) exactly,
    except that ``MODEL.WEIGHTS`` points to a trained ``.pth`` checkpoint
    instead of the pre-trained backbone pickle, and ``DATASETS.TRAIN`` is
    left unset (it is unused during pure inference).

    Args:
        model_name (str): Logical name for this model variant (e.g.
            ``"RGB-1"``). Stored in ``cfg.MODELNAME`` and used by hooks to
            name output files.
        model_type (str): Subdirectory that groups related model runs (e.g.
            ``"7030_SPLIT"``). Stored in ``cfg.MODELTYPE``.
        weights_path (str): Absolute path to the trained ``model_final.pth``
            file.  Pass an empty string to skip weight loading (useful when
            only registering datasets before weights are known).

    Returns:
        detectron2.config.CfgNode: Fully populated configuration node ready
        for ``build_model`` and ``DetectionCheckpointer``.
    """
    cfg = get_cfg()
    cfg.MODELNAME = model_name
    cfg.MODELTYPE = model_type
    cfg.OUTPUT_DIR = os.path.join(project_root, f"models/{model_type}/{model_name}")

    cfg.merge_from_file(
        model_zoo.get_config_file(
            "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"
        )
    )

    cfg.DATASETS.TEST = ("my_dataset_test", "my_dataset_val")
    cfg.DATALOADER.NUM_WORKERS = 1
    cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = False

    if weights_path:
        cfg.MODEL.WEIGHTS = weights_path

    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
    cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION = 0.5
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.3

    cfg.SOLVER.BASE_LR = 0.0005
    cfg.SOLVER.STEPS = [3000, 4000]
    cfg.SOLVER.GAMMA = 0.5
    cfg.SOLVER.WARMUP_ITERS = 800
    cfg.SOLVER.WARMUP_FACTOR = 0.1

    return cfg


def _load_model(cfg):
    """Build a Mask R-CNN model and load trained weights from ``cfg.MODEL.WEIGHTS``.

    Constructs the model via Detectron2's ``build_model``, loads the checkpoint
    with ``DetectionCheckpointer``, sets the model to evaluation mode, and
    returns it. The model is placed on whatever device ``cfg.MODEL.DEVICE``
    specifies (typically ``"cuda"``).

    Args:
        cfg (detectron2.config.CfgNode): Configuration node with
            ``MODEL.WEIGHTS`` pointing to a valid checkpoint file.

    Returns:
        torch.nn.Module: The model in evaluation mode with trained weights
        loaded.

    Raises:
        FileNotFoundError: If ``cfg.MODEL.WEIGHTS`` does not exist on disk.
    """
    weights_path = cfg.MODEL.WEIGHTS
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(
            f"Trained model checkpoint not found: {weights_path}\n"
            "Check --base-model-path, --model-type, and model name flags."
        )

    model = build_model(cfg)
    DetectionCheckpointer(model).load(weights_path)
    model.eval()
    return model


def _register_augmented_dataset(json_path, image_dir, dataset_name):
    """Register an augmented COCO JSON dataset with Detectron2's DatasetCatalog.

    Calls ``register_coco_instances`` to make ``dataset_name`` available for
    use with ``build_detection_test_loader`` and evaluation hooks. If the
    dataset name is already registered (e.g. from a previous call in the same
    process) the registration is silently skipped.

    Args:
        json_path (str): Absolute path to the COCO-format JSON annotation file
            (e.g. ``testing_output/augmented/augmented_test.json``).
        image_dir (str): Absolute path to the directory containing the
            augmented JPEG images referenced by ``json_path``.
        dataset_name (str): The name under which to register the dataset (e.g.
            ``"my_dataset_augmented_test"``).

    Raises:
        FileNotFoundError: If ``json_path`` does not exist, indicating that
            ``FailureRecreation.image_recreation()`` has not yet been run.
    """
    if not os.path.isfile(json_path):
        raise FileNotFoundError(
            f"Augmented COCO JSON not found: {json_path}\n"
            "Ensure FailureRecreation.image_recreation() completed successfully."
        )

    try:
        register_coco_instances(dataset_name, {}, json_path, image_dir)
    except AssertionError:
        # Dataset already registered in this process — safe to ignore
        pass


# ---------------------------------------------------------------------------
# Core inference function
# ---------------------------------------------------------------------------


def _run_inferencing(cfg, model, dataset_name, val_dataset_name, output_dir):
    """Run a single model on a registered dataset and save results.

    1. **Runs inference** using Detectron2's ``build_detection_test_loader``
       and a manual ``torch.no_grad()`` loop — the standard Detectron2
       inference syntax.
    2. **Saves a JSON file** of raw per-image predictions (boxes, scores,
       classes, RLE-encoded masks) to ``output_dir``.
    3. **Uses the final IoU/AP hook** (``AP_IOU_FinalResults``) to produce
       COCO AP metrics (AP, AP50, AP75, APs, APm, APl) and per-image IoU
       scores. The hook is fired via the fake-trainer pattern so no actual
       training loop is needed. All hook outputs are saved to
       ``output_dir/<model_name>/``.

    Args:
        cfg (detectron2.config.CfgNode): Configuration node for this model.
        model (torch.nn.Module): Loaded model in eval mode.
        dataset_name (str): Registered Detectron2 dataset name to run
            inference on (e.g. ``"my_dataset_rgb_test"``).
        val_dataset_name (str): Matching validation dataset name used by the
            AP/IoU hook (e.g. ``"my_dataset_rgb_val"``).
        output_dir (str): Root directory for this run's outputs. Created
            automatically if it does not exist.

    Raises:
        RuntimeError: If the inference loop produces no results.
    """
    os.makedirs(output_dir, exist_ok=True)

    model_name = cfg.MODELNAME

    # Clone cfg and point DATASETS.TEST at the target dataset so that
    # AP_IOU_FinalResults uses the correct annotations.
    cfg = cfg.clone()
    cfg.defrost()
    cfg.DATASETS.TEST = (dataset_name, val_dataset_name)
    cfg.freeze()

    print(f"\n[_run_inferencing] model={model_name}  dataset={dataset_name}")

    # ------------------------------------------------------------------
    # Run inferencing using detectron2 syntax
    # ------------------------------------------------------------------
    loader = build_detection_test_loader(cfg, dataset_name)
    results = []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            predictions = model(batch)
            for input_data, prediction in zip(batch, predictions):
                instances = prediction["instances"].to("cpu")

                entry = {
                    "image_id":  input_data.get("image_id"),
                    "file_name": input_data.get("file_name"),
                    "boxes":   instances.pred_boxes.tensor.tolist()
                               if instances.has("pred_boxes") else [],
                    "scores":  instances.scores.tolist()
                               if instances.has("scores") else [],
                    "classes": instances.pred_classes.tolist()
                               if instances.has("pred_classes") else [],
                    "masks":   [],
                }

                if instances.has("pred_masks"):
                    for mask_tensor in instances.pred_masks:
                        rle = mask_util.encode(
                            np.asfortranarray(mask_tensor.numpy())
                        )
                        rle["counts"] = rle["counts"].decode("utf-8")
                        entry["masks"].append(rle)

                results.append(entry)

    if not results:
        raise RuntimeError(
            f"Inference produced no results for model '{model_name}' "
            f"on dataset '{dataset_name}'."
        )

    # ------------------------------------------------------------------
    # Make sure to save json file
    # ------------------------------------------------------------------
    json_out_path = os.path.join(output_dir, f"{model_name}_predictions.json")
    with open(json_out_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"  Saved predictions → {json_out_path}  ({len(results)} images)")

    # ------------------------------------------------------------------
    # Use the final IoU/AP hook here but save it to testing_output.
    # Fired via the fake-trainer pattern: AP_IOU_FinalResults accesses
    # self.trainer.model and self.trainer.cfg, so a SimpleNamespace with
    # those two attributes is sufficient.
    # ------------------------------------------------------------------
    hook_output_dir = os.path.join(output_dir, model_name)
    hook = AP_IOU_FinalResults(output_dir=hook_output_dir, cfg=cfg)
    hook.trainer = types.SimpleNamespace(model=model, cfg=cfg)
    hook.after_train()
    print(f"  Saved AP/IoU results → {hook_output_dir}/json_{model_name}/")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    """Parse arguments, generate augmented images, load models, and run inference.

    Orchestrates the full failure-case inferencing pipeline:

    1. **Register all three modalities** under distinct Detectron2 dataset
       names (``my_dataset_rgb_*``, ``my_dataset_depth_*``,
       ``my_dataset_rgd_*``).

    2. **Generate augmented datasets** for RGB and RGD via
       ``FailureRecreation``. Depth is not augmented (colour augmentations
       are irrelevant to depth images).

    3. **Load all three models** before the inference loops so weights are
       not reloaded between datasets.

    4. **Run matched inferencing** — each model runs only on its own
       modality's baseline (and augmented, where applicable):

       - RGB  model → ``my_dataset_rgb_test``  + ``my_dataset_rgb_augmented_test``
       - Depth model → ``my_dataset_depth_test`` only
       - RGD  model → ``my_dataset_rgd_test``  + ``my_dataset_rgd_augmented_test``

    Command-line Arguments:
        --rgb-model (str): Name of the trained RGB model (e.g. ``"RGB-1"``).
        --depth-model (str): Name of the trained Depth model.
        --rgd-model (str): Name of the trained RGD model.
        --model-type (str): Subdirectory grouping model runs (e.g.
            ``"7030_SPLIT"``).
        --base-model-path (str): Root directory where trained model
            checkpoints are stored. Defaults to ``<project_root>/models``.
        --output-path (str): Root directory for all inference outputs.
            Defaults to ``<project_root>/testing_output``.
        --yaml-config (str): Path to the augmentation ``config.yaml`` file.
            Defaults to ``config.yaml`` in the same directory as this script.
    """
    parser = argparse.ArgumentParser(
        description="Failure-case inferencing: augment test images and evaluate all three models."
    )

    parser.add_argument("--rgb-model",    required=True, type=str,
                        help="Name of the trained RGB model (e.g. 'RGB-1')")
    parser.add_argument("--depth-model",  required=True, type=str,
                        help="Name of the trained Depth model")
    parser.add_argument("--rgd-model",    required=True, type=str,
                        help="Name of the trained RGD model")
    parser.add_argument("--model-type",   required=True, type=str,
                        help="Model type subdirectory (e.g. '7030_SPLIT')")
    parser.add_argument("--base-model-path", type=str,
                        default=os.path.join(project_root, "models"),
                        help="Root directory containing trained model checkpoints")
    parser.add_argument("--output-path", type=str,
                        default=os.path.join(project_root, "testing_output"),
                        help="Root directory for all inference outputs")
    parser.add_argument("--yaml-config", type=str,
                        default=os.path.join(current_dir, "config.yaml"),
                        help="Path to the augmentation config.yaml")

    args = parser.parse_args()

    model_type      = args.model_type
    base_model_path = args.base_model_path
    output_path     = args.output_path

    # ------------------------------------------------------------------
    # Register all three modalities under distinct Detectron2 dataset names
    # so each model can be evaluated against its own matching data.
    # ------------------------------------------------------------------
    with open(args.yaml_config) as f:
        yaml_config = yaml.safe_load(f)

    d = Dataset(data_path=os.path.join(project_root, "data/huggingface-repo/useable_data"))
    d.convert_binary_to_coco()

    for modality, test_dir, val_dir, train_dir in [
        ("rgb",   d.rgb_test_dir,   d.rgb_val_dir,   d.rgb_train_dir),
        ("depth", d.depth_test_dir, d.depth_val_dir, d.depth_train_dir),
        ("rgd",   d.rgd_test_dir,   d.rgd_val_dir,   d.rgd_train_dir),
    ]:
        try:
            register_coco_instances(
                f"my_dataset_{modality}_train", {},
                os.path.join(train_dir, "train.json"), train_dir,
            )
            register_coco_instances(
                f"my_dataset_{modality}_val", {},
                os.path.join(val_dir, "val.json"), val_dir,
            )
            register_coco_instances(
                f"my_dataset_{modality}_test", {},
                os.path.join(test_dir, "test.json"), test_dir,
            )
        except AssertionError:
            # Already registered in this process — safe to ignore
            pass

    # ------------------------------------------------------------------
    # Generate augmented datasets for RGB and RGD.
    # Depth is skipped — colour augmentations don't apply to depth images.
    # ------------------------------------------------------------------
    cfg_temp = _build_cfg("placeholder", model_type, "")

    for modality in ("rgb", "rgd"):
        aug_output = os.path.join(output_path, "augmented", modality)
        fr = FailureRecreation(
            cfg_temp, aug_output, yaml_config,
            dataset_name=f"my_dataset_{modality}_test",
        )
        fr.image_recreation()
        _register_augmented_dataset(
            json_path=os.path.join(aug_output, "augmented_test.json"),
            image_dir=os.path.join(aug_output, "images"),
            dataset_name=f"my_dataset_{modality}_augmented_test",
        )

    # ------------------------------------------------------------------
    # Load all three models before any inference loop begins so weights
    # are not reloaded between datasets.
    # ------------------------------------------------------------------
    model_names = {
        "rgb":   args.rgb_model,
        "depth": args.depth_model,
        "rgd":   args.rgd_model,
    }
    models = {}
    for variant, name in model_names.items():
        print(f"[main] Loading {variant} model: {name}")
        weights_path = os.path.join(base_model_path, model_type, name, "model_final.pth")
        cfg_variant = _build_cfg(name, model_type, weights_path)
        models[variant] = (cfg_variant, _load_model(cfg_variant))

    # ------------------------------------------------------------------
    # Run matched inferencing — each model against its own modality.
    # RGB and RGD: baseline + augmented. Depth: baseline only.
    # ------------------------------------------------------------------
    for variant, (cfg_v, model_v) in models.items():
        val_ds = f"my_dataset_{variant}_val"

        # Baseline
        _run_inferencing(
            cfg=cfg_v,
            model=model_v,
            dataset_name=f"my_dataset_{variant}_test",
            val_dataset_name=val_ds,
            output_dir=os.path.join(output_path, variant, "baseline"),
        )

        # Augmented (RGB and RGD only)
        if variant != "depth":
            _run_inferencing(
                cfg=cfg_v,
                model=model_v,
                dataset_name=f"my_dataset_{variant}_augmented_test",
                val_dataset_name=val_ds,
                output_dir=os.path.join(output_path, variant, "augmented"),
            )

    print("\n[main] Done. Results written to:", output_path)


if __name__ == "__main__":
    main()
