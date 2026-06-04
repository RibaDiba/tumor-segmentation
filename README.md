# Mouse Tumor Segmentation with Detectron2

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)
![Detectron2](https://img.shields.io/badge/Detectron2-Mask_R--CNN-0064FF)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?logo=opencv&logoColor=white)
![Platform](https://img.shields.io/badge/Platform-Linux-FCC624?logo=linux&logoColor=black)
![HPC](https://img.shields.io/badge/HPC-SLURM%20%7C%20A100-76B900)

This repository contains the full pipeline for automated instance segmentation of subcutaneous mouse tumors using Detectron2 (Mask R-CNN). Data is sourced from Biopticon's TI-2 structured-light scanner. Three model variants are trained and evaluated: **RGB**, **Depth**, and **RGD** (Red-Green-Depth).

The RGD model replaces the blue channel of the texture image with a normalized depth map derived from the scanner's point cloud, producing the strongest overall segmentation results.

---

## Repository Structure

| Path | Description |
|------|-------------|
| `configs/base.yaml` | Shared Detectron2 hyperparameters (LR schedule, ROI heads, dataloader) |
| `configs/{rgb,depth,rgd}.yaml` | Per-modality overrides, inherit from `base.yaml` via `_BASE_` |
| `requirements.txt` | Python dependencies |
| `data/huggingface-repo/useable_data/` | Cleaned raw data (scanner triplets) |
| `data/processed_data/{rgb,depth,rgd}/` | Preprocessed, split, and cached images + COCO JSON |
| `data/testing/` | pytest suite validating data integrity before training |
| `src/util/preprocessing/PreprocessingFunctions/functions.py` | Aggregates all preprocessing functions (`_io`, `_masks`, `_depth`, `_transforms`, `_splitting`, `_subset`) |
| `src/util/preprocessing/TumorDataset/tumor_dataset.py` | `Dataset` class — wraps preprocessing and Detectron2 dataset registration |
| `src/util/preprocessing/process_coco_json.py` | Converts binary masks to COCO-format JSON annotations |
| `src/pipeline/trainer/trainer.py` | Custom `Trainer` subclass with hooks injected |
| `src/pipeline/training_scripts/train.py` | Main training entry point |
| `src/pipeline/training_scripts/train.sh` | Shell wrapper: runs pytest suite then `train.py` |
| `src/pipeline/training_scripts/slurm/sbatch_scripts/` | SLURM job scripts for the Princeton Della cluster |
| `src/pipeline/hooks/loss_hook.py` | Train + validation loss curves |
| `src/pipeline/hooks/ap_hook.py` | AP / AP50 / AP75 tracked over training (validation set) |
| `src/pipeline/hooks/iou_hook.py` | Per-image IoU counts tracked over training (validation set) |
| `src/pipeline/hooks/iou_evaluator.py` | Custom `DatasetEvaluator` for per-image IoU |
| `src/pipeline/hooks/ap_final_hook.py` | Final AP + IoU results on the test set after training |
| `src/pipeline/hooks/outputs_hook.py` | Saves all model predictions as JSON (RLE-encoded masks) |
| `src/pipeline/evaluation/cross_comparison/` | Cross-model failure analysis (RGB vs. Depth vs. RGD) |

---

## Installation & Setup

```bash
git clone https://github.com/your-repository.git
cd tumor-segmentation
pip install -e .            # installs deps + makes `Detectron2` and `preprocessing` importable
pip install -e ".[dev]"     # plus pytest / black / ipykernel
```

The editable install is required: it adds `src/pipeline/` and `src/util/preprocessing/` to your import path so the training scripts run from anywhere without `sys.path` hacks.

Detectron2 (and SAM, if needed) are not on PyPI and must be installed from source separately:

```bash
pip install git+https://github.com/facebookresearch/detectron2.git
pip install git+https://github.com/facebookresearch/segment-anything.git
```

After `pip install -e .`, the training entry point is available as a console script:

```bash
tumor-segmentation-train --model-name <name> --model-type <group> --modality rgb
```

…which is equivalent to `python3 src/pipeline/training_scripts/train.py …`.

Training was run on the **Princeton Della cluster** using NVIDIA A100 GPUs. The SLURM scripts under `src/pipeline/training_scripts/slurm/sbatch_scripts/` are configured for that environment — adjust `--account` and `--partition` as needed.

---

## Running the Pipeline

The pipeline has a few prerequisites that must be satisfied before training. In particular, the `processed_data/` directory — where cached, split data lives — is produced by the first training run with `--split-cache true`.

### Initial Setup

The dataset is hosted as a HuggingFace dataset repository and is included here as a git submodule. On initial clone of this repository it will not be populated. The guide below sets up the HuggingFace submodule.

> **Note on data access.** The dataset is currently gated behind HuggingFace authentication. If you do not have access, contact the maintainers. A public release with an open license is planned — see the [Data](#data) section.

#### 1. Install `git lfs`

```bash
git lfs install
```

#### 2. Authenticate with HuggingFace

Generate a private access token from your HuggingFace account ([instructions](https://huggingface.co/docs/hub/en/security-tokens)), install the CLI, and log in:

```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli login
```

#### 3. Initialize the submodule

```bash
git submodule update --init --recursive
```

### Launch Training

The recommended entry point is the `train.sh` wrapper, which runs the pytest data-validation suite and then calls `train.py`:

```bash
cd src/pipeline/training_scripts

./train.sh \
  --name <model-name> \
  --model_type <output-subdir> \
  --iter <num-iterations> \
  --modality rgb \      # one of: rgb | depth | rgd
  --split-cache         # pass on first run to preprocess and cache data
```

#### `train.sh` arguments

| Argument | Description |
|----------|-------------|
| `--name` | Model name, used for output directories |
| `--model_type` | Subdirectory label for saved outputs (e.g. `rgb`, `depth`, `rgd`) |
| `--iter` | Number of training iterations |
| `--modality` | Image modality to train on (`rgb`, `depth`, or `rgd`) — required |
| `--split-cache` | Preprocess, split (70/10/20), and cache data before training (first run only) |
| `--augmentations` | Enable augmentation pipeline |
| `--flip_prob` | Probability of horizontal flip |
| `--rotate-prob` | Probability of rotation |
| `--rotate-degrees` | Maximum +/- rotation angle in degrees |
| `--target` | Target image count post-augmentation |
| `--root-path` | Path prefix prepended to `train.py` and the pytest suite (default: `./`) |
| `--skip-tests` | Skip the pytest validation suite |

All boolean flags are bare switches — pass `--split-cache` to enable, omit it to disable. (The legacy `--rgb true` / `--depth true` / `--rgd true` triplet has been replaced by a single `--modality` choice.)

#### `train.py` CLI (direct invocation)

`train.sh` is a thin wrapper that translates its named flags into the keyword arguments expected by `train.py`. If you want to call the Python entry point directly, the signature is:

```bash
python3 train.py --model-name <name> --model-type <group> --modality {rgb,depth,rgd} [options] [-- KEY VALUE ...]
```

| Option | Description |
|--------|-------------|
| `--model-name` | Required. Written into `cfg.MODELNAME` and used in output paths. |
| `--model-type` | Required. Experiment group / output subdirectory (`cfg.MODELTYPE`). |
| `--modality` | Required. One of `rgb`, `depth`, `rgd`. |
| `--config-dir` | Directory containing `base.yaml` and `<modality>.yaml`. Defaults to `configs/`. |
| `--split-cache` | Run preprocess → split → cache before training |
| `--augmentations` | Enable augmentations during preprocessing |
| `--flip_prob`, `--rotate_prob`, `--rotate_degrees`, `--target` | Augmentation parameters |

Any positional arguments after `--` are passed straight to `cfg.merge_from_list`, so e.g. `-- SOLVER.MAX_ITER 10000 SOLVER.BASE_LR 0.0001` overrides those fields. CLI overrides win over the YAML hierarchy.

## Configuration

Configs are layered (later wins):

1. Detectron2 model zoo defaults — Mask R-CNN R-101 FPN 3x
2. `configs/base.yaml` — project-wide overrides (LR schedule, ROI heads, dataloader)
3. `configs/{rgb,depth,rgd}.yaml` — per-modality overrides, chain via `_BASE_: base.yaml`
4. CLI: `./train.sh ... -- SOLVER.BASE_LR 0.001`

Every run snapshots its fully merged config to `models/<model_type>/<model_name>/run_*/config_used.yaml` — so a reviewer reading a checkpoint never has to guess what hyperparameters produced it.

## Output layout

```
models/<model_type>/<model_name>/
  run_YYYYMMDD-HHMM/
    config_used.yaml         # fully merged config (every key)
    model_final.pth          # weights
    metrics.json             # Detectron2 default
    events.out.tfevents.*    # TensorBoard scalars
  latest -> run_YYYYMMDD-HHMM/   # symlink to the most recent run

src/pipeline/slurm_output/<model_type>/<model_name>/run_YYYYMMDD-HHMM/
  loss_plots/  AP_Fig/  IoU_fig/  outputs/  IoU_AP_Final/
```

Each invocation creates a fresh `run_<timestamp>/` directory, so reruns never silently overwrite earlier weights.

## Experiment tracking

TensorBoard is wired in by default — `total_loss`, `loss_mask`, `AP`, `AP50`, `AP75`, and the IoU buckets are written by Detectron2's event storage:

```bash
tensorboard --logdir models/
```

Weights & Biases is opt-in. Set `WANDB_PROJECT` before launching and a `WandbWriter` is appended to the writer list automatically:

```bash
pip install wandb && wandb login
WANDB_PROJECT=tumor-seg ./train.sh --name foo --model_type rgb --iter 5000 --modality rgb
```

---

## Reproducing the Paper

> **Status:** results in this README are produced by the commands below. Update the seed, model checkpoints, and runtime numbers if you change the recipe.

The published numbers were produced with:

| Artifact | Command | Notes |
|----------|---------|-------|
| Table 1 (RGB / Depth / RGD AP, AP50, AP75, IoU) | `./train.sh --name rgb_final --model_type rgb --iter <N> --modality rgb --split-cache` (and analogously `--modality depth` / `--modality rgd`) | Final metrics are logged by `ap_final_hook` under `src/pipeline/slurm_output/<model_type>/<model_name>/run_*/IoU_AP_Final/`. |
| Figure 3 (training curves) | Loss / AP / IoU curves are written by `LossHook`, `APHook`, `IoUHook` during the same runs above. | Plots are produced by the notebooks under `src/util/plotting/` (or equivalent) — see commit history (`ef7bbf7`, `de2dc51`) for the most recent plot recipe. |
| Cross-model failure analysis | `src/pipeline/evaluation/cross_comparison/` | Run after all three models are trained. |

**Compute.** All training runs were performed on a single NVIDIA A100 (40 GB) on the Princeton Della cluster.

**Seed.** Seeds are currently inherited from Detectron2's defaults — exact reproduction of paper numbers requires fixing the seed in `cfg` before broad release (tracked as TODO).

**Expected runtime.** _TBD — fill in once a clean run has been timed end-to-end._

---

## Results

> _Numbers below are placeholders. Replace with the final published values and link to the paper / preprint once available._

| Model | AP | AP50 | AP75 | Mean per-image IoU |
|-------|----|------|------|--------------------|
| RGB   | —  | —    | —    | —                  |
| Depth | —  | —    | —    | —                  |
| RGD   | —  | —    | —    | —                  |

Paper / preprint: _TBD_.

---

## Data

The dataset is hosted on HuggingFace at `data/huggingface-repo/` (added as a git submodule). See [`data/README.md`](data/README.md) for the per-directory breakdown.

| Field | Value |
|-------|-------|
| Subjects | Mice (subcutaneous tumor model) |
| Acquisition device | Biopticon TI-2 structured-light scanner (RGB texture + depth from point cloud) |
| Modalities | RGB, depth, RGD (red+green from texture, blue replaced by normalized depth) |
| Splits | 70 / 10 / 20 train / val / test, produced by `Dataset.split_train_val_test` |
| Source datasets | `MC_Data` (600 pairs, point-cloud `.bin` files missing for some pairs and are excluded), `Invotive Data` (35 pairs), `usable_data` (the merged set actually used for training) |
| Ethics / IACUC | _TBD — add IACUC protocol number and approving institution._ |
| License | _TBD — confirm before public release._ |
| Access | Currently gated by HuggingFace authentication. |

---

## Known Issues

- **`correct_binary_masks()` uses the wrong OpenCV color-conversion flag.**
  Location: `src/util/preprocessing/PreprocessingFunctions/_masks.py:87`. The call passes `cv2.COLOR_BAYER_BG2GRAY` where `cv2.COLOR_BGR2GRAY` is intended. This treats the input as a Bayer-pattern raw image rather than a BGR image, so the resulting grayscale (and the binary mask derived from it) is incorrect.
  - **Impact on published results:** the function is invoked from `src/util/preprocessing/TumorDataset/_preprocessing.py` (`_preprocessing.py:46`, `:111`, `:122`, `:135`) and runs on every preprocessing pass. All currently published numbers were produced with this bug present.
  - **Fix:** swap the flag to `cv2.COLOR_BGR2GRAY` and re-run `--split-cache true` to regenerate the cached splits before retraining.
