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
| `configs/cfg.yaml` | Default Detectron2 training config |
| `requirements.txt` | Python dependencies |
| `data/huggingface-repo/useable_data/` | Cleaned raw data (scanner triplets) |
| `data/processed_data/{rgb,depth,rgd}/` | Preprocessed, split, and cached images + COCO JSON |
| `data/testing/` | pytest suite validating data integrity before training |
| `src/util/preprocessing/functions.py` | All preprocessing functions |
| `src/util/preprocessing/tumor_dataset.py` | `Dataset` class — wraps preprocessing and Detectron2 dataset registration |
| `src/util/preprocessing/process_coco_json.py` | Converts binary masks to COCO-format JSON annotations |
| `src/Detectron2/trainer/TrainerClass.py` | Custom `Trainer` subclass with hooks injected |
| `src/Detectron2/training_scripts/train.py` | Main training entry point |
| `src/Detectron2/training_scripts/train.sh` | Shell wrapper: runs pytest suite then `train.py` |
| `src/Detectron2/training_scripts/slurm/sbatch_scripts/` | SLURM job scripts for the Princeton Della cluster |
| `src/Detectron2/hooks/LossHook.py` | Train + validation loss curves |
| `src/Detectron2/hooks/APHook.py` | AP / AP50 / AP75 tracked over training (validation set) |
| `src/Detectron2/hooks/IoUHook.py` | Per-image IoU counts tracked over training (validation set) |
| `src/Detectron2/hooks/IoUEvaluator.py` | Custom `DatasetEvaluator` for per-image IoU |
| `src/Detectron2/hooks/APFinalHook.py` | Final AP + IoU results on the test set after training |
| `src/Detectron2/hooks/OutputsHook.py` | Saves all model predictions as JSON (RLE-encoded masks) |
| `src/Detectron2/evaluation/cross_comparison/` | Cross-model failure analysis (RGB vs. Depth vs. RGD) |

---

## Installation & Setup

```bash
git clone https://github.com/your-repository.git
cd tumor-segmentation
pip install -r requirements.txt
```

Detectron2 must be installed from source:

```bash
pip install git+https://github.com/facebookresearch/detectron2.git
```

Training was run on the **Princeton Della cluster** using NVIDIA A100 GPUs. The SLURM scripts under `src/Detectron2/training_scripts/slurm/sbatch_scripts/` are configured for that environment — adjust `--account` and `--partition` as needed.

---

## Running the Pipeline

Our pipeline set up has some specific prerequisits that must be followed. Before beginning training the envoirement needs to properly be set up. Mainly, this includes the `processed_data` directory where all the processed data is stored after moving through our pipeline. 

### Intial Setup 
Ensure that the hugging face repoistory is correctly set up, on intial clone of this repository, it may not be downloaded. Here is a brief guide on how to properly setup our huggingface database:

#### Make sure `git lfs` is installed 

```bash 
git lfs install
```

#### Authenticate with huggingface

You can do this by going to your huggingface account and generating a private access token. See how to do this [here](https://huggingface.co/docs/hub/en/security-tokens).

Also make sure that `huggingface-cli` is installed to authenticate. And then run `huggingface-cli login` in order to login to your huggingface account.
```bash 
pip install -U "huggingface_hub[cli]"
huggingface-cli login
```

#### Install the submodule 

Now you have to make sure that the submodule is installled 

```bash
git submodule update --init --recursive
```

### Launch Training
Use `train.sh` to run the data validation tests and launch training:

```bash
cd src/Detectron2/training_scripts

./train.sh \
  --name <model-name> \
  --model_type <output-subdir> \
  --iter <num-iterations> \
  --rgb true \          # or --depth true / --rgd true
  --split-cashe true    # set true on first run to preprocess and cache data
```

| Argument | Description |
|----------|-------------|
| `--name` | Model name, used for output directories |
| `--model_type` | Subdirectory label for saved outputs |
| `--iter` | Number of training iterations |
| `--rgb` / `--depth` / `--rgd` | Select which image modality to train on |
| `--split-cashe` | Preprocess, split, and cache data before training (first run only) |
| `--skip-tests` | Skip the pytest validation suite |

---

## Notes

`correct_binary_masks()` uses `cv2.COLOR_BAYER_BG2GRAY` where `cv2.COLOR_BGR2GRAY` is likely intended — this may behave unexpectedly depending on the OpenCV version.
