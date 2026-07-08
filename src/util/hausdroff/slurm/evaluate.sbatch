#!/bin/bash

#SBATCH --job-name=EVAL-RGBD-2
#SBATCH --account=puchalla
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=20G
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1

#SBATCH --output=../logs/%x.%j.out
#SBATCH --error=../logs/%x.%j.err

# --- config (edit per run) ---
NAME="EVAL-RGBD-2"
CHECKPOINT="/projects/PUCHALLA/LLP2024/tumor-refactor/models/NEW-MODALITY/RGBD-EARLY-2/latest/model_final.pth"
COCO_JSON="/projects/PUCHALLA/LLP2024/tumor-refactor/data/processed_data/rgbd_early/test/images"
JSON_NAME="test.json"
MODALITY="rgbd_early"
OUTPUT_PATH="../results/${NAME}_hd95.json"
SCORE_THRESH=0.3
# Set MISS_PENALTY to a number (e.g. 256) or -1 for per-image diagonal; empty = no penalty
MISS_PENALTY=""

# create directories
mkdir -p ../logs/${NAME}
mkdir -p ../results

# build args
ARGS=(
    --checkpoint "${CHECKPOINT}"
    --coco-json  "${COCO_JSON}"
    --json-name  "${JSON_NAME}"
    --modality   "${MODALITY}"
    --output-path "${OUTPUT_PATH}"
    --score-thresh "${SCORE_THRESH}"
)
[ -n "${MISS_PENALTY}" ] && ARGS+=(--miss-penalty "${MISS_PENALTY}")

python3 ../evaluate_hausdorff.py "${ARGS[@]}"
