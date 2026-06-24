#!/bin/bash

#SBATCH --job-name=CV-RUN-1
#SBATCH --partition=pli
#SBATCH --account=puchalla
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=20G
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1

#SBATCH --output=../logs/CV-RUN-1/%x.%j.out
#SBATCH --error=../logs/CV-RUN-1/%x.%j.err

# config:
NAME="CV-RUN-1"
MODELTYPE="CROSS-VALIDATION"
MODALITY="rgb"
K=5
ITER=5000

# create a logs dir (must match the --output/--error path above)
mkdir -p ../logs/${NAME}

# train_kfold.py preprocesses once, builds K folds, trains each sequentially,
# then writes models/${MODELTYPE}/${NAME}/kfold_summary.json. Folds run back to
# back in one process, hence the longer --time above.
python3 ../train_kfold.py \
    --model-name "${NAME}" \
    --model-type "${MODELTYPE}" \
    --modality "${MODALITY}" \
    --k "${K}" \
    SOLVER.MAX_ITER "${ITER}"
