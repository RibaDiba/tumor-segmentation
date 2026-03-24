#!/bin/bash

#SBATCH --job-name=testing-red-region-7
#SBTACH --partition=pli
#SBATCH --account=puchalla
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=4
#SBATCH --time=00:45:00
#SBATCH --gres=gpu:1

#SBATCH --output=logs/testing-red-region-7/%x.%j.out   # STDOUT → logs/testing-red-region-7/<jobname>.<jobid>.out
#SBATCH --error=logs/testing-red-region-7/%x.%j.err    # STDERR → logs/testing-red-region-7/<jobname>.<jobid>.err

# config:
RGB_MODEL="RGB-1"
DEPTH_MODEL="DEPTH-1"
RGD_MODEL="RGD-1"
MODELTYPE="AUG_1000"
YAML_CONFIG="config.yaml"
OUTPUT_PATH="../../../testing_output"
BASE_MODEL_PATH="../../../models"

# create logs dir
mkdir -p logs/testing-red-region-7

python3 _run_inference.py \
    --rgb-model   ${RGB_MODEL} \
    --depth-model ${DEPTH_MODEL} \
    --rgd-model   ${RGD_MODEL} \
    --model-type  ${MODELTYPE} \
    --yaml-config ${YAML_CONFIG} \
    --output-path ${OUTPUT_PATH} \
    --base-model-path ${BASE_MODEL_PATH}
