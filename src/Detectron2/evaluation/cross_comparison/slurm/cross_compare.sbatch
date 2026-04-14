#!/bin/bash

#SBATCH --job-name=comparison-waugs-1
#SBTACH --partition=pli 
#SBATCH --account=puchalla
#SBATCH --nodes=1 
#SBATCH --ntasks=1 
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G 
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1 

#SBATCH --output=./logs/comparison-waugs-1/%x.%j.out   # STDOUT → ./logs/comparison-waugs-1/<jobid>.o3t
#SBATCH --error=./logs/comparison-waugs-1/%x.%j.err    # STDERR → ./logs/comparison-waugs-1/<jobid>.e3r

# create a logs dir 
mkdir -p logs/${NAME}

python3 run_evaluator.py \
    --output "../../evaluation_outputs/cross_comparison" \
    --rgb-model "RGB-2" \
    --depth-model "DEPTH-5" \
    --rgd-model "RGD-3" \
    --model-type "AUG_5000" \
    --slurm-output-dir "/projects/PUCHALLA/LLP2024/tumor-segmentation/src/Detectron2/slurm_output"