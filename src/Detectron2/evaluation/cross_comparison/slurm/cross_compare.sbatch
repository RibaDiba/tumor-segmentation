#!/bin/bash

#SBATCH --job-name=testing-failed-compare-4
#SBTACH --partition=pli 
#SBATCH --account=puchalla
#SBATCH --nodes=1 
#SBATCH --ntasks=1 
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G 
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1 

#SBATCH --output=./logs/testing-failed-compare-4/%x.%j.out   # STDOUT → ./logs/testing-failed-compare-4/<jobid>.o3t
#SBATCH --error=./logs/testing-failed-compare-4/%x.%j.err    # STDERR → ./logs/testing-failed-compare-4/<jobid>.e3r

# create a logs dir 
mkdir -p logs/${NAME}

python3 run_evaluator.py --output "../../evaluation_outputs/cross_comparison" --rgb-model "rgb-7030-4" --depth-model "depth-7030-4" --rgd-model "rgd-7030-4"