#!/bin/bash

#SBATCH --job-name=test-run-image
#SBTACH --partition=pli 
#SBATCH --account=puchalla
#SBATCH --nodes=1 
#SBATCH --ntasks=1 
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G 
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1 

#SBATCH --output=../logs/test-run-image/%x.%j.out   # STDOUT → ../logs/test-run-image/<jobid>.o3t
#SBATCH --error=../logs/test-run-image/%x.%j.err    # STDERR → ../logs/test-run-image/<jobid>.e3r

srun ../check-one.py