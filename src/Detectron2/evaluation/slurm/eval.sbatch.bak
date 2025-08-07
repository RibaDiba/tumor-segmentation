#!/bin/bash

#SBATCH --job-name=test-job 
#SBTACH --partition=pli 
#SBATCH --account=puchalla
#SBATCH --nodes=1 
#SBATCH --ntasks=1 
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G 
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1 

#SBATCH --output=logs/%x.%j.out   # STDOUT → logs/<jobname>.<jobid>.out
#SBATCH --error=logs/%x.%j.err    # STDERR → logs/<jobname>.<jobid>.err

srun ../check-one.py 