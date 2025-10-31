#!/bin/bash

#SBATCH --job-name=test-8
#SBTACH --partition=pli 
#SBATCH --account=puchalla
#SBATCH --nodes=1 
#SBATCH --ntasks=1 
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G 
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1 

#SBATCH --output=./logs/test-8/%x.%j.out   # STDOUT → ./logs/test-8/<jobid>.o3t
#SBATCH --error=./logs/test-8/%x.%j.err    # STDERR → ./logs/test-8/<jobid>.e3r

# config: 
NAME="test-8"
ITER=5000

# create a logs dir 
mkdir ./logs/${NAME}

python3 save_data.py ./test-8

