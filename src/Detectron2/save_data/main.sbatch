#!/bin/bash

#SBATCH --job-name=saved-data
#SBTACH --partition=pli 
#SBATCH --account=puchalla
#SBATCH --nodes=1 
#SBATCH --ntasks=1 
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G 
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1 

#SBATCH --output=./logs/saved-data/%x.%j.out   # STDOUT → ./logs/saved-data/<jobid>.o3t
#SBATCH --error=./logs/saved-data/%x.%j.err    # STDERR → ./logs/saved-data/<jobid>.e3r

# config: 
NAME="saved-data"
ITER=5000

# create a logs dir 
mkdir ./logs/${NAME}

python3 save_data.py ./saved-data

