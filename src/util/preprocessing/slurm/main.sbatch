#!/bin/bash

#SBATCH --job-name=preprocess-images
#SBTACH --partition=pli 
#SBATCH --account=puchalla
#SBATCH --nodes=1 
#SBATCH --ntasks=1 
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G 
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1 

#SBATCH --output=./logs/preprocess-images-1/%x.%j.out   # STDOUT → ./logs/preprocess-images-1/<jobid>.o3t
#SBATCH --error=./logs/preprocess-images-1/%x.%j.err    # STDERR → ./logs/preprocess-images-1/<jobid>.e3r


# create a logs dir 
mkdir ./logs/${NAME}

python3 main.py

