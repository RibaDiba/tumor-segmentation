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

#SBATCH --output=./logs/get-images-2/%x.%j.out   # STDOUT → ./logs/get-images-2/<jobid>.o3t
#SBATCH --error=./logs/get-images-2/%x.%j.err    # STDERR → ./logs/get-images-2/<jobid>.e3r


# create a logs dir 
mkdir ./logs/${NAME}

python3 main.py

