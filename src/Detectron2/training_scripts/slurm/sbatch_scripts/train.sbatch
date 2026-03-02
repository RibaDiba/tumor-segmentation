#!/bin/bash

#SBATCH --job-name=RGD-101-1
#SBTACH --partition=pli 
#SBATCH --account=puchalla
#SBATCH --nodes=1 
#SBATCH --ntasks=1 
#SBATCH --mem=20G 
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1 

#SBATCH --output=../logs/RGD-101-1/%x.%j.out   # STDOUT → ../logs/RGD-101-1/<jobid>.o3t
#SBATCH --error=../logs/RGD-101-1/%x.%j.err    # STDERR → ../logs/RGD-101-1/<jobid>.e3r

# config: 
NAME="RGD-101-1"
MODELTYPE="101_MODELS"
ITER=5000

# create a logs dir 
mkdir logs/${NAME}



../../train.sh --name ${NAME} --model_type ${MODELTYPE} --iter ${ITER}  --root-path ../../ --rgd true --split-cashe false --skip-tests true