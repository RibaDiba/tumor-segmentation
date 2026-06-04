#!/bin/bash

#SBATCH --job-name=RGD-4
#SBTACH --partition=pli 
#SBATCH --account=puchalla
#SBATCH --nodes=1 
#SBATCH --ntasks=1 
#SBATCH --mem=20G 
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1 

#SBATCH --output=../logs/RGD-4/%x.%j.out   # STDOUT → ../logs/RGD-4/<jobid>.o3t
#SBATCH --error=../logs/RGD-4/%x.%j.err    # STDERR → ../logs/RGD-4/<jobid>.e3r

# config: 
NAME="RGD-4"
MODELTYPE="AUG_5000"
ITER=8000

# create a logs dir 
mkdir logs/${NAME}



../../train.sh --name ${NAME} --model_type ${MODELTYPE} --iter ${ITER} --root-path ../../ --modality rgd --split-cache false --skip-tests true