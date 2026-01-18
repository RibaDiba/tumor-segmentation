#!/bin/bash

#SBATCH --job-name=rgd-7030-4-augs
#SBTACH --partition=pli 
#SBATCH --account=puchalla
#SBATCH --nodes=1 
#SBATCH --ntasks=1 
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G 
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1 

#SBATCH --output=../logs/rgd-7030-4-augs/%x.%j.out   # STDOUT → ../logs/rgd-7030-4-augs/<jobid>.o3t
#SBATCH --error=../logs/rgd-7030-4-augs/%x.%j.err    # STDERR → ../logs/rgd-7030-4-augs/<jobid>.e3r

# config: 
NAME="rgd-7030-4-augs"
ITER=5000

# create a logs dir 
mkdir logs/${NAME}



../../train.sh --name ${NAME} --iter ${ITER}  --root-path ../../ --rgd true --split-cashe false --skip-tests true