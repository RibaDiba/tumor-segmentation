#!/bin/bash

#SBATCH --job-name=path-test-1-rgb
#SBTACH --partition=pli 
#SBATCH --account=puchalla
#SBATCH --nodes=1 
#SBATCH --ntasks=1 
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G 
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1 

#SBATCH --output=logs/rgb-testing-ap-1-40000/%x.%j.out   # STDOUT → logs/path-test-1-rgb/<jobid>.out
#SBATCH --error=logs/rgb-testing-ap-1-40000/%x.%j.err    # STDERR → logs/path-test-1-rgb/<jobid>.err

# config: 
NAME="path-test-1-rgb"
ITER=40000

# create a logs dir 
mkdir logs/${NAME}



../../train.sh --name ${NAME} --iter ${ITER}  --root-path ../../ --rgb true --split-cache false --skip-tests true