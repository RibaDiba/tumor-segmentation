#!/bin/bash

#SBATCH --job-name=rgb-testing-ap-1-40000
#SBTACH --partition=pli 
#SBATCH --account=puchalla
#SBATCH --nodes=1 
#SBATCH --ntasks=1 
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G 
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1 

#SBATCH --output=logs/rgb-testing-ap-1-40000/%x.%j.out   # STDOUT → logs/rgb-testing-ap-1-40000/<jobid>.out
#SBATCH --error=logs/rgb-testing-ap-1-40000/%x.%j.err    # STDERR → logs/rgb-testing-ap-1-40000/<jobid>.err

# config: 
NAME="rgb-testing-ap-1-40000"
ITER=40000

# create a logs dir 
mkdir logs/${NAME}



../train.sh --name ${NAME} --iter ${ITER}  --root-path ../ --rgb true --split-cashe false --skip-tests true