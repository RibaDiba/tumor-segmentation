#!/bin/bash

#SBATCH --job-name=final-depth-test-2-5000
#SBTACH --partition=pli 
#SBATCH --account=puchalla
#SBATCH --nodes=1 
#SBATCH --ntasks=1 
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G 
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1 

#SBATCH --output=logs/final-depth-test-2-5000/%x.%j.out   # STDOUT → logs/final-depth-test-2-5000/<jobid>.out
#SBATCH --error=logs/final-depth-test-2-5000/%x.%j.err    # STDERR → logs/final-depth-test-2-5000/<jobid>.err

# config: 
NAME="final-depth-test-2-5000"
ITER=5000

# create a logs dir 
mkdir logs/${NAME}



../train.sh --name ${NAME} --iter ${ITER}  --root-path ../ --depth true --split-cashe false --skip-tests true