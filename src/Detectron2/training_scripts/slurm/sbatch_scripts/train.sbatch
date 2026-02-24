#!/bin/bash

#SBATCH --job-name=RGB-IOU-TEST-1
#SBTACH --partition=pli 
#SBATCH --account=puchalla
#SBATCH --nodes=1 
#SBATCH --ntasks=1 
#SBATCH --mem=20G 
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1 

#SBATCH --output=../logs/RGB-IOU-TEST-1/%x.%j.out   # STDOUT → ../logs/RGB-IOU-TEST-1/<jobid>.o3t
#SBATCH --error=../logs/RGB-IOU-TEST-1/%x.%j.err    # STDERR → ../logs/RGB-IOU-TEST-1/<jobid>.e3r

# config: 
NAME="RGB-IOU-TEST-1"
MODELTYPE="IOU_TEST"
ITER=1000

# create a logs dir 
mkdir logs/${NAME}



../../train.sh --name ${NAME} --model_type ${MODELTYPE} --iter ${ITER}  --root-path ../../ --rgb     true --split-cashe false --skip-tests true