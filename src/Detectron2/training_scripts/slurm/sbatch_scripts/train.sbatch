#!/bin/bash

#SBATCH --job-name=testing-outHook-4
#SBTACH --partition=pli 
#SBATCH --account=puchalla
#SBATCH --nodes=1 
#SBATCH --ntasks=1 
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G 
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1 

#SBATCH --output=../logs/testing-outHook-4/%x.%j.out   # STDOUT → ../logs/testing-outHook-4/<jobid>.o3t
#SBATCH --error=../logs/testing-outHook-4/%x.%j.err    # STDERR → ../logs/testing-outHook-4/<jobid>.e3r

# config: 
NAME="testing-outHook-4"
ITER=1000

# create a logs dir 
mkdir logs/${NAME}



../../train.sh --name ${NAME} --iter ${ITER}  --root-path ../../ --rgb true --split-cashe false --skip-tests true