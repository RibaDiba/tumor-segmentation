#!/bin/bash

#SBATCH --job-name=rgd-5000-1
#SBTACH --partition=pli 
#SBATCH --account=puchalla
#SBATCH --nodes=1 
#SBATCH --ntasks=1 
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G 
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1 

#SBATCH --output=../logs/rgd-5000-1/%x.%j.out   # STDOUT → ../logs/rgd-5000-1/<jobid>.o3t
#SBATCH --error=../logs/rgd-5000-1/%x.%j.err    # STDERR → ../logs/rgd-5000-1/<jobid>.e3r

# config: 
NAME="rgd-5000-1"
ITER=5000

# create a logs dir 
mkdir logs/${NAME}



../../train.sh --name ${NAME} --iter ${ITER}  --root-path ../../ --rgd true --split-cashe false --skip-tests true