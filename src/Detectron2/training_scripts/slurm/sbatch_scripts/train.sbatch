#!/bin/bash

#SBATCH --job-name=RGB-TESTRUN-3
#SBTACH --partition=pli 
#SBATCH --account=puchalla
#SBATCH --nodes=1 
#SBATCH --ntasks=1 
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G 
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1 

#SBATCH --output=../logs/RGB-TESTRUN-3/%x.%j.out   # STDOUT → ../logs/RGB-TESTRUN-3/<jobid>.o3t
#SBATCH --error=../logs/RGB-TESTRUN-3/%x.%j.err    # STDERR → ../logs/RGB-TESTRUN-3/<jobid>.e3r

# config: 
NAME="RGB-TESTRUN-3"
MODELTYPE="TESTING_VALIDATION"
ITER=5000

# create a logs dir 
mkdir logs/${NAME}



../../train.sh --name ${NAME} --model_type ${MODELTYPE} --iter ${ITER}  --root-path ../../ --rgb true --split-cashe true --skip-tests false