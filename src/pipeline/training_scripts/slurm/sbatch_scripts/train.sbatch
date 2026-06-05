#!/bin/bash

#SBATCH --job-name=RGB-TEST-9
#SBTACH --partition=pli 
#SBATCH --account=puchalla
#SBATCH --nodes=1 
#SBATCH --ntasks=1 
#SBATCH --mem=20G 
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1 

#SBATCH --output=../logs/RGB-TEST-9/%x.%j.out   # STDOUT → ../logs/RGB-TEST-9/<jobid>.o3t
#SBATCH --error=../logs/RGB-TEST-9/%x.%j.err    # STDERR → ../logs/RGB-TEST-9/<jobid>.e3r

# config: 
NAME="RGB-TEST-9"
MODELTYPE="TEST-REFEACTOR"
ITER=1000

# create a logs dir (must match the --output/--error path above)
mkdir -p ../logs/${NAME}



../../train.sh --name ${NAME} --model_type ${MODELTYPE} --iter ${ITER} --root-path ../../ --modality rgb  
