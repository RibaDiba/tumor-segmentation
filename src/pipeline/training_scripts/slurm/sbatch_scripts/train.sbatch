#!/bin/bash

#SBATCH --job-name=RGD-RUN-BASE-1
#SBTACH --partition=pli 
#SBATCH --account=puchalla
#SBATCH --nodes=1 
#SBATCH --ntasks=1 
#SBATCH --mem=20G 
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1 

#SBATCH --output=../logs/RGD-RUN-BASE-1/%x.%j.out   # STDOUT → ../logs/RGD-RUN-BASE-1/<jobid>.o3t
#SBATCH --error=../logs/RGD-RUN-BASE-1/%x.%j.err    # STDERR → ../logs/RGD-RUN-BASE-1/<jobid>.e3r

# config: 
NAME="RGD-RUN-BASE-1"
MODELTYPE="NEW-MODALITY"
ITER=8000

# create a logs dir (must match the --output/--error path above)
mkdir -p ../logs/${NAME}



../../train.sh --name ${NAME} --model_type ${MODELTYPE} --iter ${ITER} --root-path ../../ --modality rgd
