#!/bin/bash

#SBATCH --job-name=testing-caching
#SBTACH --partition=pli 
#SBATCH --account=puchalla
#SBATCH --nodes=1 
#SBATCH --ntasks=1 
#SBATCH --mem=20G 
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1 

#SBATCH --output=../logs/testing-caching/%x.%j.out   # STDOUT → ../logs/testing-caching/<jobid>.out
#SBATCH --error=../logs/testing-caching/%x.%j.err    # STDERR → ../logs/testing-caching/<jobid>.err


# config:
ROTATE_DEGREES=30

# create a logs dir
mkdir -p ../logs/cache

../../cache.sh \
  --augmentations \
  --rotate_degrees ${ROTATE_DEGREES} \
  --root-path ../../ \
  --skip-tests
