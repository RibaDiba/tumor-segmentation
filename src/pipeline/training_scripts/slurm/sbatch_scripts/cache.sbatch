#!/bin/bash

#SBATCH --job-name=new-augmentations-3
#SBTACH --partition=pli 
#SBATCH --account=puchalla
#SBATCH --nodes=1 
#SBATCH --ntasks=1 
#SBATCH --mem=20G 
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1 

#SBATCH --output=../logs/new-augmentations-3/%x.%j.out   # STDOUT → ../logs/new-augmentations-3/<jobid>.o3t
#SBATCH --error=../logs/new-augmentations-3/%x.%j.err    # STDERR → ../logs/new-augmentations-3/<jobid>.e3r


# config:
AUGMENTATION="true"
ROTATE_DEGREES=30

# create a logs dir
mkdir -p ../logs/cache

../../cache.sh \
  --augmentations ${AUGMENTATION} \
  --rotate_degrees ${ROTATE_DEGREES} \
  --root-path ../../ \
  --skip-tests true
