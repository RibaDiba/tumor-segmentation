#!/bin/bash

#SBATCH --job-name=more-augmentations-4
#SBTACH --partition=pli 
#SBATCH --account=puchalla
#SBATCH --nodes=1 
#SBATCH --ntasks=1 
#SBATCH --mem=20G 
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1 

#SBATCH --output=../logs/more-augmentations-4/%x.%j.out   # STDOUT → ../logs/more-augmentations-4/<jobid>.o3t
#SBATCH --error=../logs/more-augmentations-4/%x.%j.err    # STDERR → ../logs/more-augmentations-4/<jobid>.e3r


# config:
AUGMENTATION="true"
FLIP_PROB=0.5
ROTATE_PROB=0.5
ROTATE_DEGREES=30
TARGET=1000

# create a logs dir
mkdir -p ../logs/cashe

../../cashe.sh \
  --augmentations ${AUGMENTATION} \
  --flip_prob ${FLIP_PROB} \
  --rotate_prob ${ROTATE_PROB} \
  --rotate_degrees ${ROTATE_DEGREES} \
  --target ${TARGET} \
  --root-path ../../ \
  --skip-tests true
