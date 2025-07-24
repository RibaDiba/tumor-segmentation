#!/bin/bash
#SBATCH --job-name=test-job 
#SBTACH --partition=pli 
#SBATCH --account=puchalla
#SBATCH --nodes=1 
#SBATCH --ntasks=1 
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G 
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1 

#SBATCH --output=logs/%x.%j.default_trainer.out   # STDOUT → logs/<jobname>.<jobid>.out
#SBATCH --error=logs/%x.%j.default_trainer.err    # STDERR → logs/<jobname>.<jobid>.err

../train.sh --name rgb-test-1000-self-eval --iter 1000 --root-path ../ --rgb true --split-cashe false