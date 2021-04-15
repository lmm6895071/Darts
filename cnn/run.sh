#!/bin/bash

#SBATCH -A test
#SBATCH -J lmm
# SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH -p gpu 
#SBATCH -t 3-00:00:00
#SBATCH -o out_scream.out

python train_search.py --report_freq 50 
