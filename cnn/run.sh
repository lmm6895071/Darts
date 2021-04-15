#!/bin/bash
#SBATCH -A test
#SBATCH -J 2nd-old
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH -p gpu 
#SBATCH -t 3-00:00:00
#SBATCH -o out_2nd_old_scream.out

python train_search.py --report_freq 50 --unrolled True
