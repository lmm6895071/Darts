#!/bin/bash

<<<<<<< HEAD
# SBATCH -A test
#SBATCH -J 2nd-new
=======
#SBATCH -A test
#SBATCH -J lmm
>>>>>>> d262db188b96a4364dc2ac9ef02526c8a78b4fa3
# SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH -p gpu 
#SBATCH -t 3-00:00:00
#SBATCH -o out_2nd_new_scream.out

python train_search_reverse.py --report_freq 50 --unrolled True
