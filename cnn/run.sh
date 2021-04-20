#!/bin/bash
#SBATCH -A test
#SBATCH -J 2nd-new
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH -p gpu 
#SBATCH -t 3-00:00:00
#SBATCH -o out_2nd-new-200-0.01-0.01-scream.out

python train_search_reverse.py --report_freq 50  --arch_learning_rate 0.01 --learning_rate 0.01 --epochs 800 --unrolled
