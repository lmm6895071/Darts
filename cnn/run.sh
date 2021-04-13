#!/bin/bash

salloc -A test -J lmm -N 1 --ntasks-per-node=1 --cpus-per-task=2 --gres=gpu:1 -p gpu -t 7-00:00:00 python  train_search.py 
