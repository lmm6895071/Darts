#!/bin/bash

salloc -A test -J lmm -N 1 --ntasks-per-node=1 --cpus-per-task=10 --gres=gpu:1 -p gpu -t 0-02:00:00  
