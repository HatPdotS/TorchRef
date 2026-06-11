#!/bin/sh


#SBATCH -o all_test.txt
#SBATCH -e all_test.err
#SBATCH -c 8 
#SBATCH -p gpu
#SBATCH --gres=gpu:1

pytest tests/ --run-slow --run-gpu