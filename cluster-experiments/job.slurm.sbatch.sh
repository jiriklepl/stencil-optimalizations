#!/bin/bash

#SBATCH -p gpu-long
#SBATCH -A kdss
#SBATCH --cpus-per-task=64
#SBATCH --mem=256GB
#SBATCH --gres=gpu:H100
#SBATCH --time=60:00:00
#SBATCH --output=experiments-outputs/job-%j.out

echo "Starting job $SLURM_JOB_ID"
echo "Node: " $SLURMD_NODENAME

./run-jobs.sh
