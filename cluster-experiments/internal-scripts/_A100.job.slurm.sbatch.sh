#!/bin/bash

#SBATCH -p gpu-long
#SBATCH -A kdss
#SBATCH --cpus-per-task=32
#SBATCH --mem=64GB
#SBATCH --gres=gpu:A100
#SBATCH --time=60:00:00
#SBATCH --output=experiments-outputs/job-%j.out
#SBATCH --exclusive
#SBATCH --nodelist=ampere02

echo "Starting job $SLURM_JOB_ID"
echo "Node: " $SLURMD_NODENAME

./internal-scripts/run-jobs.sh $@
