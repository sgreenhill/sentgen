#!/bin/bash
#SBATCH --job-name=Extract_Data
#SBATCH --output=extract_job.out
#SBATCH --error=extract_job.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=36

scripts/gen.sh --split 0.7,0.05,0.25 --threads 4 --jobs 9 --pairs 300000 --gen data/gen --thresh 0.0 --prepdir identity
