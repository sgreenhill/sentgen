#!/bin/bash
#SBATCH --job-name=Build_VAE
#SBATCH --output=gpu_job.out
#SBATCH --error=gpu_job.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=36

export OMP_NUM_THREADS=9
cd dmelodies_controllability
CUDA_VISIBLE_DEVICES=0 python script_train_dmelodies.py --net_type rnn --model_type beta-VAE --split 0.7 0.05 0.25 >log-beta-VAE.txt 2>&1 &
CUDA_VISIBLE_DEVICES=1 python script_train_dmelodies.py --net_type rnn --model_type ar-VAE --split 0.7 0.05 0.25 >log-ar-VAE.txt 2>&1 &
CUDA_VISIBLE_DEVICES=2 python script_train_dmelodies.py --net_type rnn --model_type interp-VAE --split 0.7 0.05 0.25 >log-interp-VAE.txt 2>&1 &
CUDA_VISIBLE_DEVICES=3 python script_train_dmelodies.py --net_type rnn --model_type s2-VAE --split 0.7 0.05 0.25 >log-s2-VAE.txt 2>&1 &
wait
mv src/saved_models src/saved_models-70-05-25
