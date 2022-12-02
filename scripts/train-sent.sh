#!/bin/bash
#SBATCH --job-name=Train_SeNT
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=120

NAME=run1e
LOG=$NAME.log
cd data
python -m Evaluate gen --jobs 128 --auto $NAME >> $LOG 2>&1
cp $LOG $NAME
