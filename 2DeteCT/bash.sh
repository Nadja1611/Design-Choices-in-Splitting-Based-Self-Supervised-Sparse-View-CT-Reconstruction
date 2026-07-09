#!/bin/bash
#SBATCH --job-name=p2p_int_64
#SBATCH --partition=a6000
#SBATCH --time=200:00:00
#SBATCH --output=/scratch/nadja/BenchS2I_2detect/logs/out_tomoselfdq_%j.out   # Standard output file (with unique job ID)
#SBATCH --error=/scratch/nadja/BenchS2I_2detect/logs/error_tomoselfdq_%j.err     # Standard error file (with unique job ID)module load cuda/11.8.0-gcc-12.2.0-bplw5nu


python  run.py -l 'MSE_data' -grid_size 3  --angles 32 --number_training_imgs 1000  --learning_rate 0.00005 -i --method 'S2I'