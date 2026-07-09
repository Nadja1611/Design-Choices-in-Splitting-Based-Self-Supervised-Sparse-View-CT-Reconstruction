#!/bin/bash
#SBATCH --job-name=s2i_int_64
#SBATCH --partition=a6000
#SBATCH --time=200:00:00
#SBATCH --output=/scratch/nadja/BenchS2I/logs/out_tomoselfdq_%j.out   # Standard output file (with unique job ID)
#SBATCH --error=/scratch/nadja/BenchS2I/logs/error_tomoselfdq_%j.err     # Standard error file (with unique job ID)module load cuda/11.8.0-gcc-12.2.0-bplw5nu

python  run_doublesplit.py -l 'MSE_data' -grid_size 2  --angles 16 --correlated_noise --number_training_imgs 1000  -i 
python  run.py -l 'MSE_data' -grid_size 2  --angles 16  --number_training_imgs 1000  -i  --noise_intensity 6000 --method 'S2I'
