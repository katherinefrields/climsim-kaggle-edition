#!/bin/bash
#SBATCH -A m4334
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 32
#SBATCH --gpus-per-node 1
#SBATCH -n 1
#SBATCH --mail-user=frieldskatherine@uci.edu
#SBATCH --mail-type=ALL
#SBATCH --output=out_%j.out
#SBATCH --error=eo_%j.err

module load conda
conda activate myenv

N_BATCHES=${N_BATCHES:-0}  # set to 0 for all batches, -1000 for 1000 batches

echo "Files in precomputed_preds dir:"
ls /pscratch/sd/k/kfrields/hugging/E3SM-MMF_saved_models/precomputed_preds/

python plot_residual_zonal_means.py --n_batches $N_BATCHES
