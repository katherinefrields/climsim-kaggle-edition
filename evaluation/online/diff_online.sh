#!/bin/bash
#SBATCH -A m4334
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:05:00
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 32
#SBATCH --gpus-per-node 1
#SBATCH -n 1
#SBATCH --image=nvcr.io/nvidia/modulus/modulus:24.01
#SBATCH --mail-user=frieldskatherine@uci.edu
#SBATCH --mail-type=ALL
#SBATCH --output=out_%j.out
#SBATCH --error=eo_%j.err



python diff_create_online_zonal_mean_bias_model.py