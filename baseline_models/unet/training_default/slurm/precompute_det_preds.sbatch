#!/bin/bash
#SBATCH -A m4334
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:05:00
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 32
#SBATCH --gpus-per-node 1
#SBATCH -n 1
#SBATCH --image=nvcr.io/nvidia/modulus/modulus:24.09
#SBATCH --mail-user=frieldskatherine@uci.edu
#SBATCH --mail-type=ALL
#SBATCH --output=out_%j.out
#SBATCH --error=eo_%j.err

cmd="python precompute_det_preds.py --config-name=config_seed_7_diff_test_smaller_unet"

export HYDRA_FULL_ERROR=1

cd ..
srun -n 1 shifter bash -c "source ddp_export.sh && $cmd"
