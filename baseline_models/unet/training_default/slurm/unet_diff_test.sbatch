#!/bin/bash
#SBATCH -A m4334
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:05:00
#SBATCH --ntasks-per-node 2
#SBATCH --cpus-per-task 2
#SBATCH --gpus-per-node 2
#SBATCH -n 2
#SBATCH --image=nvcr.io/nvidia/modulus/modulus:24.01
#SBATCH --mail-user=frieldskatherine@uci.edu
#SBATCH --mail-type=ALL
#SBATCH --output=out_%j.out
#SBATCH --error=eo_%j.err

cmd="python train_unet.py --config-name=config_seed_7_diff_test"

export HYDRA_FULL_ERROR=1

cd ..
srun -n $SLURM_NTASKS shifter bash -c "source ddp_export.sh && $cmd"