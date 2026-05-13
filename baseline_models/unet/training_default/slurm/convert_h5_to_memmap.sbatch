#!/bin/bash
#SBATCH -A m4334
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:10:00
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 32
#SBATCH --gpus-per-node 1
#SBATCH -n 1
#SBATCH --image=nvcr.io/nvidia/modulus/modulus:24.09
#SBATCH --mail-user=frieldskatherine@uci.edu
#SBATCH --mail-type=ALL
#SBATCH --output=out_%j.out
#SBATCH --error=eo_%j.err

PREDS_DIR=/pscratch/sd/k/kfrields/hugging/E3SM-MMF_saved_models/precomputed_preds

cd ..
srun -n 1 shifter bash -c "
    python convert_h5_to_memmap.py \
        ${PREDS_DIR}/train_preds.h5 \
        ${PREDS_DIR}/train_preds.npy && \
    python convert_h5_to_memmap.py \
        ${PREDS_DIR}/val_preds.h5 \
        ${PREDS_DIR}/val_preds.npy
"
