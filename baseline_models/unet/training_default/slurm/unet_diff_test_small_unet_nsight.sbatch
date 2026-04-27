#!/bin/bash
#SBATCH -A m4334
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:20:00
#SBATCH --ntasks-per-node 4
#SBATCH --cpus-per-task 32
#SBATCH --gpus-per-node 4
#SBATCH -n 4
#SBATCH --image=nvcr.io/nvidia/modulus/modulus:24.09
#SBATCH --mail-user=frieldskatherine@uci.edu
#SBATCH --mail-type=ALL
#SBATCH --output=out_%j.out
#SBATCH --error=eo_%j.err

export HYDRA_FULL_ERROR=1

PROFILE_DIR=$SCRATCH/nsight_profiles/${SLURM_JOB_ID}
mkdir -p $PROFILE_DIR

# profile rank 0 only; all other ranks run normally
# early_stop_step=200 keeps the profile to a few representative steps
PROFILE_CMD="nsys profile \
  --output ${PROFILE_DIR}/unet_rank0 \
  --trace=cuda,nvtx,osrt \
  --stats=true \
  --force-overwrite=true \
  python train_unet.py \
  --config-name=config_seed_7_diff_test_smaller_unet \
  early_stop_step=200 epochs=1"

TRAIN_CMD="python train_unet.py \
  --config-name=config_seed_7_diff_test_smaller_unet \
  early_stop_step=200 epochs=1"

cd ..
srun -n $SLURM_NTASKS shifter bash -c \
  "source ddp_export.sh && \
   if [ \$SLURM_PROCID -eq 0 ]; then $PROFILE_CMD; else $TRAIN_CMD; fi"