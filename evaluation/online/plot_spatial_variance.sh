#!/bin/bash
#SBATCH -A m4334
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:10:00
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 32
#SBATCH --gpus-per-node 1
#SBATCH -n 1
#SBATCH --mail-user=frieldskatherine@uci.edu
#SBATCH --mail-type=ALL
#SBATCH --image=nvcr.io/nvidia/modulus/modulus:24.09
#SBATCH --output=out_%j.out
#SBATCH --error=eo_%j.err

REPO_ROOT=/global/u2/k/kfrields/climsim-kaggle-edition
EXTRA_PKGS=/pscratch/sd/k/kfrields/shifter_packages
export PYTHONPATH=${EXTRA_PKGS}:${REPO_ROOT}:${PYTHONPATH}

N_BATCHES=${N_BATCHES:-100} #26280 = 1 year

# --- Diffusion model ---
DIFF_BASE=${DIFF_BASE:-"/pscratch/sd/k/kfrields/hugging/E3SM-MMF_saved_models/diffusion_models/fixed_validation_large_sign_penalty/"}

DIFF_CONFIG_PATH=${DIFF_CONFIG_PATH:-"${DIFF_BASE}/saved_config.yaml"}
DIFF_CHECKPOINT_PATH=${DIFF_CHECKPOINT_PATH:-""}
DIFF_N_BATCHES=${DIFF_N_BATCHES:-100}
DIFF_START_DOY=${DIFF_START_DOY:-""}

echo "Files in precomputed_preds dir:"
ls /pscratch/sd/k/kfrields/hugging/E3SM-MMF_saved_models/precomputed_preds/

# Build the python command
CMD="python plot_spatial_variance.py --n_batches $N_BATCHES"

if [ -n "$DIFF_BASE" ]; then
    CMD="$CMD --diff_config_path $DIFF_CONFIG_PATH"
    [ -n "$DIFF_CHECKPOINT_PATH" ] && CMD="$CMD --diff_checkpoint_path $DIFF_CHECKPOINT_PATH"
    [ -n "$DIFF_N_BATCHES" ]       && CMD="$CMD --diff_n_batches $DIFF_N_BATCHES"
    [ -n "$DIFF_START_DOY" ]       && CMD="$CMD --diff_start_doy $DIFF_START_DOY"
fi

echo "Running: $CMD"
shifter $CMD
