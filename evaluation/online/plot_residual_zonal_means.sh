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

# Extra Python packages installed once on login node into a writable dir:
#   shifter --image=nvcr.io/nvidia/modulus/modulus:24.09 \
#       pip install --target=/pscratch/sd/k/kfrields/shifter_packages cartopy omegaconf netCDF4 h5py
REPO_ROOT=/global/u2/k/kfrields/climsim-kaggle-edition
EXTRA_PKGS=/pscratch/sd/k/kfrields/shifter_packages
export PYTHONPATH=${EXTRA_PKGS}:${REPO_ROOT}:${PYTHONPATH}

N_BATCHES=${N_BATCHES:-1000} #26280 = 1 year

# --- Diffusion model ---
DIFF_BASE=/pscratch/sd/k/kfrields/hugging/E3SM-MMF_saved_models/diffusion_models/fixed_cross_attention_test/ #/pscratch/sd/k/kfrields/hugging/E3SM-MMF_saved_models/diffusion_models/improved_mse_1_epoch #

DIFF_CONFIG_PATH=${DIFF_CONFIG_PATH:-"${DIFF_BASE}/saved_config.yaml"}
DIFF_CHECKPOINT_PATH=${DIFF_CHECKPOINT_PATH:-""}
DIFF_N_BATCHES=${DIFF_N_BATCHES:-1000}
DIFF_START_DOY=${DIFF_START_DOY:-""}
UNET_MODEL_PATH=${UNET_MODEL_PATH:-"/pscratch/sd/k/kfrields/hugging/E3SM-MMF_saved_models/diffusion_models/cond_embeddings_5000_steps/unet_model.pt"}

echo "Files in precomputed_preds dir:"
ls /pscratch/sd/k/kfrields/hugging/E3SM-MMF_saved_models/precomputed_preds/

# Build the python command
CMD="python plot_residual_zonal_means.py --n_batches $N_BATCHES"

if [ -n "$DIFF_BASE" ]; then
    CMD="$CMD --diff_config_path $DIFF_CONFIG_PATH"
    [ -n "$DIFF_CHECKPOINT_PATH" ] && CMD="$CMD --diff_checkpoint_path $DIFF_CHECKPOINT_PATH"
    [ -n "$DIFF_N_BATCHES" ]       && CMD="$CMD --diff_n_batches $DIFF_N_BATCHES"
    [ -n "$DIFF_START_DOY" ]       && CMD="$CMD --diff_start_doy $DIFF_START_DOY"
    [ -n "$UNET_MODEL_PATH" ]      && CMD="$CMD --unet_model_path $UNET_MODEL_PATH"
fi

echo "Running: $CMD"
shifter $CMD
