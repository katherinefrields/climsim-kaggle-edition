#!/bin/bash
#SBATCH -A m4334
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:05:00
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --image=nvcr.io/nvidia/modulus/modulus:24.01
#SBATCH --output=offline_inference_test_%j.out
#SBATCH --mail-user=frieldskatherine@gmail.com
#SBATCH --mail-type=ALL

shifter python offline_inference_test_v2_rh_mc.py