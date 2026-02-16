#!/bin/bash
# Single-GPU AE training (stereo mode, uses left + right camera pair)

set -e

export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4

source .venv/bin/activate

python gaussianwm/train_vae.py \
    --config-name=train_vae_single_gpu \
    vae.use_kl=false \
    dataset.use_stereo=true \
    wandb.name=ae_stereo \
    use_wandb=true
