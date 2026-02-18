#!/bin/bash
# Single-GPU AE training (mono mode, matches original GWM paper)

set -e

export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4

python gaussianwm/train_vae.py \
    --config-name=train_vae_single_gpu \
    vae.use_kl=false \
    dataset.use_stereo=false \
    wandb.name=ae_mono \
    use_wandb=true
