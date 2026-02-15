#!/bin/bash
# Single-GPU VAE training: 20 epochs, batch_size=32
# Run this AFTER train_ae_single_gpu.sh completes
# Uses use_kl=true to enable KL divergence regularization

set -e

export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4

source .venv/bin/activate

python gaussianwm/train_vae.py \
    --config-name=train_vae_single_gpu \
    vae.use_kl=true \
    wandb.name=vae_baseline_20ep \
    use_wandb=true
