#!/bin/bash
# Single-GPU AE training: 10 epochs, batch_size=32
# PoC sequence: run with use_kl=false (AE) first, then use_kl=true (VAE)

set -e

export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4

source .venv/bin/activate

python gaussianwm/train_vae.py \
    --config-name=train_vae_single_gpu \
    vae.use_kl=false \
    use_wandb=true
