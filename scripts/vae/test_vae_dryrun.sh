#!/bin/bash
# VAE dry-run: 1 epoch, batch_size=4, num_workers=0

set -e

export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4

source .venv/bin/activate

python gaussianwm/train_vae.py \
    --config-name=train_vae_single_gpu \
    train.batch_size=4 \
    train.epochs=1 \
    dataloader.num_workers=0 \
    2>&1 | tee vae_dryrun.log
