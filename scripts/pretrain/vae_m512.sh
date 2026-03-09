#!/bin/bash

export HYDRA_FULL_ERROR=1

DATASET=droid
DATA_PATH=$GWM_PATH/data/

python gaussianwm/train_vae.py \
    --config-name train_vae_single_gpu \
    dataset=$DATASET \
    dataset.data_path=$DATA_PATH \
    vae.num_latents=512 \
    vae.use_kl=true \
    dataset.shuffle_buffer_size=1000 \
    train.batch_size=16 \
    train.epochs=20 \
    output_dir=$GWM_PATH/logs/vae_m512/ \
    log_dir=$GWM_PATH/logs/vae_m512/ \
    wandb.project=gaussianwm \
    wandb.name=vae_kl_m512 \
    use_wandb=true
