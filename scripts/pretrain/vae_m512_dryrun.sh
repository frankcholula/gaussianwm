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
    dataset.shuffle_buffer_size=10 \
    train.batch_size=16 \
    train.epochs=1 \
    output_dir=$GWM_PATH/logs/vae_m512_dryrun/ \
    log_dir=$GWM_PATH/logs/vae_m512_dryrun/ \
    use_wandb=true
