#!/bin/bash

export HYDRA_FULL_ERROR=1

DATASET=droid
DATA_PATH=$GWM_PATH/data/

python gaussianwm/train_diffusion.py \
    --config-name train_dit_with_vae \
    dataset=$DATASET \
    dataset.data_path=$DATA_PATH \
    world_model.vae.checkpoint_path=$GWM_PATH/logs/vae_m512/checkpoint-19.pth \
    world_model.vae.freeze=true \
    world_model.batch_size=12 \
    train.update_tokenizer=false \
    output_dir=$GWM_PATH/logs/gwm_vae_v2 \
    log_dir=$GWM_PATH/logs/gwm_vae_v2 \
    wandb.name=dit_frozen_vae_sigma_fix
