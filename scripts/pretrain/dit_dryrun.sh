#!/bin/bash

export HYDRA_FULL_ERROR=1

DATASET=droid
DATA_PATH=$GWM_PATH/data/

python gaussianwm/train_diffusion.py \
    --config-name train_gwm \
    dataset=$DATASET \
    dataset.data_path=$DATA_PATH \
    world_model.observation.use_gs=true \
    world_model.reward.use_reward_model=false \
    world_model.vae.use_vae=false \
    output_dir=$GWM_PATH/logs/gwm_dryrun \
    log_dir=$GWM_PATH/logs/gwm_dryrun \
    train.max_steps=10 \
    train.save_every=5 \
    train.log_every=1 \
    use_wandb=false
