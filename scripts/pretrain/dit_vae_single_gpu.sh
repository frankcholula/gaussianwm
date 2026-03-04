#!/bin/bash

export HYDRA_FULL_ERROR=1

DATASET=droid
DATA_PATH=$GWM_PATH/data/

python gaussianwm/train_diffusion.py \
    --config-name train_dit_with_vae \
    dataset=$DATASET \
    dataset.data_path=$DATA_PATH \
    output_dir=$GWM_PATH/logs/gwm_vae \
    log_dir=$GWM_PATH/logs/gwm_vae
