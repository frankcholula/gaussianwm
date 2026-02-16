#!/bin/bash

export HYDRA_FULL_ERROR=1

python gaussianwm/train_diffusion.py \
    --config-name train_gwm \
    world_model.observation.use_gs=true \
    world_model.reward.use_reward_model=false \
    world_model.vae.use_vae=false \
    use_wandb=false