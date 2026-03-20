#!/bin/bash
export HYDRA_FULL_ERROR=1
python gaussianwm/train_vae.py\
    --config-name train_vae_learned_q\
    use_wandb=true
