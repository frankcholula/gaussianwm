#!/bin/bash
export HYDRA_FULL_ERROR=1
python gaussianwm/train_vae.py \
    --config-name train_vae_learned_q \
    dataset.shuffle_buffer_size=10 \
    train.epochs=1 \
    output_dir=logs/vae_learned_q_dryrun/ \
    log_dir=logs/vae_learned_q_dryrun/ \
    use_wandb=false
