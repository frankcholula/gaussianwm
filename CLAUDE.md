# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Gaussian World Model (GWM) - a latent Diffusion Transformer (DiT) + 3D variational autoencoder for scene-level future state prediction with Gaussian Splatting, targeting robotic manipulation. Python 3.10, PyTorch 2.5.1, CUDA 12.1.

## Commands

### Setup
```bash
export GWM_PATH=$(pwd)
pip install uv && uv sync && source .venv/bin/activate
uv pip install git+https://github.com/dcharatan/diff-gaussian-rasterization-modified --no-build-isolation
uv pip install git+https://github.com/facebookresearch/pytorch3d.git --no-build-isolation
```

### Training
```bash
# VAE (single GPU)
python gaussianwm/train_vae.py --config-name=train_vae_single_gpu use_wandb=true

# VAE (multi-GPU, 4x)
bash scripts/pretrain/vae.sh

# Diffusion Transformer (multi-GPU)
bash scripts/pretrain/dit.sh

# VAE dry-run test
bash scripts/test_vae_dryrun.sh

# Inference
python gaussianwm/demo.py
```

### Multi-GPU Pattern
```bash
CUDA_VISIBLE_DEVICES=1,2,3,4 torchrun --nproc_per_node=4 --master_port 12345 \
    gaussianwm/train_vae.py --config-name train_vae
```

## Architecture

**Two-stage training pipeline:**
1. **VAE stage**: Images → Splatt3r (3D Gaussian points) → 3D Autoencoder → latent codes
2. **DiT stage**: Latent codes + actions → Diffusion Transformer → predicted future latents → decoded → rendered via Gaussian splatting

**Key modules in `gaussianwm/`:**
- `gwm_predictor.py` — Orchestrator: ties together Splatt3r, VAE, DiT, and reward model
- `encoder/models_ae.py` — Transformer-based 3D autoencoder (point cloud ↔ latent)
- `diffusion/models.py` — DiT with timestep/action conditioning
- `diffusion/gaussian_diffusion.py` — Forward/reverse diffusion process, noise schedules
- `processor/regressor.py` — Splatt3r wrapper: stereo images → 3D Gaussian point clouds (position, covariance, SH coefficients; 14 dims per point)
- `processor/datasets.py` — Dataset builders; DROID (RLDS format) is the primary dataset
- `reward/reward_model.py` — LSTM+CNN reward predictor
- `util/distributed_utils.py` — DDP helpers, checkpointing, gradient sync

**Configuration:** Hydra YAML configs in `configs/`. Override via CLI: `python train_vae.py --config-name=X key=value`.

**Third-party:** `third_party/splatt3r/` — Splatt3r submodule for 3D reconstruction (requires separate checkpoint download and optional CUDA kernel compilation).

## Data Format

Gaussian point clouds use 14 dimensions per point: XYZ (3) + covariance/scale/rotation (7) + SH coefficients (4). Default point cloud size is 4096 points. DROID dataset uses RLDS format with action dim 10.

## PoC Training Plan (single GPU, DROID-100)

Goal: reproduce Splatt3r point cloud reconstruction through an AE, then a VAE, before wiring into the DiT stage.

### Two-run sequence
1. **AE run** — `configs/vae/transformer.yaml` has `use_kl: false`. This routes to the deterministic `AutoEncoder` in `encoder/models_ae.py`. Validates that the encode/decode loop can reconstruct Splatt3r point clouds.
2. **VAE run** — flip `use_kl: true` in `configs/vae/transformer.yaml`. This routes to `KLAutoEncoder`, which adds mean/logvar projection and a KL bottleneck. The latent space produced here is what DiT will later operate on.

### Single-GPU targets (RTX 3090, DROID-100)
- 10 epochs is sufficient — loss converges well before epoch 10 (~0.0001 MSE)
- ~3.75s/step, ~1007 steps/epoch → ~10 hours total
- Splatt3r runs per-step (not pre-cached); fine for PoC, optimize later if needed

### Shuffle buffer caveat
`shuffle_buffer_size` in `configs/dataset/droid.yaml` controls both the tf.data shuffle window AND the val split size. The RLDS pipeline does `.take(shuffle_buffer_size)` for validation (`processor/rlds/dataset.py:591`), so this value caps how many val samples exist. Set to 1000 for DROID-100. When scaling to full DROID, increase to 10k–50k.

### Key configs
- `configs/train_vae_single_gpu.yaml` — training hyperparams (epochs, batch size, lr)
- `configs/dataset/droid.yaml` — dataset params (shuffle buffer, segment length, image size)
- `configs/vae/transformer.yaml` — model architecture (use_kl toggle, depth, latent dim)
