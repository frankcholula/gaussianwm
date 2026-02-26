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
bash scripts/vae/train_vae_multi_gpu.sh

# Diffusion Transformer (multi-GPU)
bash scripts/dit/train_dit_multi_gpu.sh

# VAE dry-run test
bash scripts/vae/test_vae_dryrun.sh

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
1. **VAE stage** (optional): Images → Splatt3r (3D Gaussian points) → 3D Autoencoder → latent codes (2048 points via FPS downsampling)
2. **DiT stage**: Gaussian features + actions → Diffusion Transformer → predicted future Gaussians → SH→RGB extraction (NOT full Gaussian splatting rendering)

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

Gaussian point clouds use 14 dimensions per point:
- **XYZ position** (3 dims): 3D coordinates in camera frame
- **Scales** (3 dims): Gaussian ellipsoid radii along principal axes
- **Rotations** (4 dims): Quaternion encoding ellipsoid orientation
- **SH coefficients / RGB colors** (3 dims): **View-independent DC-band only** (not full spherical harmonics)
  - Splatt3r outputs `pred['sh']` with shape `[B, H, W, 3, 1]` — RGB channels × 1 SH band (degree-0)
  - Converted to RGB via `RGB = 0.5 + C0 * SH_DC` (where C0=0.28209...) in `train_vae.py:60-61`
  - Higher-order SH bands (degrees 1-3) are **NOT** used — view-dependent appearance is discarded
  - This means the model learns **Lambertian-like appearance** (constant color per Gaussian)
- **Opacity** (1 dim): Alpha transparency [0, 1]

**Total: 3 + 3 + 4 + 3 + 1 = 14 dimensions per point**

**Point cloud representation:**
- **DiT training/inference** (`use_vae=false`, default): 16,384 Gaussians (128×128 grid, one per pixel, no downsampling)
  - Stored as `[14, 128, 128]` tensor (channels × height × width)
  - Splatt3r produces one Gaussian per pixel from 128×128 input images
  - The model operates on this dense grid without FPS downsampling
- **VAE training** (`use_vae=true`): 2048 or 4096 Gaussians (FPS downsampled point cloud)
  - Stored as `[N, 14]` tensor where N = `point_cloud_size` from config
  - Used only if training the optional 3D autoencoder stage

**Evaluation methodology:**
- **Training loss**: Computed on all 14 Gaussian dimensions (including XYZ geometry)
- **Inference evaluation**: Extracts only SH color channels (dims 10-13) via `rollout_obs[:, :, -3:]`
  - XYZ positions, scales, rotations, and opacity are predicted but NOT used for visualization
  - Colors are converted to RGB via `RGB = 0.5 + C0 * SH_DC` where C0=0.28209...
  - No Gaussian splatting rasterization is performed; the 3D structure is not utilized at test time
  - See `demo.py:103-105` for the official evaluation implementation

DROID dataset uses RLDS format with action dim 10 (xyz translation + rot6d rotation + gripper state).
