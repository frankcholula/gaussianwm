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

### AE Run Results (completed)
- **Config**: `train_vae_single_gpu` with `vae.use_kl=false`
- **Model**: `AutoEncoder(depth=4, dim=64, num_latents=64, output_dim=14, num_inputs=2048)`, 0.81M params
- **Training**: 10 epochs, batch_size=32, lr=1.25e-5 (base lr 1e-4, scaled by eff_batch_size/256), 5-epoch warmup, MSELoss
- **GPU**: RTX 3090, ~17.6GB VRAM, ~3.75s/step, 10.57 hours total
- **Loss progression** (MSE, per-epoch average):
  - Epoch 0: 0.0494 (rapid drop from 0.59 initial)
  - Epochs 1–3: ~0.014 (plateau during warmup phase)
  - Epoch 4: 0.0042 (warmup ends, learning rate fully ramped)
  - Epoch 9: 0.0014 (final)
- **Checkpoint**: `logs/vae_single_gpu/checkpoint-9.pth`
- **Reconstruction notebook**: `notebooks/ae_reconstruction.ipynb`
- **Evaluation** (single-sample, eval mode):
  - Overall MSE: 0.000725 (better than training avg — expected for single sample)
  - Best dims: scales (0.000013–0.000018), opacity (0.000108) — near-perfect
  - Moderate dims: x/y position (0.0002), SH colors (0.0006–0.0019)
  - Worst dims: z/depth (0.001373), rotations (0.0006–0.0015)
  - Qualitative: XYZ spatial structure preserved, but sharp distributions (scales, SH bimodal peaks, opacity near 1.0) get blurred — typical MSE-trained AE behavior
- **Verdict**: AE reconstruction working. Loss still decreasing at epoch 9. Sufficient to validate the encode/decode pipeline. See "Next steps" below for VAE recommendations.

### Next steps: VAE training recommendations
- **More epochs first**: Train AE to 20 epochs before switching to VAE — loss was clearly still decreasing at epoch 9 (0.0014 → likely ~0.0007 by epoch 20 based on the trend). Resume from checkpoint-9: `resume=logs/vae_single_gpu/checkpoint-9.pth train.epochs=20`
- **Consider num_latents=128**: Current 64 latents gives ~7x compression (2048×14 → 64×64). Doubling to 128 latents would halve compression and likely improve reconstruction of fine details (rotations, SH colors) at modest compute cost.
- **VAE kl_weight**: Hardcoded at 1e-3 in `train_vae.py:42`. Start there but monitor reconstruction vs KL tradeoff — if reconstruction degrades too much, try 1e-4.
- **dim=64 is small**: 0.81M params is tiny. For VAE, the KL bottleneck further constrains capacity. If reconstruction quality drops significantly with `use_kl=true`, try dim=128 (will increase params to ~3M).
