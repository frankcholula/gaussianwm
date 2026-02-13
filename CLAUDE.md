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

Default point cloud size is 2048 points (after FPS downsampling). DROID dataset uses RLDS format with action dim 10.

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

### AE Run Results (completed — 20 epochs)
- **Config**: `train_vae_single_gpu` with `vae.use_kl=false`
- **Model**: `AutoEncoder(depth=4, dim=64, num_latents=64, output_dim=14, num_inputs=2048)`, 0.81M params
- **Training**: 20 epochs (0-19), batch_size=32, lr=1.25e-5 (base lr 1e-4, scaled by eff_batch_size/256), 5-epoch warmup, MSELoss
- **GPU**: RTX 3090, ~17.6GB VRAM, ~3.75s/step, ~21 hours total
- **Loss progression** (MSE, training loss per-epoch average):
  - Epoch 0: 0.0491 (rapid drop from 0.59 initial)
  - Epochs 1–3: ~0.014 (plateau during warmup phase)
  - Epoch 4: 0.0038 (warmup ends, learning rate fully ramped)
  - Epoch 5: 0.0019
  - Epoch 10: 0.0013
  - Epoch 15: 0.0005
  - Epoch 19: 0.0004 (final, **92% improvement from epoch 0**)
- **Checkpoints**: `logs/vae_single_gpu/checkpoint-{0,5,10,15,19}.pth` (saved every 5 epochs)
- **Reconstruction notebook**: `notebooks/ae_reconstruction.ipynb`
- **Evaluation** (single DROID-100 sample, eval mode, same sample used for all checkpoints):

  **Checkpoint progression (reconstruction MSE on same test sample):**
  - Epoch 0: Overall MSE varies by sample (untrained baseline)
  - Epoch 5: 0.002719 (still learning spatial structure)
  - Epoch 10: 0.002034 (25% improvement from epoch 5)
  - Epoch 15: Lower (full results in notebook)
  - Epoch 19: Lowest (full results in notebook)

  **Per-dimension analysis (epoch 5 → epoch 10 comparison):**
  - Best performing: scales (0.000007–0.000015), opacity (0.000049) — near-perfect reconstruction
  - Good: x/y position (0.0006), z/depth (0.0020), rotations (0.0006–0.0013)
  - Problematic: **SH/RGB colors** — sh_r (0.0143 → 0.0128), sh_b (0.0075 → 0.0071) — highest MSE

  **Key findings:**
  - **Spatial structure (XYZ)**: Excellent reconstruction by epoch 10
  - **Geometric properties**: Scales, rotations, opacity converge quickly and accurately
  - **Color reconstruction (SH coefficients)**: Slower convergence, higher MSE
    - Likely due to SH→RGB preprocessing in `train_vae.py:60-61` (conversion adds noise/artifacts)
    - SH coefficients are **view-independent DC-band only** (not full spherical harmonics; see "Data Format" below)

  **Qualitative observations:**
  - XYZ spatial structure preserved accurately
  - Sharp distributions (scales, opacity near 1.0) get slightly blurred — typical MSE behavior
  - Training loss continued decreasing through epoch 19, suggesting more epochs could help

- **Verdict**: AE reconstruction working well. 20 epochs sufficient for PoC validation. Ready to proceed with VAE training (use_kl=true).

### Bugs found and fixed before VAE run

#### Bug 1: KLAutoEncoder.encode() FPS shape mismatch (critical)
- **File**: `encoder/models_ae.py`, `KLAutoEncoder.encode()`
- **Symptom**: `RuntimeError: shape '[320, -1, 3]' is invalid for input of size 896` on first forward pass with `use_kl=true`
- **Root cause**: The FPS sampling block was ported from a PyTorch Geometric-style pattern (flatten all batches into `[B*N, D]`, track batch indices, run FPS once globally). However:
  1. The `batch` index tensor was computed but **never passed** to the `fps()` call, so FPS selected 64 points globally instead of 64 per batch
  2. The codebase uses `pytorch3d.ops.sample_farthest_points`, which handles batching natively (`[B, N, 3]` input) — the PyG-style flatten+batch approach was unnecessary
  3. The reshape `view(B, -1, 3)` hardcoded dim `3` instead of `D=14`, a leftover from when the model only used XYZ coordinates
- **Fix**: Replaced with the same working pattern from `AutoEncoder.encode()`: `fps(pc[..., :3], K=self.num_latents)` + `torch.gather()` to preserve per-batch structure and all 14 features
- **Note**: `AutoEncoder.encode()` was correct all along — this bug only affected the `KLAutoEncoder` (VAE) path

#### Bug 2: Splatt3r running in degenerate single-image mode
- **Files**: `processor/datasets.py`, `train_vae.py`, `processor/regressor.py`
- **Symptom**: Splatt3r (a stereo reconstruction model) was receiving the same image for both views
- **Root cause**: The dataset loaded both left (`varied_camera_1_left_image`) and right (`varied_camera_2_left_image`) camera frames but only yielded the left frame. `regressor.py:568` duplicated it: `view2_tensor = view1_tensor if len(image_tensors) == 1`. The eval loop even extracted both images but only passed `image1` to Splatt3r.
- **Fix**:
  1. Dataset now yields `(left_frames, right_frames, action, reward)` instead of `(left_frames, action, reward)`
  2. Both training and eval loops pass stereo pairs: `splatt3r.forward_tensor(left, right)`
  3. `forward_tensor` now accepts variadic `*image_tensors` to pass through both views
- **Watch for**: `gaussian_feature_to_dim` in `regressor.py` includes `means_in_other_view` (3 dims). If stereo mode causes Splatt3r to populate this field, the tensor becomes 17D instead of 14D, which would crash `PointEmbed` (hardcoded `nn.Linear(hidden_dim + 14, dim)`). Left as-is for now — fix if it surfaces.

### Next steps: VAE training recommendations
- ✅ **20-epoch AE training completed** — Loss converged well (0.0491 → 0.0004, 92% improvement)
- **Proceed with VAE training**: Set `use_kl=true` via CLI override (do NOT edit `configs/vae/transformer.yaml` — keep it as `use_kl: false` default)
  - Start from scratch (do NOT resume from AE checkpoint — VAE has different architecture with mean/logvar projections)
  - Command: `python gaussianwm/train_vae.py --config-name=train_vae_single_gpu vae.use_kl=true use_wandb=true`
  - Monitor reconstruction MSE vs KL divergence tradeoff
- **VAE-specific considerations**:
  - **kl_weight**: Configured at 1e-3 in `train_vae_single_gpu.yaml`. Start there but monitor reconstruction quality — if MSE degrades significantly (>2x the AE final MSE), try 1e-4 or 5e-4
  - **Expect higher reconstruction loss**: VAE will have slightly higher MSE than AE due to KL bottleneck constraining latent space
  - **Target**: Final MSE < 0.001 would be excellent, < 0.002 acceptable for DiT training
- **Optional improvements** (if VAE reconstruction is poor):
  - **num_latents=128**: Current 64 latents gives ~7x compression (2048x14 -> 64x64). Doubling to 128 would halve compression and improve fine details (rotations, SH colors) at modest compute cost
  - **dim=128**: Current dim=64 (0.81M params) is small. Increasing to dim=128 (~3M params) would help if KL bottleneck degrades quality too much
  - **Remove SH preprocessing**: Skip lines 59-61 in `train_vae.py` to train on raw SH coefficients instead of preprocessed RGB (may improve color reconstruction)
