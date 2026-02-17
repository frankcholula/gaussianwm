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

### VAE Run Results (completed — 20 epochs)
- **Config**: `train_vae_single_gpu` with `vae.use_kl=true` (CLI override)
- **Model**: `KLAutoEncoder(depth=4, dim=64, num_latents=64, latent_dim=64, output_dim=14, num_inputs=2048)`, 0.83M params (825,166)
- **Training**: 20 epochs (0-19), batch_size=32, lr=1.25e-5, 5-epoch warmup, MSELoss + KL (kl_weight=1e-3)
- **Command**: `python gaussianwm/train_vae.py --config-name=train_vae_single_gpu vae.use_kl=true wandb.name=vae_baseline_20ep use_wandb=true`
- **GPU**: RTX 3090, ~21 hours total
- **Loss progression** (training loss per-epoch average):
  - Epoch 0: loss_vol=0.0504, loss_kl=0.0818, total=0.0505
  - Epoch 4: loss_vol=0.0099, loss_kl=0.0824, total=0.0099 (warmup ends)
  - Epoch 5: loss_vol=0.0021, loss_kl=0.0885, total=0.0021
  - Epoch 10: loss_vol=0.0013, loss_kl=0.0332, total=0.0013
  - Epoch 15: loss_vol=0.00045, loss_kl=0.0111, total=0.00046
  - Epoch 19: loss_vol=0.00031, loss_kl=0.00586, total=0.00032
- **Checkpoints**: `logs/vae_single_gpu/checkpoint-{0,5,10,15,19}.pth`
- **Reconstruction notebook**: `notebooks/vae_reconstruction.ipynb`

  **Key findings:**
  - **Reconstruction quality**: Final loss_vol (0.00031) comparable to AE final (0.00036) — KL bottleneck didn't hurt reconstruction
  - **KL convergence**: Decreased monotonically 0.082 → 0.006 over 20 epochs — no posterior collapse, latent space is well-structured
  - **kl_weight=1e-3 was appropriate**: Reconstruction quality preserved while regularizing the latent space
  - **Scale reconstruction (known issue)**: Scale dims have the lowest absolute MSE (0.000001–0.000018) but worst *relative* reconstruction — their value range is ~0.001–0.016, so even tiny errors represent ~28% of the full range. Two contributing factors: (1) MSE loss is magnitude-biased, so the optimizer under-prioritizes tiny-valued dimensions; (2) FPS in `train_vae.py:62` operates on all 14D without normalization, making scale variation invisible to the sampling criterion (vs `AutoEncoder.encode()` which uses XYZ-only FPS). Potential fixes: per-dimension normalization, per-dimension loss weighting, or XYZ-only FPS in preprocessing. Defer until after full pipeline validation.

- **Verdict**: VAE training successful. Latent space is well-structured (low KL, good reconstruction). Ready to proceed with DiT training.

### Paper's Training Paradigm (discovered via code + paper analysis)

The original GWM paper uses a **two-phase approach** (not purely two-stage or purely joint):

1. **Phase 1: VAE pretraining** (200 epochs) — `train_vae.py` with `configs/train_vae.yaml`
   - Model: `ae_d64_m64` (deterministic AE, `use_kl: false`), batch_size=16, 4 GPUs
   - Purpose: warm-start the latent space so DiT doesn't train against random encodings
2. **Phase 2: Joint VAE+DiT training** (1M steps) — `train_diffusion.py` with `configs/train_gwm.yaml`
   - Loads pretrained VAE via `pretrained_model_path` in `gwm.yaml:95`
   - `train.update_tokenizer: true` — VAE continues to be fine-tuned alongside DiT
   - Separate optimizers: `tok_lr` (VAE), `model_lr` (DiT), `reward_model_lr`
   - Dreamer-style motivation: joint training shapes the latent space for both reconstruction AND prediction

**Mono vs stereo**: The paper uses a **single Realsense D435i camera** (mono mode). Splatt3r receives the same image duplicated for both views internally (`regressor.py:568`). On the `reproduce` branch, we added a `use_stereo` toggle in `configs/dataset/droid.yaml` to support both modes.

### Known Gaps in the Codebase

#### 1. `train_diffusion.py` has no eval loop
- Config defines `eval.eval_every: 1000` but it's **never used** in the training loop
- Only trains + saves checkpoints — no validation loss, no rollouts during training

#### 2. `train_diffusion.py` batch unpacking bug
- Dataset yields 4-tuple `(left, right, action, reward)` after Bug 2 fix
- `gwm_predictor.py:199-203` interprets 4-item batch as `(obs, action, reward, pad_mask)` — right_frames get treated as actions
- **Symptom**: `RuntimeError: mat1 and mat2 shapes cannot be multiplied (65536x3 and 10x384)`
- Fix: update batch unpacking + add mono/stereo handling

#### 3. Missing SH preprocessing in `gwm_predictor.py`
- `train_vae.py:60-62` converts SH→RGB: `colors = 0.5 + C0 * points[..., -4:-1]; points[..., -4:-1] = colors / 255.0`
- `gwm_predictor._process_obs()` at line 176 passes raw Splatt3r output to VAE **without** this conversion
- Would cause VAE to receive differently-scaled color values vs what it was pretrained on

#### 4. GS-mode evaluation pipeline not connected
- `demo.py` and `gwm_predictor.rollout()` return **latent representations** when `use_gs=True`
- `demo.py:105` takes last 3 channels as RGB — only works for pixel mode (`use_gs=False`)
- **Missing steps for GS mode**:
  1. VAE decode: latent → Gaussian params (2048 × 14D) — `vae.decode()` exists but never called in eval
  2. Camera parameters: `_process_obs()` discards them (`points, _ = splatt3r.forward_tensor(...)`)
  3. CUDA rasterizer: `render_cuda()` in `third_party/splatt3r/src/pixelsplat_src/cuda_splatting.py` exists but not wired to eval
- **Result**: running demo in GS mode would produce colorful noise (latent values interpreted as pixel RGB), not meaningful video

#### 5. FPS downsampling missing in `gwm_predictor._process_obs()`
- `train_vae.py:64` does `points, _ = fps(points, K=cfg.model.point_cloud_size)` before encoding
- `_process_obs()` passes full Splatt3r output (potentially 4096+ points) directly to VAE
- VAE expects exactly `num_inputs=2048` points — would crash with assertion error

### Next steps: Joint training (faithful reproduction)

#### Phase 1: Fix bugs for joint training
- Fix batch unpacking in `gwm_predictor.py` for 4-tuple dataset format
- Add SH preprocessing to `_process_obs()`
- Add FPS downsampling to `_process_obs()`
- Add mono/stereo support to `_process_obs()` (use `use_stereo` config toggle)

#### Phase 2: Wire up GS evaluation pipeline (latent → video)

The current `demo.py` only works for pixel mode (`use_gs=False`). For GS mode, the rollout returns latent splats, not RGB. There are 4 missing steps to go from latent → rendered video:

1. **Reshape grid → tokens** — Inverse of `_process_obs()` reshape at `gwm_predictor.py:183`
   - Current: latent stored as `[B, latent_dim, nh, nw]` spatial grid
   - Need: `latent.permute(0, 2, 3, 1).reshape(B, N, latent_dim)` to get `[B, N, latent_dim]` tokens
   - Where: add to `rollout()` or a new `_decode_latent()` helper in `gwm_predictor.py`

2. **VAE decode** — `vae.decode()` exists in `encoder/models_ae.py` but never called in eval
   - Call: `gaussian_params = self.vae.decode(latent_tokens)` → `[B, 2048, 14]`
   - The 14D output contains: xyz (3), scales (3), rotations (4), RGB colors (3), opacity (1)
   - Note: colors are in preprocessed RGB space (not raw SH), since VAE was trained on preprocessed points

3. **Save camera parameters** — Currently discarded at `_process_obs()` line 176
   - `points, _ = self.splatt3r.forward_tensor(obs_flat)` — the `_` contains camera info
   - Fix: `points, camera_info = self.splatt3r.forward_tensor(obs_flat)` and store `camera_info`
   - Only need camera from the initial observation (save once, reuse for all predicted frames)
   - Extract extrinsics + intrinsics from `camera_info` for the rasterizer

4. **CUDA rasterizer** — `render_cuda()` in `third_party/splatt3r/src/pixelsplat_src/cuda_splatting.py`
   - Parse 14D Gaussian params into: means, covariances, SH coefficients, opacities
   - Call: `render_cuda(extrinsics, intrinsics, near, far, image_shape, bg_color, means, covs, sh, opacities)`
   - Returns: RGB image `[H, W, 3]` rendered from the saved camera viewpoint
   - Alternative: may need to convert preprocessed RGB back to SH format for the rasterizer, or modify render call to accept direct colors

5. **Stitch frames → video** — Already works via `imageio.mimsave()` in `demo.py:44`

**Implementation location**: Best approach is a new method `_latent_to_rgb()` in `GaussianPredictor` that wraps steps 1-4, callable from both `rollout()` and `demo.py`.

**Eval loop in `train_diffusion.py`**: Config already defines `eval.eval_every: 1000` (line 40 in `train_gwm.yaml`). Need to add eval block in the training loop that calls `model.rollout()` + `_latent_to_rgb()` on val data, computes PSNR/SSIM/LPIPS vs GT frames, and optionally saves sample videos.

#### Policy extraction
- `gwm_predictor.rollout()` is the policy integration point — takes a `policy(obs, t) → action` callable and rolls out the world model autoregressively
- `demo.py` currently uses `replay_policy` (ground-truth actions) for evaluation only
- **Options for learned policy**:
  1. **CEM/MPPI planning**: Optimize action sequences at inference time using world model + reward model as simulator (no policy network needed)
  2. **Dreamer-style**: Train policy network end-to-end with DiT as differentiable world model
  3. **Inverse dynamics**: Train action predictor on (s_t, s_{t+1}) pairs from dataset

#### Benchmarks and evaluation
- **World model quality**: Reconstruction MSE (point cloud space), rendered image metrics (PSNR/SSIM/LPIPS), FVD on predicted video sequences
- **Video generation**: Predicted latents → VAE decode → Gaussian params → CUDA rasterizer → RGB frames → MP4/GIF
- **Policy quality** (once policy exists): Task success rate on DROID manipulation tasks, action prediction MSE vs expert
- **Computational**: Inference latency per rollout step, GPU memory footprint
