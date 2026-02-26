"""
Static scene diagnostic for the GWM DiT rollout.

Tests whether the DiT is predicting a genuinely dynamic scene or just
copying the context frame forward.

Three tests:
  1. Copy-forward baseline  — does DiT beat repeating the last context frame?
  2. Inter-frame pixel change — how much do consecutive pred frames change vs GT?
  3. Drift from context frame — cumulative drift from the starting frame.

Usage:
    python scripts/eval/static_scene_test.py
"""

import os, sys, types
sys.path.insert(0, os.path.expandvars("$GWM_PATH"))

import torch
import numpy as np
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
import tensorflow_datasets as tfds
import tensorflow as tf
import glob, json
from PIL import Image as PILImage
from skimage.metrics import peak_signal_noise_ratio as psnr

# ── Settings ──────────────────────────────────────────────────────────────────
EPISODE_IDX = 0
START_FRAME = 20    # context starts here (movement begins ~frame 20)
CONTEXT     = 2
HORIZON     = 8     # match training rollout_horizon
CAMERA      = "exterior_image_1_left"

# ── Config ────────────────────────────────────────────────────────────────────
GWM_PATH = os.environ["GWM_PATH"]

GlobalHydra.instance().clear()
with initialize(config_path="../../configs", version_base=None):
    cfg = compose(config_name="train_gwm", overrides=[
        "world_model.observation.use_gs=true",
        "world_model.reward.use_reward_model=false",
        "world_model.vae.use_vae=false",
        f"dataset.data_path={GWM_PATH}/data/",
    ])

# ── Load model ────────────────────────────────────────────────────────────────
from gaussianwm.gwm_predictor import GaussianPredictor

CKPT = f"{GWM_PATH}/logs/gwm/checkpoints/model_latest.pt"
print(f"Loading model from {CKPT} ...")
model = GaussianPredictor(cfg.world_model).cuda()
state_dict = torch.load(CKPT, map_location="cuda", weights_only=False)
model.model.load_state_dict(state_dict)
model.eval()

# Remove the [-1,1] clip bug from wrap_model_output
def _wrap_noclip(self, noisy_next_obs, model_output, cs):
    return cs.c_skip * noisy_next_obs + cs.c_out * model_output
model.model.wrap_model_output = types.MethodType(_wrap_noclip, model.model)
print("wrap_model_output patched (clip removed)")

# ── Load episode ──────────────────────────────────────────────────────────────
from gaussianwm.processor.datasets import euler_to_rmat, mat_to_rot6d

print(f"\nLoading episode {EPISODE_IDX} ...")
ds    = tfds.load("droid_100", data_dir=f"{GWM_PATH}/data/", split="train", with_info=False)
ep    = next(iter(ds))
steps = list(ep["steps"])
print(f"  {len(steps)} frames  ({len(steps)/15:.1f}s @ 15Hz)")

frames, raw_cart, raw_grip = [], [], []
for step in steps:
    img = PILImage.fromarray(step["observation"][CAMERA].numpy())
    frames.append(np.array(img.resize((128, 128), PILImage.BILINEAR)))
    raw_cart.append(step["action_dict"]["cartesian_position"].numpy())
    raw_grip.append(step["action_dict"]["gripper_position"].numpy())

# Build 10-dim action: xyz(3) + rot6d(6) + gripper(1)
cart_tf = tf.constant(np.stack(raw_cart), dtype=tf.float32)
grip_tf = tf.constant(np.stack(raw_grip), dtype=tf.float32)
actions_10d = tf.concat([
    cart_tf[:, :3],
    mat_to_rot6d(euler_to_rmat(cart_tf[:, 3:6])),
    grip_tf,
], axis=-1).numpy().astype(np.float32)

# Bounds normalization (q01/q99)
stats_file = sorted(glob.glob(f"{GWM_PATH}/data/droid_100/1.0.0/dataset_statistics_*.json"))[0]
with open(stats_file) as f:
    stats = json.load(f)
q01 = np.array(stats["action"]["q01"], dtype=np.float32)
q99 = np.array(stats["action"]["q99"], dtype=np.float32)
actions_norm = 2 * (actions_10d - q01) / (q99 - q01 + 1e-8) - 1
actions_norm[:, 9] = actions_10d[:, 9]   # gripper: not normalized
actions_norm = actions_norm[1:]           # match pipeline shift

obs    = torch.from_numpy(np.stack(frames))
action = torch.from_numpy(actions_norm)

# ── Rollout ───────────────────────────────────────────────────────────────────
print(f"\nRunning rollout: START_FRAME={START_FRAME}, CONTEXT={CONTEXT}, HORIZON={HORIZON}")
print(f"  Predicting frames {START_FRAME+CONTEXT}–{START_FRAME+CONTEXT+HORIZON-1}"
      f"  ({(START_FRAME+CONTEXT)/15:.2f}s – {(START_FRAME+CONTEXT+HORIZON-1)/15:.2f}s)")

context_input = obs[START_FRAME:START_FRAME+CONTEXT].permute(0,3,1,2).float()
obs_input     = context_input.reshape(1,-1,128,128).cuda()
gt_actions    = action[START_FRAME+CONTEXT:START_FRAME+CONTEXT+HORIZON].float().cuda()

def gt_policy(obs_t, t):
    return gt_actions[t].unsqueeze(0)

with torch.no_grad():
    rollout_obs, _, _ = model.rollout(obs_input, policy=gt_policy, horizon=HORIZON)

# ── Helpers ───────────────────────────────────────────────────────────────────
C0 = 0.28209479177387814

def gauss_to_rgb(g):
    return (0.5 + C0 * g[10:13]).clamp(0,1).permute(1,2,0).cpu().numpy()

def gt_rgb(t):
    return obs[t].float().numpy() / 255.

pred_frames  = [gauss_to_rgb(rollout_obs[0, t+1, 14:].clone()) for t in range(HORIZON)]
gt_frames_   = [gt_rgb(START_FRAME + CONTEXT + t)              for t in range(HORIZON)]
context_last = gt_rgb(START_FRAME + CONTEXT - 1)

psnrs_dit  = [psnr(gt_frames_[t], pred_frames[t], data_range=1.0) for t in range(HORIZON)]
psnrs_copy = [psnr(gt_frames_[t], context_last,   data_range=1.0) for t in range(HORIZON)]

# ── Test 1: copy-forward baseline ─────────────────────────────────────────────
print("\n── Test 1: Copy-forward baseline vs DiT (SH→RGB PSNR) ───────────────")
print(f"  {'frame':>5}  {'copy PSNR':>10}  {'DiT PSNR':>10}  {'DiT - copy':>10}")
for t in range(HORIZON):
    delta = psnrs_dit[t] - psnrs_copy[t]
    flag  = "✓" if delta > 0 else "✗ WORSE than copy"
    print(f"  t={START_FRAME+CONTEXT+t:2d}   {psnrs_copy[t]:>10.2f}  {psnrs_dit[t]:>10.2f}  {delta:>+10.2f} dB  {flag}")
print(f"\n  Mean copy: {np.mean(psnrs_copy):.2f} dB  |  "
      f"Mean DiT: {np.mean(psnrs_dit):.2f} dB  |  "
      f"Δ = {np.mean(psnrs_dit) - np.mean(psnrs_copy):+.2f} dB")

# ── Test 2: inter-frame pixel change ──────────────────────────────────────────
pred_diffs = [np.abs(pred_frames[t+1] - pred_frames[t]).mean() for t in range(HORIZON-1)]
gt_diffs   = [np.abs(gt_frames_[t+1]  - gt_frames_[t] ).mean() for t in range(HORIZON-1)]
mean_ratio = np.mean(pred_diffs) / (np.mean(gt_diffs) + 1e-8)

print("\n── Test 2: Inter-frame pixel change (mean |I[t+1]−I[t]|, 0–1 scale) ─")
print(f"  {'step':>8}  {'pred Δ':>10}  {'GT Δ':>10}  {'ratio':>8}")
for t in range(HORIZON-1):
    ratio = pred_diffs[t] / (gt_diffs[t] + 1e-8)
    print(f"  {START_FRAME+CONTEXT+t}→{START_FRAME+CONTEXT+t+1}"
          f"   {pred_diffs[t]:>10.5f}  {gt_diffs[t]:>10.5f}  {ratio:>8.3f}x")
print(f"\n  Mean pred Δ: {np.mean(pred_diffs):.5f}  |  "
      f"Mean GT Δ: {np.mean(gt_diffs):.5f}  |  "
      f"ratio: {mean_ratio:.3f}x")

# ── Test 3: drift from context frame ──────────────────────────────────────────
pred_vs_ctx = [np.abs(pred_frames[t] - context_last).mean() for t in range(HORIZON)]
gt_vs_ctx   = [np.abs(gt_frames_[t]  - context_last).mean() for t in range(HORIZON)]

print("\n── Test 3: Drift from last context frame (mean |I[t]−I_ctx|) ─────────")
print(f"  {'frame':>5}  {'pred drift':>12}  {'GT drift':>12}")
for t in range(HORIZON):
    print(f"  t={START_FRAME+CONTEXT+t:2d}   {pred_vs_ctx[t]:>12.5f}  {gt_vs_ctx[t]:>12.5f}")

# ── Verdict ───────────────────────────────────────────────────────────────────
print("\n── Verdict ──────────────────────────────────────────────────────────")
if mean_ratio < 0.05:
    print(f"STATIC: predicted frames are essentially frozen (pred/GT motion = {mean_ratio:.3f}x)")
elif mean_ratio < 0.3:
    print(f"MOSTLY STATIC: pred changes much less than GT (pred/GT motion = {mean_ratio:.3f}x)")
else:
    print(f"DYNAMIC: pred changes comparably to GT (pred/GT motion = {mean_ratio:.3f}x)")
