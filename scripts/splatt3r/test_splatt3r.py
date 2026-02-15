#!/usr/bin/env python3
"""
Splatt3r testing and PLY export.

Modes:
  --mode test   (default) Stereo reconstruction + point cloud validation + multi-view rendering
  --mode export Export Splatt3r outputs to PLY files for interactive 3D viewing
"""

import os
import sys
from pathlib import Path
import torch
import numpy as np
from termcolor import cprint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2

# Add project to path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
SPLATT3R_DIR = ROOT_DIR / "third_party/splatt3r"
sys.path.insert(0, str(SPLATT3R_DIR / "src/mast3r_src"))
sys.path.insert(0, str(SPLATT3R_DIR / "src/mast3r_src/dust3r"))
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(SPLATT3R_DIR))

from gaussianwm.processor.regressor import Splatt3rRegressor, get_gaussain_tensor
from gaussianwm.processor.rlds import make_interleaved_dataset
from gaussianwm.processor.datasets import droid_dataset_transform, robomimic_transform
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

import hydra
from omegaconf import DictConfig
import tensorflow as tf


def render_gaussians(points, image_height=128, image_width=128, camera_angle=0, device='cuda'):
    """Render Gaussian splats from a specific camera angle."""
    xyz = points[:, :3]
    opacity = torch.clamp(points[:, 3:4], 0, 1)
    scales = torch.clamp(points[:, 4:7], 1e-6, 10.0)
    rotations = points[:, 7:11]
    rotations = rotations / (rotations.norm(dim=-1, keepdim=True) + 1e-8)
    rgb = torch.clamp(points[:, 11:14], 0, 1)

    cam_center = xyz.mean(dim=0)
    cam_distance = (xyz - cam_center).norm(dim=1).mean() * 2.5

    angle_rad = camera_angle * np.pi / 180.0
    camera_center = cam_center + torch.tensor([
        cam_distance * np.sin(angle_rad),
        0,
        cam_distance * np.cos(angle_rad)
    ], device=device)

    forward = (cam_center - camera_center)
    forward = forward / forward.norm()
    up = torch.tensor([0, 1, 0], device=device, dtype=torch.float32)
    right = torch.cross(up, forward)
    right = right / right.norm()
    up = torch.cross(forward, right)

    view_matrix = torch.eye(4, device=device)
    view_matrix[:3, 0] = right
    view_matrix[:3, 1] = up
    view_matrix[:3, 2] = -forward
    view_matrix[:3, 3] = camera_center
    view_matrix = torch.inverse(view_matrix)

    fov_y = 50.0 * np.pi / 180.0
    aspect = image_width / image_height
    tan_half_fov = np.tan(fov_y / 2)
    znear, zfar = 0.01, 100.0

    projection_matrix = torch.zeros(4, 4, device=device)
    projection_matrix[0, 0] = 1.0 / (aspect * tan_half_fov)
    projection_matrix[1, 1] = 1.0 / tan_half_fov
    projection_matrix[2, 2] = zfar / (zfar - znear)
    projection_matrix[2, 3] = -(zfar * znear) / (zfar - znear)
    projection_matrix[3, 2] = 1.0

    full_proj = projection_matrix @ view_matrix

    raster_settings = GaussianRasterizationSettings(
        image_height=image_height,
        image_width=image_width,
        tanfovx=aspect * tan_half_fov,
        tanfovy=tan_half_fov,
        bg=torch.ones(3, device=device),
        scale_modifier=1.0,
        viewmatrix=view_matrix,
        projmatrix=full_proj,
        sh_degree=0,
        campos=camera_center,
        prefiltered=False,
        debug=False
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    rendered_image, _ = rasterizer(
        means3D=xyz,
        means2D=torch.zeros_like(xyz[:, :2]),
        shs=None,
        colors_precomp=rgb,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=None
    )

    return rendered_image


def load_dataset(cfg):
    """Load DROID dataset with stereo camera pairs."""
    BASE_DATASET_KWARGS = {
        "data_dir": cfg.dataset.data_path,
        "image_obs_keys": {"primary": "exterior_image_1_left", "secondary": "exterior_image_2_left"},
        "state_obs_keys": ["cartesian_position", "gripper_position"],
        "language_key": "language_instruction",
        "action_proprio_normalization_type": "bounds",
        "absolute_action_mask": [True] * 10,
        "action_normalization_mask": [True] * 9 + [False],
        "standardize_fn": droid_dataset_transform,
    }

    dataset, dataset_length, _ = make_interleaved_dataset(
        dataset_kwargs_list=[{"name": "droid_100", **BASE_DATASET_KWARGS}],
        sample_weights=[1.0],
        train=True,
        shuffle_buffer_size=10,
        batch_size=None,
        balance_weights=False,
        traj_transform_kwargs=dict(
            window_size=cfg.dataset.segment_length,
            future_action_window_size=15,
            subsample_length=100,
            skip_unlabeled=False,
        ),
        frame_transform_kwargs=dict(
            resize_size=dict(
                primary=[cfg.dataset.image_size, cfg.dataset.image_size],
                secondary=[cfg.dataset.image_size, cfg.dataset.image_size],
            ),
            num_parallel_calls=200,
        ),
        traj_transform_threads=48,
        traj_read_threads=48,
    )

    dataset = dataset.map(robomimic_transform, num_parallel_calls=48)
    return dataset, dataset_length


def extract_stereo_tensors(sample, device):
    """Extract and prepare stereo image tensors from a dataset sample."""
    left_img = sample['obs']['camera/image/varied_camera_1_left_image'][0]
    right_img = sample['obs']['camera/image/varied_camera_2_left_image'][0]

    left_tensor = torch.from_numpy(left_img).permute(2, 0, 1).unsqueeze(0).float()
    right_tensor = torch.from_numpy(right_img).permute(2, 0, 1).unsqueeze(0).float()

    if left_tensor.max() > 1.0:
        left_tensor /= 255.0
        right_tensor /= 255.0

    return left_tensor.to(device), right_tensor.to(device), left_img, right_img


def run_test(cfg, splatt3r, dataset, device, output_dir, num_samples=3):
    """Run comprehensive Splatt3r test with stereo input and multi-view rendering."""
    cprint(f"\nTesting {num_samples} samples with stereo input...", "yellow")

    with torch.no_grad():
        for i, sample in enumerate(dataset.as_numpy_iterator()):
            if i >= num_samples:
                break

            cprint(f"\n{'='*60}", "cyan")
            cprint(f"Sample {i+1}/{num_samples}", "cyan", attrs=["bold"])

            left_tensor, right_tensor, left_img, right_img = extract_stereo_tensors(sample, device)
            cprint(f"Left: {left_tensor.shape}, Right: {right_tensor.shape}", "white")

            pred_left, pred_right = splatt3r.forward(left_tensor, right_tensor)
            points_left = get_gaussain_tensor(pred_left)
            points_right = get_gaussain_tensor(pred_right)
            cprint(f"Left view: {points_left.shape[1]} Gaussians", "green")
            cprint(f"Right view: {points_right.shape[1]} Gaussians", "green")

            points = points_left[0]
            cprint(f"XYZ range: [{points[:, :3].min():.3f}, {points[:, :3].max():.3f}]", "white")
            cprint(f"Opacity: [{points[:, 3].min():.3f}, {points[:, 3].max():.3f}]", "white")
            cprint(f"RGB range: [{points[:, 11:14].min():.3f}, {points[:, 11:14].max():.3f}]", "white")

            # Create 2x5 comprehensive visualization
            fig = plt.figure(figsize=(20, 10))

            ax1 = plt.subplot(2, 5, 1)
            left_np = left_img if left_img.dtype == np.uint8 else (left_img * 255).astype(np.uint8)
            ax1.imshow(left_np)
            ax1.set_title("Left Camera Input")
            ax1.axis('off')

            ax2 = plt.subplot(2, 5, 2)
            right_np = right_img if right_img.dtype == np.uint8 else (right_img * 255).astype(np.uint8)
            ax2.imshow(right_np)
            ax2.set_title("Right Camera Input")
            ax2.axis('off')

            ax3 = plt.subplot(2, 5, 3, projection='3d')
            xyz = points[:, :3].cpu().numpy()
            rgb = np.clip(points[:, 11:14].cpu().numpy(), 0, 1)
            indices = np.random.choice(xyz.shape[0], min(2048, xyz.shape[0]), replace=False)
            ax3.scatter(xyz[indices, 0], xyz[indices, 1], xyz[indices, 2],
                       c=rgb[indices], s=1, alpha=0.6)
            ax3.set_title("3D Point Cloud")
            ax3.set_xlabel('X')
            ax3.set_ylabel('Y')
            ax3.set_zlabel('Z')

            angles = [0, 45, 90]
            for idx, angle in enumerate(angles):
                rendered = render_gaussians(points, camera_angle=angle, device=device)
                ax = plt.subplot(2, 5, 4 + idx)
                rendered_np = rendered.cpu().permute(1, 2, 0).numpy()
                ax.imshow(np.clip(rendered_np, 0, 1))
                ax.set_title(f"Rendered {angle}")
                ax.axis('off')

                if angle == 0:
                    ax_bottom = plt.subplot(2, 5, 9)
                    diff = np.abs(left_np.astype(float) / 255.0 - rendered_np)
                    ax_bottom.imshow(diff)
                    ax_bottom.set_title("Difference from Left")
                    ax_bottom.axis('off')

                    ax_bottom2 = plt.subplot(2, 5, 10)
                    diff_right = np.abs(right_np.astype(float) / 255.0 - rendered_np)
                    ax_bottom2.imshow(diff_right)
                    ax_bottom2.set_title("Difference from Right")
                    ax_bottom2.axis('off')

            plt.tight_layout()
            save_path = output_dir / f"sample_{i:03d}_comprehensive.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            cprint(f"Saved: {save_path.name}", "green")


def run_export(cfg, splatt3r, dataset, device, output_dir, num_samples=5):
    """Export Splatt3r outputs to PLY files for interactive 3D viewing."""
    import utils.export as export_utils

    cprint(f"\nExporting {num_samples} samples to PLY...", "yellow")

    with torch.no_grad():
        for i, sample in enumerate(dataset.as_numpy_iterator()):
            if i >= num_samples:
                break

            cprint(f"\n--- Sample {i+1}/{num_samples} ---", "cyan")

            left_tensor, right_tensor, left_img, right_img = extract_stereo_tensors(sample, device)
            pred_left, pred_right = splatt3r.forward(left_tensor, right_tensor)

            ply_path = output_dir / f"sample_{i:03d}.ply"
            export_utils.save_as_ply(pred_left, pred_right, str(ply_path))

            left_save = left_img if left_img.dtype == np.uint8 else (left_img * 255).astype(np.uint8)
            right_save = right_img if right_img.dtype == np.uint8 else (right_img * 255).astype(np.uint8)
            cv2.imwrite(str(output_dir / f"sample_{i:03d}_left.jpg"), cv2.cvtColor(left_save, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(output_dir / f"sample_{i:03d}_right.jpg"), cv2.cvtColor(right_save, cv2.COLOR_RGB2BGR))

            cprint(f"Exported: {ply_path.name} ({ply_path.stat().st_size // 1024} KB)", "green")

    cprint("\nView PLY files at:", "yellow")
    cprint("  https://projects.markkellogg.org/threejs/demo_gaussian_splats_3d.php", "blue")
    cprint("  https://playcanvas.com/supersplat/editor", "blue")


@hydra.main(config_path="../../configs", config_name="train_vae", version_base=None)
def main(cfg: DictConfig):
    import argparse
    # Hydra consumes its own args; parse --mode from sys.argv before Hydra strips them
    mode = "test"
    for i, arg in enumerate(sys.argv):
        if arg == "--mode" and i + 1 < len(sys.argv):
            mode = sys.argv[i + 1]
        elif arg.startswith("--mode="):
            mode = arg.split("=", 1)[1]

    cprint("=" * 80, "cyan")
    cprint(f"Splatt3r — mode: {mode}", "cyan", attrs=["bold"])
    cprint("=" * 80, "cyan")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(f"test_outputs/splatt3r_{mode}")
    output_dir.mkdir(parents=True, exist_ok=True)

    cprint(f"Device: {device}", "blue")
    cprint(f"Output: {output_dir.absolute()}", "blue")

    splatt3r = Splatt3rRegressor().to(device).eval()
    cprint("Splatt3r loaded", "green")

    dataset, dataset_length = load_dataset(cfg)
    cprint(f"Dataset loaded: {dataset_length} samples", "green")

    if mode == "export":
        run_export(cfg, splatt3r, dataset, device, output_dir)
    else:
        run_test(cfg, splatt3r, dataset, device, output_dir)

    cprint(f"\nDone! Results: {output_dir.absolute()}", "green")


if __name__ == "__main__":
    main()
