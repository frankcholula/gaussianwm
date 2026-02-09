#!/usr/bin/env python3
"""
Comprehensive Splatt3r test: stereo input, point clouds, and multi-view rendering.
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
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from gaussianwm.processor.regressor import Splatt3rRegressor
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

import hydra
from omegaconf import DictConfig
import tensorflow as tf

# Import dataset loading utilities
from gaussianwm.processor.rlds import make_interleaved_dataset
from gaussianwm.processor.datasets import droid_dataset_transform, robomimic_transform


def visualize_point_cloud(points, save_path=None, title="Point Cloud"):
    """Visualize point cloud with RGB colors."""
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()
    
    xyz = points[:, :3]
    rgb = points[:, 11:14]
    rgb = np.clip(rgb, 0, 1)
    
    # Sample for visualization
    if xyz.shape[0] > 2048:
        indices = np.random.choice(xyz.shape[0], 2048, replace=False)
        xyz, rgb = xyz[indices], rgb[indices]
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=rgb, s=1, alpha=0.6)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    # Equal aspect ratio
    max_range = np.array([xyz[:, 0].max()-xyz[:, 0].min(),
                          xyz[:, 1].max()-xyz[:, 1].min(),
                          xyz[:, 2].max()-xyz[:, 2].min()]).max() / 2.0
    mid_x, mid_y, mid_z = (xyz[:, 0].max()+xyz[:, 0].min()) * 0.5, \
                           (xyz[:, 1].max()+xyz[:, 1].min()) * 0.5, \
                           (xyz[:, 2].max()+xyz[:, 2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def render_gaussians(points, image_height=128, image_width=128, camera_angle=0, device='cuda'):
    """
    Render Gaussian splats from a specific camera angle.
    
    Args:
        points: [N, 14] Gaussian parameters
        camera_angle: Rotation angle in degrees around Y-axis
        device: Device to render on
    """
    xyz = points[:, :3]
    opacity = torch.clamp(points[:, 3:4], 0, 1)
    scales = torch.clamp(points[:, 4:7], 1e-6, 10.0)
    rotations = points[:, 7:11]
    rotations = rotations / (rotations.norm(dim=-1, keepdim=True) + 1e-8)
    rgb = torch.clamp(points[:, 11:14], 0, 1)
    
    # Camera setup: orbit around scene center
    cam_center = xyz.mean(dim=0)
    cam_distance = (xyz - cam_center).norm(dim=1).mean() * 2.5
    
    angle_rad = camera_angle * np.pi / 180.0
    camera_center = cam_center + torch.tensor([
        cam_distance * np.sin(angle_rad),
        0,
        cam_distance * np.cos(angle_rad)
    ], device=device)
    
    # View matrix (camera looking at center)
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
    
    # Projection matrix
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


@hydra.main(config_path="../configs", config_name="train_vae", version_base=None)
def main(cfg: DictConfig):
    cprint("=" * 80, "cyan")
    cprint("Splatt3r Comprehensive Test: Stereo Input + Multi-View Rendering", "cyan", attrs=["bold"])
    cprint("=" * 80, "cyan")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path("test_outputs/splatt3r_comprehensive")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cprint(f"\n📍 Device: {device}", "blue")
    cprint(f"📁 Output: {output_dir.absolute()}", "blue")
    
    # Load Splatt3r
    cprint("\n🔧 Loading Splatt3r...", "yellow")
    splatt3r = Splatt3rRegressor().to(device).eval()
    cprint("✅ Splatt3r loaded", "green")
    
    # Load DROID dataset with BOTH cameras
    cprint("\n📦 Loading DROID dataset (stereo)...", "yellow")
    
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
    
    from gaussianwm.processor.rlds import make_interleaved_dataset
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
    
    # Apply robomimic transform
    dataset = dataset.map(robomimic_transform, num_parallel_calls=48)
    cprint(f"✅ Dataset loaded: {dataset_length} samples", "green")
    
    # Test on samples
    num_samples = 3
    cprint(f"\n🎨 Testing {num_samples} samples with stereo input...", "yellow")
    
    with torch.no_grad():
        for i, sample in enumerate(dataset.as_numpy_iterator()):
            if i >= num_samples:
                break
                
            cprint(f"\n{'='*60}", "cyan")
            cprint(f"Sample {i+1}/{num_samples}", "cyan", attrs=["bold"])
            cprint(f"{'='*60}", "cyan")
            
            # Extract both camera views
            left_img = sample['obs']['camera/image/varied_camera_1_left_image'][0]  # First frame
            right_img = sample['obs']['camera/image/varied_camera_2_left_image'][0]
            
            # Prepare tensors
            left_tensor = torch.from_numpy(left_img).permute(2, 0, 1).unsqueeze(0).float()
            right_tensor = torch.from_numpy(right_img).permute(2, 0, 1).unsqueeze(0).float()
            
            # Normalize if needed
            if left_tensor.max() > 1.0:
                left_tensor /= 255.0
                right_tensor /= 255.0
            
            left_tensor = left_tensor.to(device)
            right_tensor = right_tensor.to(device)
            
            cprint(f"Left image: {left_tensor.shape}, range [{left_tensor.min():.3f}, {left_tensor.max():.3f}]", "white")
            cprint(f"Right image: {right_tensor.shape}, range [{right_tensor.min():.3f}, {right_tensor.max():.3f}]", "white")
            
            # Run splatt3r with STEREO input
            cprint("\n🔥 Running Splatt3r (stereo mode)...", "yellow")
            pred_left, pred_right = splatt3r.forward(left_tensor, right_tensor)
            
            # Extract Gaussian parameters
            from gaussianwm.processor.regressor import get_gaussain_tensor
            points_left = get_gaussain_tensor(pred_left)
            points_right = get_gaussain_tensor(pred_right)
            cprint(f"✅ Left view: {points_left.shape[1]} Gaussians", "green")
            cprint(f"✅ Right view: {points_right.shape[1]} Gaussians", "green")
            
            # Use left view Gaussians for visualization
            points = points_left[0]
            
            # Statistics
            cprint(f"\n📊 Point Cloud Statistics:", "magenta")
            cprint(f"  XYZ range: [{points[:, :3].min():.3f}, {points[:, :3].max():.3f}]", "white")
            cprint(f"  Opacity: [{points[:, 3].min():.3f}, {points[:, 3].max():.3f}]", "white")
            cprint(f"  RGB range: [{points[:, 11:14].min():.3f}, {points[:, 11:14].max():.3f}]", "white")
            
            # Create comprehensive visualization
            cprint(f"\n🎨 Creating visualizations...", "yellow")
            
            fig = plt.figure(figsize=(20, 10))
            
            # Row 1: Input images
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
            
            # Point cloud visualization
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
            
            # Rendered views from different angles
            angles = [0, 45, 90]
            for idx, angle in enumerate(angles):
                cprint(f"  Rendering view at {angle}°...", "white")
                rendered = render_gaussians(points, camera_angle=angle, device=device)
                
                ax = plt.subplot(2, 5, 4 + idx)
                rendered_np = rendered.cpu().permute(1, 2, 0).numpy()
                ax.imshow(np.clip(rendered_np, 0, 1))
                ax.set_title(f"Rendered {angle}°")
                ax.axis('off')
                
                # Bottom row: comparison with input for 0° view
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
            
            cprint(f"💾 Saved: {save_path.name}", "green")
    
    cprint("\n" + "=" * 80, "cyan")
    cprint("✅ Test Complete!", "green", attrs=["bold"])
    cprint(f"📁 Results: {output_dir.absolute()}", "green")
    cprint("=" * 80, "cyan")


if __name__ == "__main__":
    main()
