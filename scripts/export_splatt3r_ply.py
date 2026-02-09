#!/usr/bin/env python3
"""
Export DROID samples to PLY format for interactive viewing.
Uses splatt3r's official export utility.
"""

import os
import sys
from pathlib import Path
import torch
import numpy as np
from termcolor import cprint

# Add paths
ROOT_DIR = Path(__file__).resolve().parent.parent
SPLATT3R_DIR = ROOT_DIR / "third_party/splatt3r"
sys.path.insert(0, str(SPLATT3R_DIR / "src/mast3r_src"))
sys.path.insert(0, str(SPLATT3R_DIR / "src/mast3r_src/dust3r"))
sys.path.insert(0, str(ROOT_DIR))

from gaussianwm.processor.regressor import Splatt3rRegressor
import hydra
from omegaconf import DictConfig

# Import splatt3r's export utility
sys.path.insert(0, str(SPLATT3R_DIR))
import utils.export as export_utils


@hydra.main(config_path="../configs", config_name="train_vae", version_base=None)
def main(cfg: DictConfig):
    cprint("=" * 80, "cyan")
    cprint("Exporting DROID Samples to Interactive PLY Format", "cyan", attrs=["bold"])
    cprint("=" * 80, "cyan")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path("test_outputs/splatt3r_ply")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cprint(f"\n📍 Device: {device}", "blue")
    cprint(f"📁 Output: {output_dir.absolute()}", "blue")
    
    # Load Splatt3r
    cprint("\n🔧 Loading Splatt3r...", "yellow")
    splatt3r = Splatt3rRegressor().to(device).eval()
    cprint("✅ Splatt3r loaded", "green")
    
    # Load dataset with both cameras
    cprint("\n📦 Loading DROID dataset...", "yellow")
    from gaussianwm.processor.rlds import make_interleaved_dataset
    from gaussianwm.processor.datasets import droid_dataset_transform, robomimic_transform
    
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
    cprint(f"✅ Dataset loaded: {dataset_length} samples", "green")
    
    # Export samples
    num_samples = 5
    cprint(f"\n🎨 Exporting {num_samples} samples to PLY...", "yellow")
    
    with torch.no_grad():
        for i, sample in enumerate(dataset.as_numpy_iterator()):
            if i >= num_samples:
                break
                
            cprint(f"\n--- Sample {i+1}/{num_samples} ---", "cyan")
            
            # Extract both camera views
            left_img = sample['obs']['camera/image/varied_camera_1_left_image'][0]
            right_img = sample['obs']['camera/image/varied_camera_2_left_image'][0]
            
            # Prepare tensors
            left_tensor = torch.from_numpy(left_img).permute(2, 0, 1).unsqueeze(0).float()
            right_tensor = torch.from_numpy(right_img).permute(2, 0, 1).unsqueeze(0).float()
            
            if left_tensor.max() > 1.0:
                left_tensor /= 255.0
                right_tensor /= 255.0
            
            left_tensor = left_tensor.to(device)
            right_tensor = right_tensor.to(device)
            
            # Run splatt3r
            cprint("Running Splatt3r (stereo)...", "yellow")
            pred_left, pred_right = splatt3r.forward(left_tensor, right_tensor)
            
            # Export to PLY
            ply_path = output_dir / f"sample_{i:03d}.ply"
            export_utils.save_as_ply(pred_left, pred_right, str(ply_path))
            
            # Save input images for reference
            import cv2
            left_img_path = output_dir / f"sample_{i:03d}_left.jpg"
            right_img_path = output_dir / f"sample_{i:03d}_right.jpg"
            
            left_save = left_img if left_img.dtype == np.uint8 else (left_img * 255).astype(np.uint8)
            right_save = right_img if right_img.dtype == np.uint8 else (right_img * 255).astype(np.uint8)
            
            cv2.imwrite(str(left_img_path), cv2.cvtColor(left_save, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(right_img_path), cv2.cvtColor(right_save, cv2.COLOR_RGB2BGR))
            
            cprint(f"✅ Exported: {ply_path.name} ({ply_path.stat().st_size // 1024} KB)", "green")
    
    cprint("\n" + "=" * 80, "cyan")
    cprint("✅ Export Complete!", "green", attrs=["bold"])
    cprint(f"📁 PLY files: {output_dir.absolute()}", "green")
    cprint("\n🌐 View your .ply files at:", "yellow")
    cprint("   • https://projects.markkellogg.org/threejs/demo_gaussian_splats_3d.php", "blue")
    cprint("   • https://playcanvas.com/supersplat/editor", "blue")
    cprint("   • Or use the splatt3r demo: cd third_party/splatt3r && python demo.py", "blue")
    cprint("=" * 80, "cyan")


if __name__ == "__main__":
    main()
