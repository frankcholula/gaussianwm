#!/usr/bin/env python3
"""Verify GWM Docker installation: imports, CUDA extensions, and CPU forward pass."""
import importlib
import os
import sys
import traceback


def check_import(module_name, description=None):
    desc = description or module_name
    try:
        mod = importlib.import_module(module_name)
        version = getattr(mod, "__version__", "")
        suffix = f" ({version})" if version else ""
        print(f"  [PASS] {desc}{suffix}")
        return True
    except Exception as e:
        print(f"  [FAIL] {desc}: {e}")
        return False


def check_attr(module_name, attr_name, description=None):
    desc = description or f"{module_name}.{attr_name}"
    try:
        mod = importlib.import_module(module_name)
        getattr(mod, attr_name)
        print(f"  [PASS] {desc}")
        return True
    except Exception as e:
        print(f"  [FAIL] {desc}: {e}")
        return False


def main():
    results = []

    # === Core Python dependencies ===
    print("\n=== Core Dependencies ===")
    for mod, desc in [
        ("torch", "PyTorch"),
        ("torchvision", "TorchVision"),
        ("numpy", "NumPy"),
        ("einops", "einops"),
        ("lightning", "PyTorch Lightning"),
        ("hydra", "Hydra"),
        ("omegaconf", "OmegaConf"),
        ("timm", "timm"),
        ("wandb", "W&B"),
        ("cv2", "OpenCV"),
        ("imageio", "imageio"),
        ("lpips", "LPIPS"),
        ("diffusers", "Diffusers"),
        ("tensorflow", "TensorFlow"),
        ("tensorflow_datasets", "TF Datasets"),
        ("safetensors", "safetensors"),
        ("termcolor", "termcolor"),
        ("dotmap", "DotMap"),
    ]:
        results.append(check_import(mod, desc))

    # === CUDA extensions ===
    print("\n=== CUDA Extensions ===")
    results.append(check_import("diff_gaussian_rasterization", "diff-gaussian-rasterization"))
    results.append(check_attr("diff_gaussian_rasterization", "GaussianRasterizer", "GaussianRasterizer class"))
    results.append(check_import("pytorch3d", "PyTorch3D"))
    results.append(check_attr("pytorch3d.ops", "sample_farthest_points", "pytorch3d FPS"))

    # === PyTorch CUDA status (informational) ===
    print("\n=== PyTorch CUDA Status ===")
    import torch

    print(f"  torch.version.cuda = {torch.version.cuda}")
    print(f"  torch.cuda.is_available() = {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    # === CroCo CUDA kernel ===
    print("\n=== CroCo CUDA Kernel ===")
    curope_dir = os.path.join(
        os.environ.get("GWM_PATH", "."),
        "third_party/splatt3r/src/mast3r_src/dust3r/croco/models/curope",
    )
    so_files = [f for f in os.listdir(curope_dir) if f.endswith(".so")] if os.path.isdir(curope_dir) else []
    if so_files:
        print(f"  [PASS] curope compiled: {so_files}")
        results.append(True)
    else:
        print(f"  [FAIL] no .so in {curope_dir}")
        results.append(False)

    # === GWM package imports ===
    print("\n=== GWM Package ===")
    gwm_path = os.environ.get("GWM_PATH", ".")
    for p in [
        gwm_path,
        os.path.join(gwm_path, "third_party/splatt3r"),
        os.path.join(gwm_path, "third_party/splatt3r/src/pixelsplat_src"),
        os.path.join(gwm_path, "third_party/splatt3r/src/mast3r_src"),
        os.path.join(gwm_path, "third_party/splatt3r/src/mast3r_src/dust3r"),
    ]:
        if p not in sys.path:
            sys.path.insert(0, p)

    results.append(check_import("gaussianwm", "gaussianwm package"))
    results.append(check_import("gaussianwm.encoder.models_ae", "encoder.models_ae"))
    results.append(check_import("gaussianwm.diffusion.models", "diffusion.models"))
    results.append(check_import("gaussianwm.diffusion.gaussian_diffusion", "diffusion.gaussian_diffusion"))
    results.append(check_import("gaussianwm.reward.reward_model", "reward.reward_model"))
    results.append(check_import("gaussianwm.util.distributed_utils", "util.distributed_utils"))

    # === Functional: AutoEncoder CPU forward pass ===
    print("\n=== Functional: AutoEncoder (CPU) ===")
    try:
        from gaussianwm.encoder.models_ae import create_autoencoder

        model = create_autoencoder(depth=2, dim=32, M=16, latent_dim=32, output_dim=14, N=64, deterministic=True)
        params = sum(p.numel() for p in model.parameters())
        x = torch.randn(2, 64, 14)
        out = model(x, x)
        assert "logits" in out and out["logits"].shape == (2, 64, 14)
        print(f"  [PASS] AE forward OK ({params} params), output {out['logits'].shape}")
        results.append(True)
    except Exception as e:
        print(f"  [FAIL] {e}")
        traceback.print_exc()
        results.append(False)

    # === Functional: KLAutoEncoder CPU forward pass ===
    print("\n=== Functional: KLAutoEncoder (CPU) ===")
    try:
        from gaussianwm.encoder.models_ae import create_autoencoder

        model_kl = create_autoencoder(depth=2, dim=32, M=16, latent_dim=32, output_dim=14, N=64, deterministic=False)
        params_kl = sum(p.numel() for p in model_kl.parameters())
        out_kl = model_kl(x, x)
        assert "logits" in out_kl and "kl" in out_kl
        print(f"  [PASS] KL-AE forward OK ({params_kl} params), kl={out_kl['kl'].mean().item():.6f}")
        results.append(True)
    except Exception as e:
        print(f"  [FAIL] {e}")
        traceback.print_exc()
        results.append(False)

    # === Summary ===
    passed = sum(results)
    total = len(results)
    print(f"\n{'=' * 50}")
    if passed == total:
        print(f"ALL {total} CHECKS PASSED")
    else:
        print(f"FAILED: {total - passed}/{total} checks failed")
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
