"""Probe whether GWM's released VAE decoder uses its latents or just its GT queries."""
import torch
import torch.nn.functional as F
import einops
import sys
sys.path.insert(0, "gaussianwm")
import util.tensor_utils as TensorUtils

from gaussianwm.encoder.models_ae import create_autoencoder, fps
from gaussianwm.processor.datasets import DroidDataset
from gaussianwm.processor.regressor import Splatt3rRegressor

device = torch.device("cuda")
torch.manual_seed(0)

N_SCENES = int(sys.argv[1]) if len(sys.argv) > 1 else 8
SH_C0 = 0.28209479177387814

ds = DroidDataset(
    data_path="/mnt/Data", segment_length=10, context_length=2, action_dim=10,
    image_size=128, split="train", shuffle_buffer_size=10,
)
frames = []
for i, (obs, _, _) in enumerate(ds):
    if i >= N_SCENES:
        break
    left, _ = obs
    frames.append(left[0])  # one frame per episode for scene diversity
images = torch.stack(frames)  # [B, H, W, C] uint8
images = TensorUtils.to_device(TensorUtils.to_float(images), device)
images = einops.rearrange(images, "b h w c -> b c h w")

splatt3r = Splatt3rRegressor().to(device)
chunks = []
with torch.no_grad():
    for img_chunk in images.split(8):
        pts, _ = splatt3r.forward_tensor(img_chunk)
        colors = 0.5 + SH_C0 * pts[..., -4:-1]
        pts[..., -4:-1] = colors / 255.0
        pts, _ = fps(pts, K=2048)
        chunks.append(pts)
points = torch.cat(chunks).to(device)
print(f"points: {tuple(points.shape)}  std={points.std():.4f}")


def probe(ckpt_path, use_kl):
    model = create_autoencoder(
        depth=4, dim=64, M=64, latent_dim=64, output_dim=14, N=2048,
        deterministic=not use_kl,
    ).to(device)
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = sd.get("model", sd)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"\n== {ckpt_path}  (use_kl={use_kl})")
    if missing or unexpected:
        print(f"   state_dict missing={missing} unexpected={unexpected}")
    model.eval()

    with torch.no_grad():
        if use_kl:
            _, z = model.encode(points)
        else:
            z = model.encode(points)

        def mse(latents):
            recon = model.decode(latents, queries=points)
            return F.mse_loss(recon, points).item()

        matched = mse(z)
        shuffled = mse(torch.roll(z, 1, dims=0))
        zeroed = mse(torch.zeros_like(z))
        noise = mse(torch.randn_like(z))
    mean_baseline = points.var().item()  # MSE of predicting the global mean

    print(f"   matched latents : {matched:.6f}")
    print(f"   shuffled latents: {shuffled:.6f}  (x{shuffled / matched:.2f})")
    print(f"   zeroed latents  : {zeroed:.6f}  (x{zeroed / matched:.2f})")
    print(f"   random latents  : {noise:.6f}  (x{noise / matched:.2f})")
    print(f"   predict-mean    : {mean_baseline:.6f}  (x{mean_baseline / matched:.2f})")


probe("logs/vae_single_gpu/checkpoint-19.pth", use_kl=True)
probe("logs/ae_single_gpu/checkpoint-19.pth", use_kl=False)
