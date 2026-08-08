"""Roll out the GWM raw-feature DiT on held-out NWM episodes and write SH->RGB + rendered RGB predictions."""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

CTX_GWM, CTX_EVAL, HORIZON, SEG, RES = 2, 4, 64, 68, 128
CKPT_DIR = REPO / "outputs/2026-08-04/20-44-14/logs/nwm_scene_50k/checkpoints"
C0 = 0.28209479177387814


def sh_rgb(g14):  # [14,H,W] -> [H,W,3] u8, direct DC-band readout
    return ((0.5 + C0 * g14[10:13]).clamp(0, 1) * 255).round().byte().permute(1, 2, 0).cpu().numpy()


def load_model():
    GlobalHydra.instance().clear()
    with initialize(config_path="../configs", version_base=None):
        cfg = compose(config_name="train_nwm_scene")
    from gaussianwm.gwm_predictor import GaussianPredictor
    model = GaussianPredictor(cfg.world_model).cuda().eval()
    ckpt = max(CKPT_DIR.rglob("model*.pt"), key=lambda p: p.stat().st_mtime)
    model.model.load_state_dict(torch.load(ckpt, map_location="cuda"))  # strict
    print(f"loaded (strict) {ckpt}")
    return model, str(ckpt)


def make_renderer():
    # eval_dit.ipynb psnrs_render path (identity c2w, DC-band SH), except focal: the
    # notebook's fixed focal=W mismatches Splatt3r's implied intrinsics on this data
    # (oracle GT-splat re-render 6.8dB vs 14.5dB), so LS-fit fx/fy per frame from the
    # predicted pointmap itself
    from src.pixelsplat_src.cuda_splatting import render_cuda
    from utils.geometry import build_covariance, normalize_intrinsics
    c2w = torch.eye(4, device="cuda").unsqueeze(0)
    near = torch.tensor([0.1], device="cuda")
    far = torch.tensor([100.0], device="cuda")
    bg = torch.zeros(1, 3, device="cuda")
    cc = (RES - 1) / 2
    u = torch.arange(RES, device="cuda").float().repeat(RES) - cc
    v = torch.arange(RES, device="cuda").float().repeat_interleave(RES) - cc

    def render(g14):  # [14,H,W] -> [H,W,3] u8
        g = g14.reshape(14, RES * RES).T.contiguous()
        x, y = g[:, 0] / g[:, 2], g[:, 1] / g[:, 2]
        K = torch.eye(3, device="cuda")
        K[0, 0], K[1, 1], K[0, 2], K[1, 2] = (u * x).sum() / (x * x).sum(), (v * y).sum() / (y * y).sum(), cc, cc
        K = normalize_intrinsics(K.unsqueeze(0), (RES, RES))
        covs = build_covariance(g[:, 3:6], g[:, 6:10])
        img = render_cuda(c2w, K, near, far, (RES, RES), bg, g[:, 0:3].unsqueeze(0),
                          covs.unsqueeze(0), g[:, 10:13].unsqueeze(-1).unsqueeze(0),
                          g[:, 13].clamp(0, 1).unsqueeze(0))
        return (img[0].permute(1, 2, 0).clamp(0, 1) * 255).round().byte().detach().cpu().numpy()

    return render


def prep_frames(frames):  # (T,V,224,224,3) u8 -> (T,128,128,3) u8, as NWMEpisodeDataset._segment
    x = torch.from_numpy(frames[:, 0].copy()).permute(0, 3, 1, 2).float()
    x = F.interpolate(x, size=(RES, RES), mode="bilinear", align_corners=False, antialias=True)
    return x.round().clamp(0, 255).byte().permute(0, 2, 3, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/mnt/Data/nwm-baselines/preds/gwm.npz")
    ap.add_argument("--data", default="/mnt/Data/nwm-baselines/ogb_scene_ep_eval")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch", type=int, default=2)
    args = ap.parse_args()

    shards = sorted(Path(args.data).glob("ep*.npz"))[:args.limit]
    assert shards, f"no ep*.npz under {args.data}"
    model, ckpt = load_model()
    render = make_renderer()

    preds_sh = np.empty((len(shards), HORIZON, RES, RES, 3), dtype=np.uint8)
    preds_rd = np.empty_like(preds_sh)
    for b0 in range(0, len(shards), args.batch):
        tic = time.time()
        idx = list(range(b0, min(b0 + args.batch, len(shards))))
        ctxs, acts = [], []
        for i in idx:
            d = np.load(shards[i])
            t0 = (i * 37) % (201 - SEG)
            fr = prep_frames(d["frames"][t0 + CTX_EVAL - CTX_GWM:t0 + CTX_EVAL])
            ctxs.append(fr.permute(0, 3, 1, 2).float().reshape(-1, RES, RES))
            # training pairing: action[n+i-1] predicts frame[n+i] -> first action index t0+3
            acts.append(torch.from_numpy(d["actions"][t0 + CTX_EVAL - 1:t0 + CTX_EVAL - 1 + HORIZON].copy()))
        obs = torch.stack(ctxs).cuda()  # [B, 2*3, 128, 128] in 0..255
        act = torch.stack(acts).cuda()  # [B, 64, A]
        with torch.no_grad():
            obss, _, _ = model.rollout(obs, lambda o, t: act[:, t], HORIZON)
        for j, i in enumerate(idx):
            for t in range(HORIZON):
                g = obss[j, t + 1, 14:]  # newest frame = last 14ch
                preds_sh[i, t] = sh_rgb(g)
                preds_rd[i, t] = render(g)
        del obss, obs, act
        torch.cuda.empty_cache()
        print(f"ep[{idx[0]}:{idx[-1] + 1}] {time.time() - tic:.1f}s  "
              f"peak {torch.cuda.max_memory_allocated() / 2 ** 30:.2f}GiB")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out_rd = out.with_name(out.stem + "_render.npz")
    np.savez_compressed(out, pred=preds_sh, model="gwm-sh", ckpt=ckpt)
    np.savez_compressed(out_rd, pred=preds_rd, model="gwm-render", ckpt=ckpt)
    print(f"pred {preds_sh.shape} {preds_sh.dtype}\nwrote {out}\nwrote {out_rd}")


if __name__ == "__main__":
    main()
