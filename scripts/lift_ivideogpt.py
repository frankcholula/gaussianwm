"""Post-hoc 3D lift baseline: Splatt3r-lift iVideoGPT rollout frames, render at rig v2 + v5..v8."""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from gaussianwm.processor.regressor import Splatt3rRegressor  # noqa: E402 (appends splatt3r sys.paths)
from src.pixelsplat_src.cuda_splatting import render_cuda  # noqa: E402
from utils.geometry import build_covariance, normalize_intrinsics  # noqa: E402

LIFT, RES, SEG, HORIZON = 224, 128, 68, 64
TGT_VIEWS = (5, 6, 7, 8)
CC = (LIFT - 1) / 2


def up224(frames64):  # (T,64,64,3) u8 -> (T,3,224,224) float [0,1], reverse of down64
    f = torch.from_numpy(frames64).permute(0, 3, 1, 2).float()
    f = F.interpolate(f, size=(LIFT, LIFT), mode="bilinear", align_corners=False, antialias=True)
    return f.round().clamp(0, 255) / 255.0


def down128(frames224):  # (V,224,224,3) u8 -> (V,3,128,128) float [0,1], as NWMEpisodeDataset._segment
    f = torch.from_numpy(frames224.copy()).permute(0, 3, 1, 2).float()
    f = F.interpolate(f, size=(RES, RES), mode="bilinear", align_corners=False, antialias=True)
    return f.round().clamp(0, 255) / 255.0


class Lifter:
    def __init__(self):
        self.model = Splatt3rRegressor().cuda().eval()
        self.u = torch.arange(LIFT, device="cuda").float().repeat(LIFT) - CC
        self.v = torch.arange(LIFT, device="cuda").float().repeat_interleave(LIFT) - CC
        self.near = torch.tensor([0.1], device="cuda")
        self.far = torch.tensor([100.0], device="cuda")
        self.bg = torch.zeros(1, 3, device="cuda")
        self.eye = torch.eye(4, device="cuda")

    @torch.no_grad()
    def lift(self, img):  # (3,224,224) [0,1] -> g [N,14], covs [N,3,3] (identical-pair input as _process_obs)
        g = self.model.forward_tensor(img.unsqueeze(0).cuda())[0][0].float()
        return g, build_covariance(g[:, 3:6], g[:, 6:10])

    def ls_K(self, g):  # per-frame LS-fit focal from the pointmap, as rollout_nwm_eval
        x, y = g[:, 0] / g[:, 2], g[:, 1] / g[:, 2]
        K = torch.eye(3, device="cuda")
        K[0, 0], K[1, 1], K[0, 2], K[1, 2] = (self.u * x).sum() / (x * x).sum(), (self.v * y).sum() / (y * y).sum(), CC, CC
        return normalize_intrinsics(K.unsqueeze(0), (LIFT, LIFT))[0]

    @torch.no_grad()
    def render(self, g, covs, c2w, K):  # -> (128,128,3) float [0,1]
        img = render_cuda(c2w.unsqueeze(0), K.unsqueeze(0), self.near, self.far, (RES, RES), self.bg,
                          g[:, 0:3].unsqueeze(0), covs.unsqueeze(0),
                          g[:, 10:13].unsqueeze(-1).unsqueeze(0), g[:, 13].clamp(0, 1).unsqueeze(0))
        return img[0].permute(1, 2, 0).clamp(0, 1)


def psnr(a, b):
    return -10 * torch.log10(((a - b) ** 2).mean()).item()


@torch.no_grad()
def calibrate(lf, g, covs, rels, K, gt128):  # fit per-episode scale s on ctx-time GT only
    def score(s):
        ps = []
        for r, gtv in zip(rels, gt128):
            c = r.clone()
            c[:3, 3] /= s
            ps.append(psnr(lf.render(g, covs, c, K), gtv))
        return float(np.mean(ps)), ps
    grid = np.geomspace(0.05, 20.0, 31)
    for _ in range(3):
        best = max(grid, key=lambda s: score(s)[0])
        step = grid[1] / grid[0]
        grid = np.geomspace(best / step, best * step, 15)
    return best, score(best)[1]


def to_u8(img):
    return (img * 255).round().clamp(0, 255).byte().cpu().numpy()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preds", default="/mnt/Data/nwm-baselines/preds/ivideogpt.npz")
    ap.add_argument("--data", default="/mnt/Data/nwm-baselines/ogb_scene_ep_eval")
    ap.add_argument("--tgt", default="/mnt/Data/nwm-baselines/ogb_scene_ep_eval_tgt")
    ap.add_argument("--out-dir", default="/mnt/Data/nwm-baselines/preds")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--panel", default=None, help="write smoke panel png (rows=episodes)")
    args = ap.parse_args()

    pred64 = np.load(args.preds)["pred"]  # (256,64,64,64,3) u8
    shards = sorted(Path(args.data).glob("ep*.npz"))[: args.limit]
    n = min(len(shards), pred64.shape[0])
    shards = shards[:n]

    poses = np.load(Path(args.data) / "poses.npz")
    c2w = torch.from_numpy(poses["c2w"]).float().cuda()  # (9,4,4) OpenCV world-frame
    rels = [torch.linalg.inv(c2w[2]) @ c2w[v] for v in TGT_VIEWS]  # splat frame = v2 cam frame
    # all views render with the frame's own LS-fit K: the splat is self-consistent with its
    # implied intrinsics (fx_norm ~1.5-1.7), not the rig K (1.207) -- rig K costs 4-5dB at ctx calib

    lf = Lifter()
    out = {v: np.empty((n, HORIZON, RES, RES, 3), dtype=np.uint8) for v in (2, *TGT_VIEWS)}
    panel_rows = []
    for i, shard in enumerate(shards):
        tic = time.time()
        t0 = (i * 37) % (201 - SEG)
        ctx224 = torch.from_numpy(np.load(shard)["frames"][t0 + 3, 0].copy()).permute(2, 0, 1).float() / 255.0
        gt128 = [x.permute(1, 2, 0).cuda() for x in down128(np.load(Path(args.tgt) / shard.name)["frames"][t0 + 3])]

        g, covs = lf.lift(ctx224)
        s, cal = calibrate(lf, g, covs, rels, lf.ls_K(g), gt128)
        srels = []
        for r in rels:
            c = r.clone()
            c[:3, 3] /= s
            srels.append(c)
        if args.panel:
            ctx_v5 = lf.render(g, covs, srels[0], lf.ls_K(g))
        del g, covs

        for t in range(HORIZON):
            g, covs = lf.lift(up224(pred64[i, t : t + 1])[0])
            K = lf.ls_K(g)
            out[2][i, t] = to_u8(lf.render(g, covs, lf.eye, K))
            for v, c in zip(TGT_VIEWS, srels):
                out[v][i, t] = to_u8(lf.render(g, covs, c, K))
            if args.panel and t == 0:
                panel_rows.append([to_u8(gt128[0]), to_u8(ctx_v5), out[5][i, 0],
                                   to_u8(down128(np.load(Path(args.tgt) / shard.name)["frames"][t0 + 4])[0].permute(1, 2, 0))])
            del g, covs
        torch.cuda.empty_cache()
        print(f"ep{i} t0={t0} s={s:.4f} calPSNR v5..v8=[{', '.join(f'{p:.2f}' for p in cal)}] "
              f"{time.time() - tic:.1f}s peak {torch.cuda.max_memory_allocated() / 2**30:.2f}GiB", flush=True)

    for v in out:
        p = Path(args.out_dir) / f"ivgpt_splat_v{v}.npz"
        np.savez_compressed(p, pred=out[v], model="ivideogpt+splatt3r", view=v)
        print(f"wrote {p} {out[v].shape}")

    if args.panel and panel_rows:
        from PIL import Image
        grid = np.concatenate([np.concatenate(r, axis=1) for r in panel_rows], axis=0)
        Image.fromarray(grid).save(args.panel)
        print(f"panel (GT v5 ctx | ctx render v5 | pred[0] render v5 | GT v5 t0+4) -> {args.panel}")


if __name__ == "__main__":
    main()
