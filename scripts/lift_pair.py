"""2-view pred-pair Splatt3r lift: true image pair (v2 + vb) per timestep, render at rig v2 + v5..v8."""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from gaussianwm.processor.regressor import Splatt3rRegressor  # noqa: E402
from src.pixelsplat_src.cuda_splatting import render_cuda  # noqa: E402
from utils.geometry import build_covariance, normalize_intrinsics  # noqa: E402

LIFT, RES, SEG, HORIZON = 224, 128, 68, 64
TGT_VIEWS = (5, 6, 7, 8)
CC = (LIFT - 1) / 2


def up224(img):  # (H,W,3) u8 -> (3,224,224) float [0,1]
    t = torch.from_numpy(img.copy()).permute(2, 0, 1)[None].float()
    if img.shape[0] != LIFT:
        t = F.interpolate(t, size=(LIFT, LIFT), mode="bilinear", align_corners=False, antialias=True)
    return t.round().clamp(0, 255)[0] / 255.0


def down128(frames224):  # (V,224,224,3) u8 -> (V,3,128,128) float [0,1]
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
    def lift_pair(self, im_a, im_b):  # two (3,224,224) [0,1]; both heads in view-A frame
        g1, g2 = self.model.forward_tensor(im_a.unsqueeze(0).cuda(), im_b.unsqueeze(0).cuda())
        g = torch.cat([g1[0], g2[0]]).float()
        return g, build_covariance(g[:, 3:6], g[:, 6:10]), g1[0].float()

    def ls_K(self, g1):  # LS-fit focal on head-1 pointmap (view-A pixels)
        x, y = g1[:, 0] / g1[:, 2], g1[:, 1] / g1[:, 2]
        K = torch.eye(3, device="cuda")
        K[0, 0], K[1, 1], K[0, 2], K[1, 2] = (self.u * x).sum() / (x * x).sum(), (self.v * y).sum() / (y * y).sum(), CC, CC
        return normalize_intrinsics(K.unsqueeze(0), (LIFT, LIFT))[0]

    @torch.no_grad()
    def render(self, g, covs, c2w, K):
        img = render_cuda(c2w.unsqueeze(0), K.unsqueeze(0), self.near, self.far, (RES, RES), self.bg,
                          g[:, 0:3].unsqueeze(0), covs.unsqueeze(0),
                          g[:, 10:13].unsqueeze(-1).unsqueeze(0), g[:, 13].clamp(0, 1).unsqueeze(0))
        return img[0].permute(1, 2, 0).clamp(0, 1)


def psnr(a, b):
    return -10 * torch.log10(((a - b) ** 2).mean()).item()


@torch.no_grad()
def calibrate(lf, g, covs, rels, K, gt128):  # per-episode scale on ctx-time GT only
    def score(s):
        ps = []
        for r, gtv in zip(rels, gt128):
            c = r.clone()
            c[:3, 3] /= s
            ps.append(psnr(lf.render(g, covs, c, K), gtv))
        return float(np.mean(ps))
    grid = np.geomspace(0.05, 20.0, 31)
    for _ in range(3):
        best = max(grid, key=score)
        step = grid[1] / grid[0]
        grid = np.geomspace(best / step, best * step, 15)
    return best


def to_u8(img):
    return (img * 255).round().clamp(0, 255).byte().cpu().numpy()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--vb", type=int, required=True)       # rig index of pair view B (A is always v2)
    ap.add_argument("--preds-a", default=None)             # 64px pred npz for v2 (ignored with --gt-input)
    ap.add_argument("--preds-b", default=None)
    ap.add_argument("--data", required=True)               # v2 GT export (ctx + poses.npz)
    ap.add_argument("--dir-b", required=True)              # GT export dir for view B
    ap.add_argument("--idx-b", type=int, default=0)
    ap.add_argument("--tgt", required=True)                # v5-8 GT dir (calibration)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--gt-input", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    shards = sorted(Path(args.data).glob("ep*.npz"))[: args.limit]
    shards_b = sorted(Path(args.dir_b).glob("ep*.npz"))[: args.limit]
    n = len(shards)
    pa = None if args.gt_input else np.load(args.preds_a)["pred"]
    pb = None if args.gt_input else np.load(args.preds_b)["pred"]

    poses = np.load(Path(args.data) / "poses.npz")
    c2w = torch.from_numpy(poses["c2w"]).float().cuda()
    rels = [torch.linalg.inv(c2w[2]) @ c2w[v] for v in TGT_VIEWS]  # splat frame = v2 cam frame

    lf = Lifter()
    out = {v: np.empty((n, HORIZON, RES, RES, 3), dtype=np.uint8) for v in (2, *TGT_VIEWS)}
    for i, (sa, sb) in enumerate(zip(shards, shards_b)):
        tic = time.time()
        t0 = (i * 37) % (201 - SEG)
        fa, fb = np.load(sa)["frames"][:, 0], np.load(sb)["frames"][:, args.idx_b]
        gt128 = [x.permute(1, 2, 0).cuda() for x in down128(np.load(Path(args.tgt) / sa.name)["frames"][t0 + 3])]

        g, covs, g1 = lf.lift_pair(up224(fa[t0 + 3]), up224(fb[t0 + 3]))
        s = calibrate(lf, g, covs, rels, lf.ls_K(g1), gt128)
        srels = []
        for r in rels:
            c = r.clone()
            c[:3, 3] /= s
            srels.append(c)
        del g, covs

        for t in range(HORIZON):
            if args.gt_input:
                ia, ib = up224(fa[t0 + 4 + t]), up224(fb[t0 + 4 + t])
            else:
                ia, ib = up224(pa[i, t]), up224(pb[i, t])
            g, covs, g1 = lf.lift_pair(ia, ib)
            K = lf.ls_K(g1)
            out[2][i, t] = to_u8(lf.render(g, covs, lf.eye, K))
            for v, c in zip(TGT_VIEWS, srels):
                out[v][i, t] = to_u8(lf.render(g, covs, c, K))
            del g, covs
        torch.cuda.empty_cache()
        if i % 25 == 0:
            print(f"{i}/{n} s={s:.3f} {time.time() - tic:.1f}s/ep", flush=True)

    stem = f"{'gt2v' if args.gt_input else 'ivgpt2v'}_splat_p2{args.vb}"
    od = Path(args.out_dir); od.mkdir(parents=True, exist_ok=True)
    for v in (2, *TGT_VIEWS):
        np.savez_compressed(od / f"{stem}_v{v}.npz", pred=out[v], model=stem, view=v)
        print(f"done: {out[v].shape} -> {od / f'{stem}_v{v}.npz'}", flush=True)


if __name__ == "__main__":
    main()
