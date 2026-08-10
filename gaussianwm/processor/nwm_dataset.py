"""NWM episode shards (ep*.npz: frames (T,V,H,W,3) u8, actions (T,A)) as GWM training segments."""
import functools
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import IterableDataset


class NWMEpisodeDataset(IterableDataset):
    def __init__(self, data_path, segment_length=10, image_size=128, view=0,
                 split="train", val_ratio=0.05, seed=42, data_path2=None):
        shards = sorted(Path(data_path).glob("ep*.npz"))
        assert shards, f"no ep*.npz under {data_path}"
        perm = np.random.default_rng(seed).permutation(len(shards))
        n_val = max(1, int(len(shards) * val_ratio))
        keep = sorted(perm[n_val:] if split == "train" else perm[:n_val])
        self.shards = [shards[i] for i in keep]
        self.shards2 = None
        if data_path2 is not None:  # second view dir, same episode names
            self.shards2 = [Path(data_path2) / s.name for s in self.shards]
            missing = [s for s in self.shards2 if not s.exists()]
            assert not missing, f"missing in {data_path2}: {missing[:3]}"
        self.segment_length = segment_length
        self.image_size = image_size
        self.view = view
        self.split = split
        self.rng = np.random.default_rng(seed + (0 if split == "train" else 1))
        self.lengths = [np.load(s)["actions"].shape[0] for s in self.shards]
        self.windows = [(i, t) for i, n in enumerate(self.lengths)
                        for t in range(0, n - segment_length + 1, segment_length)]

    @functools.lru_cache(maxsize=64)
    def _episode(self, i):
        d = np.load(self.shards[i])
        frames2 = None
        if self.shards2 is not None:
            frames2 = np.load(self.shards2[i])["frames"]
            assert frames2.shape[0] == d["frames"].shape[0]
        return d["frames"], d["actions"], frames2

    def __len__(self):
        return len(self.windows)

    def _view_seg(self, frames, t):
        seg = torch.from_numpy(frames[t:t + self.segment_length, self.view].copy())  # (T,H,W,3) u8
        if seg.shape[1] != self.image_size:
            seg = F.interpolate(seg.permute(0, 3, 1, 2).float(), size=(self.image_size,) * 2,
                                mode="bilinear", align_corners=False, antialias=True)
            seg = seg.round().clamp(0, 255).byte().permute(0, 2, 3, 1)
        return seg

    def _segment(self, i, t):
        frames, actions, frames2 = self._episode(i)
        seg = self._view_seg(frames, t)
        if frames2 is not None:  # (T,2,H,W,3), primary view first
            seg = torch.stack([seg, self._view_seg(frames2, t)], dim=1)
        act = torch.from_numpy(actions[t:t + self.segment_length].copy())
        return seg, act, torch.zeros(self.segment_length, 1)

    def __iter__(self):
        if self.split == "train":  # infinite random windows
            while True:
                i = int(self.rng.integers(len(self.shards)))
                t = int(self.rng.integers(self.lengths[i] - self.segment_length + 1))
                yield self._segment(i, t)
        else:  # deterministic finite pass
            for i, t in self.windows:
                yield self._segment(i, t)
