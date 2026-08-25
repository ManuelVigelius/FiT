"""Quadtree structures from ImageNet latents, tokenized for next-token training.

Each item is one latent file turned into the 17-token sequence of
`structure.py`. The tree is produced by the same GT-variance planner the oracle
loader uses — `gt_variance_grids` on the clean latent x0, then
`plan_from_gt_variance` — so the prior learns the distribution of trees the
compressor is actually trained against.

No timestep anywhere
--------------------
The prior runs ONCE, before the diffusion process starts, so there is no t to
condition on and no noise to add: the tree is read off the clean latent and
`quadtree_threshold` is the only knob. In particular there is no
`max_res_schedule` here — that schedule exists on the compressor's path to cap
resolution as a function of t, and importing it would make the targets a
t-marginal mixture of coarse and fine trees that the prior has no way to tell
apart. Targets are the UNCAPPED tree (`min_leaf_size=1`).

One consequence worth knowing: an uncapped GT tree spends as many tokens as the
variance asks for, up to 256 for a busy image. If generated trees need to fit a
budget, lower `quadtree_threshold` (or filter on the sampler side via the token
counts `sample_plans` returns).

Everything is CPU-side and cheap (three avg-pools plus a small recursion), so it
runs in the DataLoader workers.
"""

import glob
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from fit.data.in1k_quadtree_latent_dataset import build_shard_indices
from fit.data.in1k_gt_quadtree_latent_dataset import (gt_variance_grids,
                                                      plan_from_gt_variance)
from fit.quadtree_compression.variance_prediction import load_latent

from quadtree_prior import structure as S


class QuadtreeStructureDataset(Dataset):
    """Yields (inputs, targets, loss_mask, label) for one latent file.

        inputs    (17,) long   embedding ids to feed (BOS / previous class / COVERED)
        targets   (17,) long   class to predict, IGNORE_INDEX where covered
        loss_mask (17,) bool   True where supervised
        label     ()    long   ImageNet class, or -1 when the file has none

    Deterministic: the tree is a function of the clean latent and `threshold`
    alone, so an item does not change between epochs and needs no RNG.

    Sharding is handled by the sampler passed to the DataLoader (see
    `build_shard_indices`), matching the rest of the codebase.
    """

    def __init__(self, root_dir, crop=32, threshold=0.5):
        super().__init__()
        self.root_dir = root_dir
        self.crop = crop
        self.threshold = float(threshold)
        self.files = self._discover_files(root_dir)
        if not self.files:
            raise RuntimeError(f"no .safetensors feature files found under {root_dir}")

    @staticmethod
    def _discover_files(root_dir):
        # Same layout convention as the quadtree latent loaders.
        candidates = []
        for sub in ('greater_than_256_crop', 'greater_than_512_crop'):
            d = os.path.join(root_dir, sub)
            if os.path.isdir(d):
                candidates.extend(sorted(glob.glob(os.path.join(d, '*.safetensors'))))
        if not candidates:
            candidates = sorted(glob.glob(
                os.path.join(root_dir, '**', '*.safetensors'), recursive=True))
        return candidates

    def __len__(self):
        return len(self.files)

    def _center_crop(self, latent):
        """Center-crop (or reflect-pad) a (C, Hs, Ws) latent to (C, crop, crop)."""
        c = self.crop
        _, Hs, Ws = latent.shape
        ph, pw = max(0, c - Hs), max(0, c - Ws)
        if ph or pw:
            latent = F.pad(latent[None], (0, pw, 0, ph), mode='reflect')[0]
            _, Hs, Ws = latent.shape
        top = (Hs - c) // 2
        left = (Ws - c) // 2
        return latent[:, top:top + c, left:left + c]

    def __getitem__(self, idx):
        x0 = self._center_crop(load_latent(self.files[idx])).float()

        # Uncapped: the clean latent's variance is the only thing deciding the
        # tree. See the module docstring on why no resolution schedule applies.
        levels, positions, sizes = plan_from_gt_variance(
            gt_variance_grids(x0), self.threshold, self.crop, min_leaf_size=1)
        grid = S.sizes_grid_from_plan(
            dict(levels=levels, positions=positions, sizes=sizes),
            patch_grid=self.crop // 2)
        targets, inputs, loss_mask = S.encode_sizes(grid)

        return dict(inputs=inputs, targets=targets, loss_mask=loss_mask,
                    label=self._load_label(idx))

    def _load_label(self, idx):
        from safetensors import safe_open
        try:
            with safe_open(self.files[idx], 'pt') as f:
                if 'label' in f.keys():
                    return f.get_tensor('label').reshape(()).long()
        except Exception:
            pass
        return torch.tensor(-1, dtype=torch.long)


class QuadtreeStructureLoader:
    """Builds the per-epoch, per-rank DataLoader for structure training.

    Unlike the compressor loaders there is no token budget to pack against — every
    sequence is exactly `SEQ_LEN` long — so this is an ordinary fixed-batch
    DataLoader.
    """

    def __init__(self, train):
        self.cfg = train
        self.root_dir = train.data_path
        self.crop = getattr(train, 'crop', 32)
        self.batch_size = getattr(train.loader, 'batch_size', 256)
        self.num_workers = getattr(train.loader, 'num_workers', 4)
        self.seed = getattr(train, 'seed', 42)

        self.dataset = QuadtreeStructureDataset(
            self.root_dir, crop=self.crop,
            threshold=getattr(train, 'quadtree_threshold', 0.5))

    def train_len(self):
        return len(self.dataset)

    def train_iter(self, epoch=0, rank=0, world_size=1):
        indices = build_shard_indices(
            len(self.dataset), rank, world_size, epoch, seed=self.seed)
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            sampler=indices,                  # disjoint, pre-shuffled shard
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            prefetch_factor=2 if self.num_workers > 0 else None,
        )
