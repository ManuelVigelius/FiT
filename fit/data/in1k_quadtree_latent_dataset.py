"""
Variable-length quadtree-compressed latent dataset for varlen (packed) training.

This mirrors the varlen packing of `in1k_latent_dataset.py`, but the number of
tokens an image contributes is *not known ahead of time*: it depends on the
variance-guided quadtree compression (`quadtree_compression.compress`), which in
turn needs a forward pass of the `VariancePredictor`. That breaks the
pre-computed-length `TokenBudgetBatchSampler` used by the fixed-grid dataset — we
can't pack by length before we know the lengths.

Instead we pool *plans* and pack lazily:

    1.  A plain (sharded, shuffled) DataLoader yields batches of RAW noisy
        latents (x_t, x0, t, label) — cheap, length-agnostic.
    2.  `QuadtreePlanPool` runs the FROZEN `VariancePredictor` on the GPU and
        turns each image's variance grids into a quadtree *plan* — the tree
        structure and hence the exact token count, with NO latent values read.
    3.  Every step it pops whole images until the next would exceed
        `max_seq_len`, and hands that image batch (plus plans) to the caller.

The caller then runs the TRAINED compressor (`PredictiveVarianceCompressor`) on
exactly those images. This ordering is what keeps the learned compressor
trainable: because the predictor is frozen, plans can be computed far ahead at
any convenient batch size, but the trained encoder only ever runs on the images
that are in this step's sequence — so no encoder gradient is stranded when
`zero_grad` fires. The compressor's batch size is whatever the packing produced
and changes every step.

Whole images are never split across sequences (doc_id contiguity + the quadtree
token grouping are preserved); the sequence tail is zero-padded.

The plan layout (per image) comes from `quadtree_compression.plan_from_variance`:
    levels    (n_i,)     encoder level l, leaf side N == 2**l
    positions (n_i, 2)   (y, x) center in latent-pixel coords
    sizes     (n_i,)     leaf side (1 / 2 / 4 / 8)
"""

import math
import os

import torch
from torch.utils.data import DataLoader, Dataset

# `compress` lives in the sibling package `quadtree_compression`. Import defensively
# so this module can be imported for its dataset/loader pieces even if the compressor
# package path isn't wired up yet at import time.
try:
    from fit.quadtree_compression.quadtree_compression import (
        compress_from_variance, plan_from_variance)
    from fit.quadtree_compression.variance_prediction import (
        VariancePredictor, load_latent, predict_variance)  # load_latent: safetensors -> (4, 2H, 2W)
except Exception:  # pragma: no cover - allows partial import in dev
    compress_from_variance = None
    plan_from_variance = None
    VariancePredictor = None
    load_latent = None
    predict_variance = None


def _sample_t(snr_type: str, t0: float, t1: float, gen=None) -> torch.Tensor:
    """Sample a single flow-matching timestep in [t0, t1] as a 0-dim tensor."""
    if snr_type == 'uniform':
        return torch.rand((), generator=gen) * (t1 - t0) + t0
    elif snr_type == 'lognorm':
        u = torch.randn((), generator=gen)
        return torch.sigmoid(u) * (t1 - t0) + t0
    else:
        raise ValueError(f"Unknown snr type: {snr_type}")


# --------------------------------------------------------------------------- #
# Raw-latent dataset: yields noisy latents (length-agnostic)                  #
# --------------------------------------------------------------------------- #
class QuadtreeRawLatentDataset(Dataset):
    """Yields a single noisy latent x_t (C, crop, crop), its timestep t, and label.

    This is the length-agnostic input side: it does NOT run the variance model or
    the quadtree compressor. It only de-patchifies a features file into a spatial
    latent, center-crops/pads to `crop`, and applies the flow-matching forward
    process x_t = (1 - t) * x0 + t * eps (t=0 clean, t=1 pure noise), matching the
    convention the VariancePredictor was trained on.

    Sharding is handled by the sampler passed to the DataLoader (see
    `build_shard_indices`), not here.
    """

    def __init__(self, root_dir, crop=32, snr_type='lognorm',
                 train_eps=None, seed=0):
        super().__init__()
        self.root_dir = root_dir
        self.crop = crop
        self.snr_type = snr_type
        self.t0 = 0.0 if train_eps is None else float(train_eps)
        self.t1 = 1.0
        # Discover the flat list of feature files. We reuse the same directory
        # layout as IN1kLatentDataset but only need the greater_than_*_crop set
        # (square, croppable). Fall back to any .safetensors under root.
        self.files = self._discover_files(root_dir)
        if not self.files:
            raise RuntimeError(f"no .safetensors feature files found under {root_dir}")
        self._base_seed = seed

    @staticmethod
    def _discover_files(root_dir):
        import glob
        # Prefer the crop subfolder (square latents), else sweep recursively.
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
        import torch.nn.functional as F
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
        x0 = self._center_crop(load_latent(self.files[idx])).float()   # (C, crop, crop)

        # Fresh randomness per item; independent of workers/ranks (the sampler
        # already guarantees disjoint indices, so per-item RNG only needs variety).
        gen = torch.Generator().manual_seed(
            (self._base_seed * 1_000_003 + idx) & 0x7FFF_FFFF)
        t = _sample_t(self.snr_type, self.t0, self.t1, gen=gen)        # 0-dim
        eps = torch.randn(x0.shape, generator=gen)
        x_t = (1.0 - t) * x0 + t * eps

        # label: stored per-file (see LatentVarianceDataset / safetensors "label").
        label = self._load_label(idx)
        # x0 (the clean latent) rides along: the packer compresses it on the SAME
        # quadtree structure decided from x_t to produce the clean training target.
        return dict(x_t=x_t, x0=x0, t=t.float(), label=label)

    def _load_label(self, idx):
        from safetensors import safe_open
        try:
            with safe_open(self.files[idx], 'pt') as f:
                if 'label' in f.keys():
                    return f.get_tensor('label').reshape(()).long()
        except Exception:
            pass
        return torch.tensor(-1, dtype=torch.long)


def _raw_collate(samples):
    """Stack a plain batch of raw latents for a single GPU forward pass."""
    x_t = torch.stack([s['x_t'] for s in samples])          # (B, C, H, W)
    x0 = torch.stack([s['x0'] for s in samples])            # (B, C, H, W) clean
    t = torch.stack([s['t'] for s in samples])              # (B,)
    label = torch.stack([s['label'] for s in samples])      # (B,)
    return dict(x_t=x_t, x0=x0, t=t, label=label)


# --------------------------------------------------------------------------- #
# Per-GPU disjoint sharding                                                    #
# --------------------------------------------------------------------------- #
def build_shard_indices(num_files, rank, world_size, epoch, seed=42):
    """Disjoint, per-rank index shard of a globally shuffled index list.

    Globally shuffle [0, num_files) with a shared per-epoch seed, then take every
    `world_size`-th index starting at `rank`. Each rank gets a disjoint subset, no
    sample is shared across GPUs within an epoch, and classes interleave evenly.
    (Sharding-by-contiguous-block is equivalent for disjointness; strided just
    mixes labels more.)
    """
    g = torch.Generator().manual_seed(seed + epoch)
    perm = torch.randperm(num_files, generator=g)
    return perm[rank::world_size].tolist()


# --------------------------------------------------------------------------- #
# GPU-side plan pool + budget-aware image batching                            #
# --------------------------------------------------------------------------- #
class QuadtreePlanPool:
    """Pool *quadtree plans* and hand out image batches that fit a token budget.

    This is the step that makes learned compression trainable. The variance
    predictor is frozen, so it can run arbitrarily far ahead under `no_grad` on
    whatever batch size is convenient. Its output fixes each image's quadtree
    structure and therefore its exact token count — *before* any trained layer
    touches the latent.

    So we pool plans, not tokens. Each `__next__` selects the longest prefix of
    pooled images whose token counts sum to <= `max_seq_len` and returns those
    images together with their plans. The caller (the training loop) then runs the
    trained compressor on exactly that image batch, so every image whose encoder
    forward ran also contributes to this step's loss — no gradient is stranded
    across a `zero_grad`.

    The image batch size is whatever the arithmetic yields (41, 37, ...) and
    changes every step. Nothing downstream depends on it being constant.

    Yields dict:
        x_t    (B, C, H, W)  noisy latents, full resolution (NOT pooled)
        x0     (B, C, H, W)  clean latents, for the full-resolution loss
        t      (B,)          timesteps
        label  (B,)          class labels
        plans  list of B dicts: levels (n,), positions (n,2), sizes (n,)
        n_tok  int           total valid tokens across the batch
    """

    def __init__(self, raw_loader, model, *, max_seq_len=1024, device='cuda',
                 refill_target=None, crop=32, threshold=0.0):
        self.raw_loader = raw_loader
        self.model = model.to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.device = device
        self.max_seq_len = max_seq_len
        self.crop = crop
        self.threshold = threshold
        # Overshoot the budget so one refill usually covers a full sequence.
        self.refill_target = refill_target or int(math.ceil(max_seq_len * 1.5))

        self._raw_iter = iter(raw_loader)
        self._pool = []
        self._pool_tokens = 0
        self._exhausted = False

    def __iter__(self):
        return self

    @torch.no_grad()
    def _refill(self):
        """Pull raw batches, run the FROZEN predictor, pool the resulting plans."""
        while (not self._exhausted) and self._pool_tokens < self.refill_target:
            try:
                batch = next(self._raw_iter)
            except StopIteration:
                self._exhausted = True
                break

            x_t = batch['x_t'].to(self.device, non_blocking=True)
            x0 = batch['x0'].to(self.device, non_blocking=True)
            t = batch['t'].to(self.device, non_blocking=True)
            labels = batch['label']

            # One batched forward of the frozen predictor; the quadtree walk is
            # per-image and variable-length, so it loops — but it only reads the
            # variance grids, never the latent, and allocates no values.
            var = predict_variance(self.model, x_t, t)
            for i in range(x_t.shape[0]):
                var_i = [v[i] for v in var]
                levels, positions, sizes = plan_from_variance(
                    var_i, self.threshold, self.crop)
                n_i = int(sizes.shape[0])
                # An image whose token count alone exceeds the budget can never be
                # packed whole; skip rather than deadlock.
                if n_i > self.max_seq_len:
                    continue
                self._pool.append(dict(
                    x_t=x_t[i], x0=x0[i], t=t[i], label=labels[i],
                    plan=dict(levels=levels, positions=positions, sizes=sizes),
                    n_tok=n_i))
                self._pool_tokens += n_i

    def __next__(self):
        if self._pool_tokens < self.max_seq_len:
            self._refill()
        if not self._pool:
            raise StopIteration

        # Greedily take whole images until the next would overflow the budget.
        picked, n_tok = [], 0
        while self._pool:
            n_i = self._pool[0]['n_tok']
            if picked and n_tok + n_i > self.max_seq_len:
                break
            item = self._pool.pop(0)
            self._pool_tokens -= n_i
            picked.append(item)
            n_tok += n_i
            if n_tok >= self.max_seq_len:
                break

        return dict(
            x_t=torch.stack([p['x_t'] for p in picked]),
            x0=torch.stack([p['x0'] for p in picked]),
            t=torch.stack([p['t'] for p in picked]),
            label=torch.stack([p['label'] for p in picked]).to(self.device),
            plans=[p['plan'] for p in picked],
            n_tok=n_tok,
        )


# --------------------------------------------------------------------------- #
# Loader assembly                                                             #
# --------------------------------------------------------------------------- #
class INQuadtreeLatentLoader:
    """Builds the raw DataLoader + GPU-side packed iterator for quadtree training.

    Usage (per rank):
        loader = INQuadtreeLatentLoader(train_config)
        for epoch in range(...):
            packed = loader.train_iter(model, epoch, rank, world_size)
            for batch in packed:      # batch is a packed sequence dict on `device`
                ...
    """

    def __init__(self, train):
        self.cfg = train
        self.root_dir = train.data_path
        self.crop = getattr(train, 'crop', 32)
        self.num_workers = getattr(train.loader, 'num_workers', 4)
        self.raw_batch_size = getattr(train.loader, 'raw_batch_size', 64)
        self.max_seq_len = getattr(train, 'max_tokens', 1024)
        self.pad_to_multiple = getattr(train, 'pad_to_multiple', 128)
        self.threshold = getattr(train, 'quadtree_threshold', 0.0)
        self.snr_type = getattr(train, 'snr_type', 'lognorm')
        self.train_eps = getattr(train, 'train_eps', None)
        self.seed = getattr(train, 'seed', 42)
        self.latent_channels = getattr(train, 'latent_channels', 4)

        self.dataset = QuadtreeRawLatentDataset(
            self.root_dir, crop=self.crop, snr_type=self.snr_type,
            train_eps=self.train_eps, seed=self.seed)

    def train_len(self):
        return len(self.dataset)

    def _raw_loader(self, epoch, rank, world_size):
        indices = build_shard_indices(
            len(self.dataset), rank, world_size, epoch, seed=self.seed)
        return DataLoader(
            self.dataset,
            batch_size=self.raw_batch_size,
            sampler=indices,                  # disjoint, pre-shuffled shard
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=_raw_collate,
            prefetch_factor=2 if self.num_workers > 0 else None,
        )

    def train_iter(self, model, epoch, rank=0, world_size=1, device='cuda'):
        """Plan pool for one epoch's shard.

        `model` is the FROZEN VariancePredictor. The returned iterator yields
        image batches sized to the token budget; the caller runs the trained
        compressor on each batch (see PredictiveVarianceCompressor).
        """
        raw_loader = self._raw_loader(epoch, rank, world_size)
        return QuadtreePlanPool(
            raw_loader, model,
            max_seq_len=self.max_seq_len,
            device=device,
            crop=self.crop,
            threshold=self.threshold,
        )


# --------------------------------------------------------------------------- #
# Smoke test                                                                   #
# --------------------------------------------------------------------------- #
def _smoke_test():
    """End-to-end shape/packing check on synthetic in-memory data (no files)."""
    torch.manual_seed(0)
    C, H = 4, 32
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    from fit.quadtree_compression.predictive_compressor import (
        PredictiveVarianceCompressor)

    vp = VariancePredictor(latent_channels=C, region_sizes=(2, 4, 8)).to(device).eval()
    comp = PredictiveVarianceCompressor(latent_channels=C, c=64, d=128, crop=H,
                                        pad_to_multiple=128).to(device)

    class _FakeRaw(torch.utils.data.Dataset):
        def __len__(self):
            return 40

        def __getitem__(self, i):
            g = torch.Generator().manual_seed(i)
            return dict(x_t=torch.randn(C, H, H, generator=g),
                        x0=torch.randn(C, H, H, generator=g),
                        t=torch.rand((), generator=g).float(),
                        label=torch.tensor(i % 10, dtype=torch.long))

    raw_loader = DataLoader(_FakeRaw(), batch_size=8, collate_fn=_raw_collate)
    pool = QuadtreePlanPool(raw_loader, vp, max_seq_len=512, device=device,
                            crop=H, threshold=0.5)

    seen = 0
    for step, sel in enumerate(pool):
        B = sel['x_t'].shape[0]
        assert sel['n_tok'] <= 512
        packed = comp(sel['x_t'], sel['plans'], sel['label'], sel['t'], x0=sel['x0'])
        N = packed['feature'].shape[1]
        assert packed['feature'].shape == (1, N, comp.d)
        assert packed['grid'].shape == (1, 2, N)
        assert packed['tsize'].shape == (1, N)
        assert packed['size'].shape == (1, B, 2)
        assert N % 128 == 0
        valid = int(packed['mask'].sum())
        assert valid == sel['n_tok'] <= 512
        assert int(packed['doc_ids'].max()) == B - 1
        assert int((packed['doc_ids'] >= 0).sum()) == valid
        assert packed['feature'].requires_grad, "encoder gradient must flow"
        seen += B
        print(f"step {step:2d}  N={N:4d}  valid={valid:4d}  n_pack={B:2d}")

    print(f"packed {seen} images total; smoke test passed")


if __name__ == '__main__':
    _smoke_test()
