"""
Variable-length quadtree-compressed latent dataset for varlen (packed) training.

This mirrors the varlen packing of `in1k_latent_dataset.py`, but the number of
tokens an image contributes is *not known ahead of time*: it depends on the
variance-guided quadtree compression (`quadtree_compression.compress`), which in
turn needs a forward pass of the `VariancePredictor`. That breaks the
pre-computed-length `TokenBudgetBatchSampler` used by the fixed-grid dataset — we
can't pack by length before we know the lengths.

Instead we pack lazily with a *sample pool*:

    1.  A plain (sharded, shuffled) DataLoader yields batches of RAW noisy
        latents (x_t, t, label) — cheap, length-agnostic.
    2.  On the GPU, `QuadtreePackedIterator` maintains a pool of already-
        compressed images (each a variable-length token/pos/size triple). Every
        step it pops whole images into the current sequence until the next image
        would exceed `max_seq_len`, then emits the packed sequence (padded to a
        multiple) with per-image `doc_ids`. Leftover images stay in the pool.
    3.  Whenever the pool can't guarantee it can fill one more `max_seq_len`
        sequence, it pulls the next raw batch, runs the variance predictor +
        `compress` on it (batched on the GPU), and refills the pool. The raw
        batch is sized so a single refill very likely overshoots `max_seq_len`.

Per-GPU disjointness: each rank consumes a strided shard of a globally shuffled
index list (every `world_size`-th index), so no sample is ever seen by two ranks
within an epoch.

Whole images are never split across sequences (doc_id contiguity + the quadtree
token grouping are preserved); the sequence tail is zero-padded.

The compressed token layout (per image) comes straight from
`quadtree_compression.compress`:
    tokens    (n_i, 4 * latent_channels)   patchified quadtree leaf values
    positions (n_i, 2)                      (y, x) center in latent-pixel coords
    sizes     (n_i,)                        leaf side (1 / 2 / 4 / 8)
"""

import math
import os

import torch
from torch.utils.data import DataLoader, Dataset

# `compress` lives in the sibling package `quadtree_compression`. Import defensively
# so this module can be imported for its dataset/loader pieces even if the compressor
# package path isn't wired up yet at import time.
try:
    from fit.quadtree_compression.quadtree_compression import compress_from_variance
    from fit.quadtree_compression.variance_prediction import (
        VariancePredictor, load_latent, predict_variance)  # load_latent: safetensors -> (4, 2H, 2W)
except Exception:  # pragma: no cover - allows partial import in dev
    compress_from_variance = None
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
        return dict(x_t=x_t, t=t.float(), label=label)

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
    t = torch.stack([s['t'] for s in samples])              # (B,)
    label = torch.stack([s['label'] for s in samples])      # (B,)
    return dict(x_t=x_t, t=t, label=label)


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
# GPU-side pool + packing                                                      #
# --------------------------------------------------------------------------- #
class QuadtreePackedIterator:
    """Lazily pack quadtree-compressed images into fixed-budget varlen sequences.

    Wraps a plain raw-latent DataLoader. Holds the VariancePredictor on `device`.
    Maintains a pool of compressed images; each `__next__` emits one packed
    sequence dict (see below). Refills the pool by pulling raw batches and running
    variance-predict + `compress` on the GPU whenever the pool might not cover the
    next `max_seq_len` sequence.

    Emitted sequence dict (B=1, packed) — mirrors `packed_collate_fn`:
        feature  (1, N_total, 4C)   packed quadtree tokens
        grid     (1, 2, N_total)    per-token (y, x) center positions
        tsize    (1, N_total)       per-token leaf side (1/2/4/8) — compression level
        mask     (1, N_total)       1 for valid tokens, 0 for padding
        doc_ids  (1, N_total)       image index within sequence, -1 for padding
        label    (1, n_pack)        class label per packed image
        t        (1, n_pack)        timestep per packed image
        n_pack   (1,)               number of images packed
    """

    def __init__(self, raw_loader, model, *, max_seq_len=1024,
                 pad_to_multiple=128, device='cuda',
                 refill_target=None, latent_channels=4):
        self.raw_loader = raw_loader
        self.model = model.to(device).eval()
        self.device = device
        self.max_seq_len = max_seq_len
        self.pad_to_multiple = pad_to_multiple
        self.latent_channels = latent_channels
        # Pool must be able to cover one full sequence before we emit. Refill until
        # the pooled token count is at least this many tokens (or the source drips
        # dry). Default: overshoot max_seq_len by 50% so a single refill usually
        # suffices and we rarely block mid-pack.
        self.refill_target = refill_target or int(math.ceil(max_seq_len * 1.5))

        self._raw_iter = iter(raw_loader)
        self._pool = []            # list of dict(tokens, positions, sizes, label, t)
        self._pool_tokens = 0      # total valid tokens currently pooled
        self._exhausted = False

    def __iter__(self):
        return self

    @torch.no_grad()
    def _refill(self):
        """Pull raw batches, compress them, extend the pool until target/exhausted."""
        while (not self._exhausted) and self._pool_tokens < self.refill_target:
            try:
                batch = next(self._raw_iter)
            except StopIteration:
                self._exhausted = True
                break

            x_t = batch['x_t'].to(self.device, non_blocking=True)   # (B, C, H, W)
            t = batch['t'].to(self.device, non_blocking=True)       # (B,)
            labels = batch['label']                                 # (B,) cpu

            # Run the variance predictor ONCE on the whole batch (the model forward
            # is the expensive part and is trivially batched). The quadtree walk is
            # inherently per-image / variable length, so we loop over the batch and
            # feed each image its own slice of the variance grids to
            # `compress_from_variance` — no per-image model calls.
            threshold = self._threshold()
            var = predict_variance(self.model, x_t, t)   # list of (B, H/N, W/N)
            for i in range(x_t.shape[0]):
                var_i = [v[i] for v in var]              # per-image variance grids
                tokens, positions, sizes = compress_from_variance(
                    x_t[i], var_i, threshold=threshold)
                n_i = int(tokens.shape[0])
                # Guard: an image whose *minimum* token count already exceeds the
                # sequence budget can never be packed whole. Skip with a warning
                # rather than deadlock. (n_i is smallest at the coarsest split.)
                if n_i > self.max_seq_len:
                    continue
                self._pool.append(dict(
                    tokens=tokens, positions=positions, sizes=sizes,
                    label=labels[i], t=t[i].detach().float().cpu()))
                self._pool_tokens += n_i

    def _threshold(self):
        # Compression threshold. Kept as a method so subclasses / configs can make
        # it schedule-dependent (e.g. anneal over training). Constant by default.
        return getattr(self, 'threshold', 0.0)

    def __next__(self):
        # Ensure the pool can (likely) cover a full sequence.
        if self._pool_tokens < self.max_seq_len:
            self._refill()
        if not self._pool:
            raise StopIteration

        # Greedily pack whole images until the next would overflow max_seq_len.
        packed, raw_len = [], 0
        while self._pool:
            n_i = int(self._pool[0]['tokens'].shape[0])
            if packed and raw_len + n_i > self.max_seq_len:
                break
            item = self._pool.pop(0)
            self._pool_tokens -= n_i
            packed.append(item)
            raw_len += n_i
            # A single image may itself reach the budget; emit it alone.
            if raw_len >= self.max_seq_len:
                break

        return self._collate(packed)

    def _collate(self, packed):
        """Concatenate packed images into one padded sequence with doc_ids."""
        C4 = 4 * self.latent_channels
        n_pack = len(packed)
        raw_len = sum(int(p['tokens'].shape[0]) for p in packed)
        N_total = int(math.ceil(raw_len / self.pad_to_multiple) * self.pad_to_multiple)
        N_total = max(N_total, self.pad_to_multiple)

        dev = self.device
        feat = torch.zeros(1, N_total, C4, device=dev)
        grid = torch.zeros(1, 2, N_total, device=dev)
        tsize = torch.zeros(1, N_total, dtype=torch.long, device=dev)
        mask = torch.zeros(1, N_total, dtype=torch.uint8, device=dev)
        doc = torch.full((1, N_total), -1, dtype=torch.int32, device=dev)
        label = torch.full((1, n_pack), -1, dtype=torch.int64, device=dev)
        tvec = torch.zeros(1, n_pack, dtype=torch.float32, device=dev)

        offset = 0
        for img_idx, p in enumerate(packed):
            n_i = int(p['tokens'].shape[0])
            sl = slice(offset, offset + n_i)
            feat[0, sl] = p['tokens'].to(dev)
            grid[0, :, sl] = p['positions'].to(dev).transpose(0, 1)   # (2, n_i)
            tsize[0, sl] = p['sizes'].to(dev)
            mask[0, sl] = 1
            doc[0, sl] = img_idx
            label[0, img_idx] = p['label'].to(dev)
            tvec[0, img_idx] = p['t'].to(dev)
            offset += n_i

        return dict(
            feature=feat, grid=grid, tsize=tsize, mask=mask, doc_ids=doc,
            label=label, t=tvec,
            n_pack=torch.tensor([n_pack], dtype=torch.int32, device=dev),
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
        raw_loader = self._raw_loader(epoch, rank, world_size)
        it = QuadtreePackedIterator(
            raw_loader, model,
            max_seq_len=self.max_seq_len,
            pad_to_multiple=self.pad_to_multiple,
            device=device,
            latent_channels=self.latent_channels,
        )
        it.threshold = self.threshold
        return it


# --------------------------------------------------------------------------- #
# Smoke test                                                                   #
# --------------------------------------------------------------------------- #
def _smoke_test():
    """End-to-end shape/packing check on synthetic in-memory data (no files)."""
    torch.manual_seed(0)
    C, H = 4, 32
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = VariancePredictor(latent_channels=C, region_sizes=(2, 4, 8)).to(device).eval()

    # Fake raw dataset: a list of (x_t, t, label) dicts, bypassing safetensors.
    class _FakeRaw(torch.utils.data.Dataset):
        def __len__(self):
            return 40

        def __getitem__(self, i):
            g = torch.Generator().manual_seed(i)
            return dict(x_t=torch.randn(C, H, H, generator=g),
                        t=torch.rand((), generator=g).float(),
                        label=torch.tensor(i % 10, dtype=torch.long))

    raw_loader = DataLoader(_FakeRaw(), batch_size=8, collate_fn=_raw_collate)
    it = QuadtreePackedIterator(raw_loader, model, max_seq_len=512,
                                pad_to_multiple=128, device=device,
                                latent_channels=C)
    it.threshold = 0.5

    seen_docs = 0
    for step, batch in enumerate(it):
        N = batch['feature'].shape[1]
        n_pack = int(batch['n_pack'])
        valid = int(batch['mask'].sum())
        assert batch['feature'].shape == (1, N, 4 * C)
        assert batch['grid'].shape == (1, 2, N)
        assert batch['tsize'].shape == (1, N)
        assert N % 128 == 0
        assert valid <= 512
        # doc_ids consistent with n_pack and mask
        max_doc = int(batch['doc_ids'].max())
        assert max_doc == n_pack - 1, (max_doc, n_pack)
        assert int((batch['doc_ids'] >= 0).sum()) == valid
        seen_docs += n_pack
        print(f"step {step:2d}  N={N:4d}  valid={valid:4d}  n_pack={n_pack:2d}")

    print(f"packed {seen_docs} images total; smoke test passed")


if __name__ == '__main__':
    _smoke_test()
