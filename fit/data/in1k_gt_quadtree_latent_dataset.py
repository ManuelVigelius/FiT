"""
Ground-truth-variance quadtree dataset for varlen (packed) training.

Sibling of `in1k_quadtree_latent_dataset.py`. Same output contract — the caller
gets image batches plus per-image quadtree *plans* and runs the trained
`PredictiveVarianceCompressor` on exactly those images — but the tree is decided
from the TRUE variance of the clean latent x0 instead of a `VariancePredictor`
forward on x_t.

Why
---
The predictor is a small model, so at inference it is a real error source: a
mis-predicted variance puts the tree in the wrong place and there is no way to
tell that apart from a compressor/transformer failure. This dataset removes that
variable. It is an ORACLE — it reads x0, which is not available at sampling time
— so it is a diagnostic/ablation baseline, not a deployable configuration. Train
both and the gap between them is the predictor's contribution to the error.

What replaces the predictor
---------------------------
Two ingredients decide the tree here:

1.  **Ground-truth variance.** `region_variance_targets(x0, (2, 4, 8))` gives the
    exact per-channel variance of the clean latent over each region size; we take
    the channel-wise MAX to land on the same (H/N, W/N) grids and the same
    readout convention `predict_variance` produces. A region is compressible iff
    that variance is below `quadtree_threshold`. This is precisely the quantity
    the predictor is trained to approximate, so the two paths differ in nothing
    but prediction error.

2.  **A time-dependent max resolution.** The GT variance is independent of t,
    but the amount of detail worth spending tokens on is not: at high t the
    latent is nearly pure noise and no fine structure is recoverable, so the tree
    should stay coarse regardless of variance. `max_res_schedule` is a piecewise
    list of `(t_start, min_leaf_size)` pairs — for a sample at timestep t, the
    entry with the largest `t_start <= t` gives the FINEST leaf side the tree may
    use. The quadtree simply stops splitting there:

        max_res_schedule: [[0.0, 1], [0.7, 2], [0.85, 4], [0.95, 8]]

    means t in [0, 0.7) may split all the way to size 1 (lossless), t in
    [0.95, 1] is forced to a single size-8 level — full compression, 4 tokens for
    a 32x32 latent — no matter how the variance falls. Note the direction: bigger
    leaf side == coarser == fewer tokens, so the schedule's second entry is a
    *floor* on leaf size, i.e. a cap on resolution. Setting the floor to 8 pins
    every patch at the coarsest level; setting it to 1 imposes no cap at all.

    The schedule is also what makes the token count controllable. Without it the
    GT variance alone would happily emit 256 lossless tokens for a busy image at
    every timestep.

CPU, not GPU
------------
This is the other reason the file exists. The predictor path had to move latents
to the GPU before it knew any token counts, which forced the whole
`QuadtreePlanPool` dance: pool plans on-device, then hand the trained compressor
exactly the images that fit the budget so no encoder gradient is stranded across
a `zero_grad`. None of that is needed here — the plan comes from three avg-pools
on x0, which is cheap and runs in the DataLoader WORKER. So each worker returns a
sample that already knows its own token count, and packing is plain CPU
arithmetic (`GTQuadtreePacker`) over the loader's output.

The gradient-alignment property the pool existed to protect still holds, and more
simply: the plan never depends on a trained module at all, so the compressor only
ever runs on the images the packer already selected for this step.

Plan layout (per image), identical to the predictor path:
    levels    (n_i,)     encoder level l, leaf side N == 2**l
    positions (n_i, 2)   (y, x) patch center in latent-pixel coords
    sizes     (n_i,)     leaf side (1 / 2 / 4 / 8)
"""

import glob
import math
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from fit.data.in1k_quadtree_latent_dataset import (_sample_t,
                                                   build_shard_indices)
from fit.quadtree_compression.quadtree_compression import (LEAF_SIZES,
                                                           REGION_SIZES)
from fit.quadtree_compression.variance_prediction import (
    load_latent, region_variance_targets)

# Default: no cap anywhere. Overridden from config in practice — an uncapped GT
# tree spends far more tokens than the predictor path does.
DEFAULT_MAX_RES_SCHEDULE = ((0.0, 1),)


# --------------------------------------------------------------------------- #
# Time-dependent resolution cap                                               #
# --------------------------------------------------------------------------- #
def normalize_max_res_schedule(schedule):
    """Validate a `[[t_start, min_leaf_size], ...]` schedule into a sorted tuple.

    Entries are `(t_start, min_leaf_size)`: from `t_start` onward the quadtree
    may not split finer than `min_leaf_size` latent pixels. Sizes must come from
    LEAF_SIZES (1 / 2 / 4 / 8) and the list must cover t = 0, so every timestep
    resolves to exactly one entry.
    """
    if schedule is None:
        schedule = DEFAULT_MAX_RES_SCHEDULE
    out = []
    for entry in schedule:
        t_start, size = float(entry[0]), int(entry[1])
        if size not in LEAF_SIZES:
            raise ValueError(
                f"max_res_schedule leaf size {size} not in {LEAF_SIZES}")
        out.append((t_start, size))
    out.sort(key=lambda e: e[0])
    if not out or out[0][0] > 0.0:
        raise ValueError(
            "max_res_schedule must contain an entry with t_start <= 0.0 so that "
            f"every timestep is covered; got {out}")
    return tuple(out)


def min_leaf_size_at(t, schedule):
    """Finest leaf side the tree may use at timestep `t` (last entry with t_start <= t)."""
    t = float(t)
    size = schedule[0][1]
    for t_start, s in schedule:
        if t >= t_start:
            size = s
        else:
            break
    return size


# --------------------------------------------------------------------------- #
# Ground-truth variance -> quadtree plan                                      #
# --------------------------------------------------------------------------- #
def gt_variance_grids(x0, region_sizes=REGION_SIZES):
    """Channel-max ground-truth variance of x0 per region size.

    `region_variance_targets` keeps all channels — that is what the predictor's
    per-channel likelihood needs — while the quadtree only ever reads the
    channel-wise max, which is also what `predict_variance` returns at readout.
    Reducing here keeps the two paths comparing the same number.

    x0: (C, H, W) clean latent -> list of (H/N, W/N), one per region size.
    """
    per_channel = region_variance_targets(x0, region_sizes)
    return [v.amax(dim=0) for v in per_channel]


def plan_from_gt_variance(var, threshold, latent_size, min_leaf_size=1):
    """Decide the quadtree structure from GT variance, capped at `min_leaf_size`.

    Structurally identical to `quadtree_compression.plan_from_variance` — same
    recursion, same emission order, same return layout — with one extra stopping
    rule: a patch never splits below `min_leaf_size`, whatever the variance says.
    That is the time-dependent resolution cap. With `min_leaf_size == 1` this
    reduces exactly to `plan_from_variance`.

    Kept as its own function rather than a flag on `plan_from_variance` because
    that one is on the inference path, where the cap has no meaning.

    var          : list of per-scale channel-max variance grids, var[s] of shape
                   (H/N_s, W/N_s) for N_s in REGION_SIZES (2, 4, 8).
    threshold    : regions with variance < threshold are compressible.
    latent_size  : H (== W) of the latent in latent pixels.
    min_leaf_size: the FINEST leaf side (1/2/4/8) the tree may split down to.
                   Bigger means coarser, i.e. more compression and fewer tokens.

    Returns (levels, positions, sizes); see the module docstring.
    """
    H = int(latent_size)
    device = var[0].device
    flat_by_size = {n: (var[s] < threshold) for s, n in enumerate(REGION_SIZES)}

    levels, positions, sizes = [], [], []

    def patch_is_flat(n, ly, lx):
        f = flat_by_size[n]
        return bool(f[ly:ly + 2, lx:lx + 2].all())

    def emit(n, ly, lx):
        levels.append(int(n).bit_length() - 1)       # N=1,2,4,8 -> l=0,1,2,3
        positions.append((float(ly * n + n), float(lx * n + n)))
        sizes.append(n)

    def recurse(n, ly, lx):
        # Stop at the schedule's floor even when the patch is not flat: at this
        # timestep there is no detail worth the extra tokens.
        if n > min_leaf_size and n > 1 and not patch_is_flat(n, ly, lx):
            m = n // 2
            for dy in (0, 1):
                for dx in (0, 1):
                    recurse(m, 2 * (ly + dy), 2 * (lx + dx))
            return
        emit(n, ly, lx)

    coarse_n = LEAF_SIZES[-1]
    coarse_leaves = H // coarse_n
    for ly in range(0, coarse_leaves, 2):
        for lx in range(0, coarse_leaves, 2):
            recurse(coarse_n, ly, lx)

    return (
        torch.tensor(levels, dtype=torch.long, device=device),
        torch.tensor(positions, dtype=torch.float32, device=device),
        torch.tensor(sizes, dtype=torch.long, device=device),
    )


# --------------------------------------------------------------------------- #
# Dataset: one item == one planned image                                      #
# --------------------------------------------------------------------------- #
class GTQuadtreeLatentDataset(Dataset):
    """Yields a noisy latent, its clean latent, and its ready-made quadtree plan.

    Unlike `QuadtreeRawLatentDataset` this is NOT length-agnostic: because the
    plan needs no model, the worker computes it here and the item already knows
    its own token count. Everything downstream is plain packing arithmetic.

    Sharding is handled by the sampler passed to the DataLoader (see
    `build_shard_indices`), not here.
    """

    def __init__(self, root_dir, crop=32, snr_type='lognorm', train_eps=None,
                 seed=0, threshold=0.0, max_res_schedule=None):
        super().__init__()
        self.root_dir = root_dir
        self.crop = crop
        self.snr_type = snr_type
        self.t0 = 0.0 if train_eps is None else float(train_eps)
        self.t1 = 1.0
        self.threshold = float(threshold)
        self.max_res_schedule = normalize_max_res_schedule(max_res_schedule)
        self.files = self._discover_files(root_dir)
        if not self.files:
            raise RuntimeError(f"no .safetensors feature files found under {root_dir}")
        self._base_seed = seed

    @staticmethod
    def _discover_files(root_dir):
        # Same layout as the predictor path: prefer the square crop folders, else
        # sweep recursively.
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
        x0 = self._center_crop(load_latent(self.files[idx])).float()   # (C, crop, crop)

        gen = torch.Generator().manual_seed(
            (self._base_seed * 1_000_003 + idx) & 0x7FFF_FFFF)
        t = _sample_t(self.snr_type, self.t0, self.t1, gen=gen)        # 0-dim
        eps = torch.randn(x0.shape, generator=gen)
        x_t = (1.0 - t) * x0 + t * eps

        # The tree is decided from the CLEAN latent (this is the oracle part) and
        # capped by the timestep's max resolution.
        var = gt_variance_grids(x0)
        levels, positions, sizes = plan_from_gt_variance(
            var, self.threshold, self.crop,
            min_leaf_size=min_leaf_size_at(t, self.max_res_schedule))

        return dict(
            x_t=x_t, x0=x0, t=t.float(), label=self._load_label(idx),
            plan=dict(levels=levels, positions=positions, sizes=sizes),
            n_tok=int(sizes.shape[0]),
        )

    def _load_label(self, idx):
        from safetensors import safe_open
        try:
            with safe_open(self.files[idx], 'pt') as f:
                if 'label' in f.keys():
                    return f.get_tensor('label').reshape(()).long()
        except Exception:
            pass
        return torch.tensor(-1, dtype=torch.long)


def _passthrough_collate(samples):
    """Hand the packer a plain list — grouping is its job, not the collate's.

    The DataLoader batch here is just a transport unit for worker parallelism;
    which images share a *sequence* is decided by the token budget afterwards,
    and a sample can carry over into the next selection. Stacking at this point
    would only have to be undone.
    """
    return samples


# --------------------------------------------------------------------------- #
# CPU packing to the token budget                                             #
# --------------------------------------------------------------------------- #
class GTQuadtreePacker:
    """Greedily group planned samples into token-budget-sized image batches.

    Every item already carries `n_tok`, so this is pure arithmetic: keep a small
    buffer, and each `__next__` take whole images until the next would overflow
    `max_seq_len`. Nothing is dropped — a sample that does not fit stays at the
    head of the buffer for the following step — except an image whose own token
    count exceeds the whole budget, which can never be packed and would otherwise
    deadlock.

    Whole images are never split across sequences (doc_id contiguity and the
    quadtree token grouping are preserved); the compressor zero-pads the tail.

    Yields dict:
        x_t    (B, C, H, W)  noisy latents, full resolution (NOT pooled)
        x0     (B, C, H, W)  clean latents, for the full-resolution loss
        t      (B,)          timesteps
        label  (B,)          class labels
        plans  list of B dicts: levels (n,), positions (n,2), sizes (n,)
        n_tok  int           total valid tokens across the batch

    Tensors land on `device`; `plans` stay on the CPU, which is where
    `plan_to_masks` wants them anyway (it builds masks with python-level indexing).
    """

    def __init__(self, loader, *, max_seq_len=1024, device='cuda',
                 refill_target=None):
        self.loader = loader
        self.max_seq_len = max_seq_len
        self.device = device
        # Overshoot the budget so one refill usually covers a full sequence.
        self.refill_target = refill_target or int(math.ceil(max_seq_len * 1.5))

        self._iter = iter(loader)
        self._buf = []
        self._buf_tokens = 0
        self._exhausted = False

    def __iter__(self):
        return self

    def _refill(self):
        while (not self._exhausted) and self._buf_tokens < self.refill_target:
            try:
                samples = next(self._iter)
            except StopIteration:
                self._exhausted = True
                break
            for s in samples:
                if s['n_tok'] > self.max_seq_len:
                    # Cannot ever be packed whole; skip rather than stall. Tune
                    # max_res_schedule / threshold if this fires often.
                    continue
                self._buf.append(s)
                self._buf_tokens += s['n_tok']

    def __next__(self):
        if self._buf_tokens < self.max_seq_len:
            self._refill()
        if not self._buf:
            raise StopIteration

        picked, n_tok = [], 0
        while self._buf:
            n_i = self._buf[0]['n_tok']
            if picked and n_tok + n_i > self.max_seq_len:
                break
            item = self._buf.pop(0)
            self._buf_tokens -= n_i
            picked.append(item)
            n_tok += n_i
            if n_tok >= self.max_seq_len:
                break

        dev = self.device
        return dict(
            x_t=torch.stack([p['x_t'] for p in picked]).to(dev, non_blocking=True),
            x0=torch.stack([p['x0'] for p in picked]).to(dev, non_blocking=True),
            t=torch.stack([p['t'] for p in picked]).to(dev, non_blocking=True),
            label=torch.stack([p['label'] for p in picked]).to(dev, non_blocking=True),
            plans=[p['plan'] for p in picked],
            n_tok=n_tok,
        )


# --------------------------------------------------------------------------- #
# Loader assembly                                                             #
# --------------------------------------------------------------------------- #
class INGTQuadtreeLatentLoader:
    """Builds the DataLoader + CPU packer for ground-truth-variance quadtree training.

    Drop-in for `INQuadtreeLatentLoader` on the trainer side, with one difference
    in the call: `train_iter` takes NO model. The trainer's `vp` argument is
    accepted and ignored so the same training script works against either loader.

        loader = INGTQuadtreeLatentLoader(train_config)
        for epoch in range(...):
            packed = loader.train_iter(None, epoch, rank, world_size)
            for selection in packed:
                packed = compressor(selection['x_t'], selection['plans'], ...)
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
        schedule = getattr(train, 'max_res_schedule', None)
        # OmegaConf ListConfig -> plain tuples
        if schedule is not None:
            schedule = [(float(e[0]), int(e[1])) for e in schedule]
        self.max_res_schedule = normalize_max_res_schedule(schedule)

        self.dataset = GTQuadtreeLatentDataset(
            self.root_dir, crop=self.crop, snr_type=self.snr_type,
            train_eps=self.train_eps, seed=self.seed, threshold=self.threshold,
            max_res_schedule=self.max_res_schedule)

    def train_len(self):
        return len(self.dataset)

    def _loader(self, epoch, rank, world_size):
        indices = build_shard_indices(
            len(self.dataset), rank, world_size, epoch, seed=self.seed)
        return DataLoader(
            self.dataset,
            batch_size=self.raw_batch_size,
            sampler=indices,                  # disjoint, pre-shuffled shard
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=_passthrough_collate,
            prefetch_factor=2 if self.num_workers > 0 else None,
        )

    def train_iter(self, model=None, epoch=0, rank=0, world_size=1, device='cuda'):
        """Packed iterator for one epoch's shard.

        `model` is accepted and IGNORED — there is no variance predictor on this
        path — so `train_quadtree.py` can drive either loader unchanged.
        """
        return GTQuadtreePacker(
            self._loader(epoch, rank, world_size),
            max_seq_len=self.max_seq_len, device=device)


# --------------------------------------------------------------------------- #
# Smoke test                                                                   #
# --------------------------------------------------------------------------- #
def _smoke_test():
    """Schedule/plan invariants + end-to-end packing on synthetic data (no files)."""
    torch.manual_seed(0)
    C, H = 4, 32
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ---- schedule lookup ---------------------------------------------------
    sched = normalize_max_res_schedule([[0.0, 1], [0.7, 2], [0.85, 4], [0.95, 8]])
    assert min_leaf_size_at(0.0, sched) == 1
    assert min_leaf_size_at(0.69, sched) == 1
    assert min_leaf_size_at(0.7, sched) == 2
    assert min_leaf_size_at(0.9, sched) == 4
    assert min_leaf_size_at(1.0, sched) == 8

    # ---- plan: coverage, cap, and threshold extremes ------------------------
    x0 = torch.randn(C, H, H)
    var = gt_variance_grids(x0)
    assert [tuple(v.shape) for v in var] == [(H // n, H // n) for n in REGION_SIZES]

    for min_leaf in LEAF_SIZES:
        for thr in (0.0, 0.5, float('inf')):
            levels, positions, sizes = plan_from_gt_variance(
                var, thr, H, min_leaf_size=min_leaf)
            n = sizes.shape[0]
            # every token covers (2*size)^2 latent px; the leaves must tile exactly
            covered = sum(int((2 * s) ** 2) for s in sizes.tolist())
            assert covered == H * H, (min_leaf, thr, covered)
            assert positions.shape == (n, 2) and levels.shape == (n,)
            assert (2 ** levels == sizes).all()
            # the cap is a floor on leaf side
            assert int(sizes.min()) >= min_leaf, (min_leaf, thr)

    # thr=0 with no cap -> nothing flat -> all lossless, 16x16 patch grid
    _, _, sizes = plan_from_gt_variance(var, 0.0, H, min_leaf_size=1)
    assert (sizes == 1).all() and sizes.shape[0] == (H // 2) ** 2
    # a size-8 cap forces full compression regardless of variance
    _, _, sizes = plan_from_gt_variance(var, 0.0, H, min_leaf_size=8)
    assert (sizes == 8).all() and sizes.shape[0] == (H // 8 // 2) ** 2
    # thr=inf -> everything flat -> coarsest, cap or no cap
    _, _, sizes = plan_from_gt_variance(var, float('inf'), H, min_leaf_size=1)
    assert (sizes == 8).all()

    # ---- packing + compressor ----------------------------------------------
    from fit.quadtree_compression.predictive_compressor import (
        PredictiveVarianceCompressor)

    comp = PredictiveVarianceCompressor(latent_channels=C, c=64, d=128, crop=H,
                                        pad_to_multiple=128).to(device)

    class _FakePlanned(torch.utils.data.Dataset):
        """Stands in for GTQuadtreeLatentDataset: same item dict, no files."""

        def __len__(self):
            return 40

        def __getitem__(self, i):
            g = torch.Generator().manual_seed(i)
            x0 = torch.randn(C, H, H, generator=g)
            t = torch.rand((), generator=g)
            eps = torch.randn(C, H, H, generator=g)
            levels, positions, sizes = plan_from_gt_variance(
                gt_variance_grids(x0), 0.9, H,
                min_leaf_size=min_leaf_size_at(t, sched))
            return dict(x_t=(1 - t) * x0 + t * eps, x0=x0, t=t.float(),
                        label=torch.tensor(i % 10, dtype=torch.long),
                        plan=dict(levels=levels, positions=positions, sizes=sizes),
                        n_tok=int(sizes.shape[0]))

    loader = DataLoader(_FakePlanned(), batch_size=8,
                        collate_fn=_passthrough_collate)
    packer = GTQuadtreePacker(loader, max_seq_len=512, device=device)

    seen = 0
    for step, sel in enumerate(packer):
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

    assert seen == 40, f"packer dropped samples: {seen}/40"
    print(f"packed {seen} images total; smoke test passed")


if __name__ == '__main__':
    _smoke_test()
