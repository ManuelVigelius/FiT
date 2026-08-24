"""
Variance-guided quadtree compression of a noisy latent, with a patchified output.

Given a noisy latent x_t and its timestep t, we ask the variance predictor from
`variance_prediction.py` how much (channel-max) variance the final image x0 has
over the non-overlapping 2x2 / 4x4 / 8x8 regions of x_t. Low-variance regions are
"flat" in x0 and can be represented coarsely; high-variance regions must be kept
fine. A `threshold` decides the cut: a region is compressible iff its predicted
variance is below it.

Output convention
-----------------
The quadtree tiles the (32x32) latent with square *leaves* of side 1, 2, 4 or 8
latent pixels. A size-1 leaf is a single RAW latent pixel — no compression at all.
A leaf of side N > 1 carries one per-channel value = the average of the noisy latent
x_t over that N x N region (channel-wise mean pool), matching the region convention
of the variance model.

The output is then PATCHIFIED: 2x2 neighbouring leaves are concatenated along the
channel axis into a single token, halving the spatial grid. All four leaves in a
patch must therefore share the same size — so a leaf can only be compressed as far
as the *coarsest* size that the whole 2x2 patch it belongs to can afford. We
enforce this by running the quadtree on the PATCH grid: the finest patch unit is a
2x2 block of size-1 leaves (a 2x2 latent-pixel area = one lossless patchify token).

Patch levels (leaf side N in latent px  ->  token spatial extent  ->  patch grid):
    level 0 : N=1  -> 2x2 latent px   -> 16x16 patch grid  (LOSSLESS, raw pixels)
    level 1 : N=2  -> 4x4 latent px   -> 8x8 patch grid
    level 2 : N=4  -> 8x8 latent px   -> 4x4 patch grid
    level 3 : N=8  -> 16x16 latent px -> 2x2 patch grid

A patch sits at a size-N level (N in 2/4/8) iff every one of its four size-N leaves
is compressible, i.e. the model's variance at scale N is below `threshold` there.
Size 1 has no variance prediction — it is the fallback the tree lands on when even
the size-2 patch above it isn't flat, so it is emitted uncompressed and never split
further. The recursion is top-down (coarsest first) and consistent: a patch only
collapses to a coarse size when the whole footprint is flat at that size.

`compress` returns three parallel sequences (one entry per output token):
    tokens    : (num_tokens, 4 * latent_channels)  concatenated leaf values
                (order within a patch: row-major over the 2x2 leaves)
    positions : (num_tokens, 2)  (y, x) center of the patch in latent-pixel coords
    sizes     : (num_tokens,)    leaf side N in latent px (1 / 2 / 4 / 8) — the
                token's compression level; N=1 is uncompressed (raw 2x2 pixels),
                N=8 is 4x coarser per axis.
"""

import torch
import torch.nn.functional as F

try:
    # When imported as part of the `fit.quadtree_compression` package.
    from .variance_prediction import VariancePredictor, predict_variance
except ImportError:
    # When run directly from inside this directory (python quadtree_compression.py).
    from variance_prediction import VariancePredictor, predict_variance


# Region sizes the variance model predicts for (finest -> coarsest).
REGION_SIZES = (2, 4, 8)
# Full leaf-side ladder including the lossless size-1 (raw pixel) level. The
# quadtree walks these coarsest -> finest; size 1 is the uncompressed fallback and
# has no variance prediction.
LEAF_SIZES = (1,) + REGION_SIZES


def _region_means(latent, n):
    """Channel-wise mean of the latent over non-overlapping n x n regions.

    latent: (C, H, W) -> (C, H/n, W/n). Matches the avg-pool region convention the
    variance model is trained against.
    """
    return F.avg_pool2d(latent[None].float(), n, stride=n)[0]


@torch.no_grad()
def compress(model, x_t, t, threshold):
    """Variance-guided, patchified quadtree compression of a single noisy latent.

    model     : trained VariancePredictor (region_sizes == (2, 4, 8)).
    x_t       : (C, H, W) noisy latent, H == W (32 for this model).
    t         : scalar timestep (float, or 0-dim / 1-elem tensor) in [0, 1].
    threshold : regions with predicted variance < threshold are compressible.

    Returns (tokens, positions, sizes); see module docstring for shapes/semantics.

    This is a thin wrapper: it runs the variance forward for the single latent and
    hands the per-scale variance grids to `compress_from_variance`. When compressing
    a whole batch, run the variance predictor once on the batch and call
    `compress_from_variance` per image instead — see
    `in1k_quadtree_latent_dataset.QuadtreePlanPool`. For the LEARNED
    compression path use `plan_from_variance` instead — it decides the same tree
    without pooling any values, leaving the detail for the PyramidEncoder.
    """
    device = x_t.device
    t = torch.as_tensor(t, dtype=torch.float32, device=device).reshape(1)

    # Per-scale channel-max variance grids: var[s] is (H/N_s, W/N_s), N_s in (2,4,8).
    var = predict_variance(model, x_t[None].to(device), t)
    var = [v[0] for v in var]                       # drop batch dim -> list of (h, w)
    return compress_from_variance(x_t, var, threshold)


def compress_from_variance(x_t, var, threshold, x_target=None):
    """Patchified quadtree compression from *precomputed* per-scale variance grids.

    x_t       : (C, H, W) noisy latent, H == W (32 for this model).
    var       : list of per-scale channel-max variance grids, var[s] of shape
                (H/N_s, W/N_s) for N_s in REGION_SIZES (2, 4, 8) — i.e. exactly one
                image's slice of `predict_variance`'s output.
    threshold : regions with predicted variance < threshold are compressible.
    x_target  : optional (C, H, W) *second* latent (e.g. the clean x0) to compress
                on the SAME quadtree structure. Its leaf values are mean-pooled
                identically and emitted token-aligned with `tokens`. Used to build
                a clean-target training signal without re-deciding the tree.

    Returns (tokens, positions, sizes) — or (tokens, positions, sizes, targets)
    when `x_target` is given, where `targets` is (num_tokens, 4*C) aligned with
    `tokens`. See the module docstring for shapes/semantics.

    Split out from `compress` so the variance predictor can be run once on a whole
    batch and the (per-image, variable-length) quadtree walk done separately.
    """
    device = x_t.device
    C, H, _ = x_t.shape
    var_by_size = {n: var[s] for s, n in enumerate(REGION_SIZES)}

    # Per-size leaf values (channel-mean pooled latent) on each leaf grid; n=1 is
    # the identity, i.e. the raw per-pixel latent (lossless level).
    val_by_size = {n: _region_means(x_t, n) for n in LEAF_SIZES}     # (C, H/n, W/n)
    # Same pooling for the optional parallel target latent (same tree, same means).
    tgt_by_size = (
        {n: _region_means(x_target, n) for n in LEAF_SIZES}
        if x_target is not None else None
    )

    # "flat[n][i, j]" : is the n x n region (i, j) compressible on its own? Only the
    # model's predicted sizes have a flatness test; size 1 has none (never compressed).
    flat_by_size = {n: (var_by_size[n] < threshold) for n in REGION_SIZES}

    tokens, positions, sizes = [], [], []
    targets = [] if x_target is not None else None

    def patch_is_flat(n, ly, lx):
        """Are all four size-n leaves of the patch anchored at leaf (ly, lx) flat?

        The patch groups the 2x2 leaves (ly:ly+2, lx:lx+2) on the size-n leaf grid.
        Patchified leaves must share a size, so the whole patch may collapse to
        size n only if every one of its four leaves is individually compressible.
        """
        f = flat_by_size[n]
        return bool(f[ly:ly + 2, lx:lx + 2].all())

    def _leaf_token(vals, ly, lx):
        # row-major 2x2 leaf block -> concat 4 leaves along channel axis
        block = vals[:, ly:ly + 2, lx:lx + 2]        # (C, 2, 2)
        return block.permute(1, 2, 0).reshape(4 * C)  # (2,2,C)->(4C,), row-major

    def emit(n, ly, lx):
        """Emit one token for the patch of four size-n leaves at leaf-grid (ly, lx)."""
        tokens.append(_leaf_token(val_by_size[n], ly, lx))
        if targets is not None:
            targets.append(_leaf_token(tgt_by_size[n], ly, lx))
        # center of the patch footprint in latent-pixel coords: 2 leaves * n px wide,
        # anchored at leaf (ly, lx) which starts at pixel (ly*n, lx*n).
        cy = ly * n + n                              # (ly*n + (ly+2)*n) / 2
        cx = lx * n + n
        positions.append(torch.tensor([cy, cx], dtype=torch.float32, device=device))
        sizes.append(n)

    def recurse(n, ly, lx):
        """Decide the patch of four size-n leaves at (ly, lx); split if not flat.

        n indexes the patch level (coarsest first). If the patch is flat at this
        (coarse) size it becomes one token; otherwise it splits into four child
        patches at the next finer size (n // 2), each covering one of its leaves.
        Size 1 has no flatness test — it is the lossless leaf and always emits.
        """
        if n > 1 and not patch_is_flat(n, ly, lx):
            # split: each of the 4 size-n leaves becomes a 2x2 block of size-(n/2)
            # leaves, i.e. a child patch. Child leaf-grid anchor doubles.
            m = n // 2
            for dy in (0, 1):
                for dx in (0, 1):
                    recurse(m, 2 * (ly + dy), 2 * (lx + dx))
            return
        emit(n, ly, lx)

    # Start on the coarsest patch grid. Coarsest leaf side is LEAF_SIZES[-1] (8);
    # its leaf grid is (H/8) x (W/8), and patches step by 2 leaves.
    coarse_n = LEAF_SIZES[-1]
    coarse_leaves = H // coarse_n
    for ly in range(0, coarse_leaves, 2):
        for lx in range(0, coarse_leaves, 2):
            recurse(coarse_n, ly, lx)

    tokens = torch.stack(tokens)                     # (num_tokens, 4C)
    positions = torch.stack(positions)               # (num_tokens, 2)
    sizes = torch.tensor(sizes, dtype=torch.long, device=device)  # (num_tokens,)
    if targets is not None:
        return tokens, positions, sizes, torch.stack(targets)
    return tokens, positions, sizes


# --------------------------------------------------------------------------- #
# Smoke test                                                                  #
# --------------------------------------------------------------------------- #
def smoke_test():
    """Shapes + invariants on random data, at a few thresholds."""
    torch.manual_seed(0)
    C, H, W = 4, 32, 32
    model = VariancePredictor(latent_channels=C, region_sizes=REGION_SIZES).eval()
    x_t = torch.randn(C, H, W)
    t = torch.rand(1)

    for thr in (0.0, 0.5, 1e9):
        tokens, positions, sizes = compress(model, x_t, t, threshold=thr)
        n = tokens.shape[0]
        # each token covers (2*size)^2 latent px; total coverage must tile the latent
        covered = sum(int((2 * s) ** 2) for s in sizes.tolist())
        assert covered == H * W, (thr, covered, H * W)
        assert tokens.shape == (n, 4 * C)
        assert positions.shape == (n, 2)
        assert sizes.shape == (n,)
        counts = {s: int((sizes == s).sum()) for s in LEAF_SIZES}
        print(f"threshold={thr:>8}: {n:4d} tokens  by-size={counts}  "
              f"covered={covered}/{H*W}")

    # thr=0 -> nothing compressible -> all lossless (size 1); patch grid 16x16 = 256.
    tokens, _, sizes = compress(model, x_t, t, threshold=0.0)
    assert (sizes == 1).all() and tokens.shape[0] == (H // 1 // 2) ** 2
    # thr=inf -> everything compressible -> all coarsest (size 8); patch grid 2x2 = 4.
    tokens, _, sizes = compress(model, x_t, t, threshold=float("inf"))
    assert (sizes == 8).all() and tokens.shape[0] == (H // 8 // 2) ** 2
    print("smoke test passed")


if __name__ == "__main__":
    smoke_test()


# --------------------------------------------------------------------------- #
# Plan-only walk (for the learned PredictiveVarianceCompressor)               #
# --------------------------------------------------------------------------- #
def plan_from_variance(var, threshold, latent_size):
    """Decide the quadtree *structure* only — no latent values are read.

    This is `compress_from_variance` with the value-pooling stripped out. The
    learned `PyramidEncoder` produces the token contents from the FULL-RESOLUTION
    latent, so the tree decision no longer needs (and must not do) the mean-pool
    that would destroy the very detail the encoder is meant to see.

    var        : list of per-scale channel-max variance grids, var[s] of shape
                 (H/N_s, W/N_s) for N_s in REGION_SIZES (2, 4, 8).
    threshold  : regions with predicted variance < threshold are compressible.
    latent_size: H (== W) of the latent in latent pixels.

    Returns (levels, positions, sizes):
        levels    (n,) long   encoder level l, where leaf side N == 2**l
        positions (n, 2)      (y, x) patch center in latent-pixel coords
        sizes     (n,) long   leaf side N in latent px (1 / 2 / 4 / 8)

    Token ORDER is the recursion order, matching `compress_from_variance`.
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
        if n > 1 and not patch_is_flat(n, ly, lx):
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


def plan_to_masks(levels, positions, sizes, latent_size, n_levels=4):
    """Turn a per-token plan into the per-level boolean masks PyramidEncoder wants.

    PyramidEncoder.features returns level-l features on a (H/2^(l+1)) grid; a
    token at level l with leaf side N == 2**l covers a 2N x 2N latent-pixel patch
    whose top-left is at (cy - N, cx - N). That patch is exactly ONE cell of the
    level-l grid, at index ((cy - N) / (2N), (cx - N) / (2N)).

    Returns (masks, order) where masks is a list of n_levels (1, H_l, W_l) bool
    tensors and `order` is a long tensor that permutes level-major token order
    (level 0 tokens, then level 1, ... — the order PyramidEncoder emits) back into
    the plan's original recursion order.
    """
    H = int(latent_size)
    device = levels.device
    masks = []
    for l in range(n_levels):
        g = H // (2 ** (l + 1))
        masks.append(torch.zeros(1, g, g, dtype=torch.bool, device=device))

    for l in range(n_levels):
        sel = (levels == l)
        if not bool(sel.any()):
            continue
        n = 2 ** l
        pos = positions[sel]                                   # (k, 2) centers
        iy = ((pos[:, 0] - n) / (2 * n)).round().long()
        ix = ((pos[:, 1] - n) / (2 * n)).round().long()
        masks[l][0, iy, ix] = True

    # PyramidEncoder gathers with a boolean mask (row-major within each level) and
    # concatenates levels in order 0..n_levels-1. Recover the mapping back to the
    # plan's recursion order so tokens stay aligned with positions/sizes.
    idx = torch.arange(levels.shape[0], device=device)
    gather_order = []
    for l in range(n_levels):
        sel = (levels == l)
        if not bool(sel.any()):
            continue
        n = 2 ** l
        pos = positions[sel]
        iy = ((pos[:, 0] - n) / (2 * n)).round().long()
        ix = ((pos[:, 1] - n) / (2 * n)).round().long()
        g = H // (2 ** (l + 1))
        # row-major rank within this level's mask == the order gather_tokens yields
        rank = iy * g + ix
        gather_order.append(idx[sel][rank.argsort()])
    order_levelmajor = torch.cat(gather_order) if gather_order else idx
    inverse = torch.empty_like(order_levelmajor)
    inverse[order_levelmajor] = torch.arange(
        order_levelmajor.shape[0], device=device)
    return masks, inverse
