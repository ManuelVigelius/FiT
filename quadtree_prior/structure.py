"""The 17-token sequence encoding of a quadtree, and conversion to/from plans.

Why 17 tokens
-------------
The compressor tiles a 32x32 latent with leaves of side 1/2/4/8, then PATCHIFIES
2x2 neighbouring leaves into one token — so all four leaves in a patch share a
size, and the tree really lives on the 16x16 PATCH grid. A patch-grid cell is
2x2 latent pixels; a leaf of side N spans N x N latent pixels, hence a patch of
side N covers 2N x 2N latent pixels and occupies N x N cells of the patch grid.

    leaf side N=1 -> 1x1 patch cells   (lossless)   16x16 grid of these
    leaf side N=2 -> 2x2 patch cells                 8x8
    leaf side N=4 -> 4x4 patch cells                 4x4
    leaf side N=8 -> 8x8 patch cells (max compress)  2x2

So the whole 16x16 patch grid is exactly a 2x2 arrangement of size-8 patches,
and the quadtree over it has depth 4. Two levels of that tree carry all the
decisions the prior must make:

    token 0        the ROOT. Its four quadrants are the four 8x8 patches.
                   Output: 4 bits (one per quadrant) = 16 classes. Bit set means
                   "this quadrant is NOT compressed at 8x8, descend into it".

    tokens 1..16   one per 4x4 region, in row-major order over the 4x4 grid of
                   4x4-sized patches. Output: 17 classes — either the special
                   class STOP_4 ("this whole region is one size-4 patch") or a
                   4-bit mask over its four 2x2 quadrants, bit set meaning "this
                   quadrant splits to size 1 (lossless)" and bit clear meaning
                   "this quadrant is one size-2 patch".

A 4x4 region that lies inside an 8x8 quadrant the root already compressed is
fully determined; it is neither predicted nor supervised, and its INPUT token
gets the dedicated COVERED embedding so the model can see that it was skipped.

Bit order within a quadrant mask is row-major over the 2x2: bit 0 = top-left,
bit 1 = top-right, bit 2 = bottom-left, bit 3 = bottom-right.

Plan layout
-----------
`plan_from_structure` emits the same (levels, positions, sizes) triple as
`fit.data.in1k_gt_quadtree_latent_dataset.plan_from_gt_variance`, in the same
recursion order (coarsest-first, row-major within each split), so a generated
plan is a drop-in for an oracle one.
"""

import torch

# Patch grid side for a 32x32 latent: 32 // 2.
PATCH_GRID = 16
# Leaf sides the tree may land on, coarsest -> finest.
LEAF_SIZES = (8, 4, 2, 1)

# Sequence layout.
N_REGIONS = 16          # 4x4 regions of size-4 patches
SEQ_LEN = 1 + N_REGIONS  # root + one token per region

# Vocabulary. The root emits a 4-bit mask (16 values); region tokens emit the
# same 4-bit mask plus one extra "stop here at size 4" class.
ROOT_VOCAB = 16
STOP_4 = 16             # region class meaning "no split; one size-4 patch"
REGION_VOCAB = 17

# Input-embedding vocabulary, shared by both positions. Region tokens are fed the
# class the PREVIOUS step emitted, so the embedding table must cover every
# emittable class plus two structural markers.
BOS = REGION_VOCAB      # 17 - start of sequence, fed at position 0
COVERED = REGION_VOCAB + 1  # 18 - region inside an 8x8-compressed quadrant
INPUT_VOCAB = REGION_VOCAB + 2  # 19

# Loss ignore index for masked-out (covered) region targets.
IGNORE_INDEX = -100


def _quadrant_offsets(half):
    """Row-major (dy, dx) cell offsets of the four quadrants of a `2*half` block."""
    return ((0, 0), (0, half), (half, 0), (half, half))


# --------------------------------------------------------------------------- #
# sizes grid  ->  token sequence                                              #
# --------------------------------------------------------------------------- #
def encode_sizes(sizes_grid):
    """Tokenize a (16, 16) per-patch-cell leaf-size grid into the sequence.

    `sizes_grid[y, x]` is the leaf side (1/2/4/8) of the patch covering cell
    (y, x); a size-N patch fills an N x N block of equal entries. This is the
    form `sizes_grid_from_plan` produces from an oracle plan.

    Returns (targets, inputs, loss_mask):
        targets   (17,) long   class to predict at each position; region entries
                               that are covered hold IGNORE_INDEX.
        inputs    (17,) long   embedding ids to FEED, already shifted: position 0
                               is BOS, position i>0 is target i-1, except covered
                               regions which are fed COVERED.
        loss_mask (17,) bool   True where the position is supervised.

    Note `inputs` is built from the ground-truth targets — teacher forcing. At
    sampling time `sample.py` builds the same thing incrementally.
    """
    if tuple(sizes_grid.shape) != (PATCH_GRID, PATCH_GRID):
        raise ValueError(f"sizes_grid must be ({PATCH_GRID}, {PATCH_GRID}), "
                         f"got {tuple(sizes_grid.shape)}")
    grid = sizes_grid.to(torch.long)

    targets = torch.full((SEQ_LEN,), IGNORE_INDEX, dtype=torch.long)
    inputs = torch.empty((SEQ_LEN,), dtype=torch.long)
    loss_mask = torch.zeros((SEQ_LEN,), dtype=torch.bool)

    # ---- root: which of the four 8x8 quadrants are NOT size-8 --------------
    root = 0
    covered = [False] * N_REGIONS       # per 4x4 region, indexed row-major
    for q, (qy, qx) in enumerate(_quadrant_offsets(8)):
        block = grid[qy:qy + 8, qx:qx + 8]
        is_8 = bool((block == 8).all())
        if not is_8:
            root |= (1 << q)
        else:
            # Mark the four 4x4 regions this quadrant contains as determined.
            for ry in range(2):
                for rx in range(2):
                    covered[(qy // 4 + ry) * 4 + (qx // 4 + rx)] = True
    targets[0] = root
    inputs[0] = BOS
    loss_mask[0] = True

    # ---- regions: for each 4x4 block of cells, stop-at-4 or a 2x2 mask -----
    for r in range(N_REGIONS):
        pos = 1 + r
        if covered[r]:
            targets[pos] = IGNORE_INDEX
            inputs[pos] = COVERED
            continue
        ry, rx = (r // 4) * 4, (r % 4) * 4
        block = grid[ry:ry + 4, rx:rx + 4]
        if bool((block == 4).all()):
            cls = STOP_4
        else:
            cls = 0
            for q, (qy, qx) in enumerate(_quadrant_offsets(2)):
                sub = block[qy:qy + 2, qx:qx + 2]
                # Bit set == splits to lossless size 1. A well-formed grid has
                # this sub-block either all-2 or all-1.
                if not bool((sub == 2).all()):
                    cls |= (1 << q)
        targets[pos] = cls
        loss_mask[pos] = True

    # Teacher forcing: feed the previous position's target, except where the
    # position is covered (already written above).
    for pos in range(1, SEQ_LEN):
        if inputs[pos] != COVERED:
            prev = targets[pos - 1]
            # A covered predecessor emitted nothing; feed COVERED again.
            inputs[pos] = COVERED if prev == IGNORE_INDEX else prev

    return targets, inputs, loss_mask


def covered_mask_from_root(root_cls):
    """Which of the 16 region positions the root's 4-bit mask already determines.

    Returns a (16,) bool tensor, True where the region needs no prediction.
    """
    covered = torch.zeros(N_REGIONS, dtype=torch.bool)
    for q, (qy, qx) in enumerate(_quadrant_offsets(8)):
        if not (int(root_cls) >> q) & 1:        # quadrant is a size-8 patch
            for ry in range(2):
                for rx in range(2):
                    covered[(qy // 4 + ry) * 4 + (qx // 4 + rx)] = True
    return covered


# --------------------------------------------------------------------------- #
# token sequence  ->  sizes grid                                              #
# --------------------------------------------------------------------------- #
def decode_sizes(tokens):
    """Rebuild the (16, 16) leaf-size grid from a (17,) class sequence.

    Region entries that the root marked covered are ignored, so a sampler is free
    to leave them at any value.
    """
    tokens = tokens.to(torch.long)
    grid = torch.zeros((PATCH_GRID, PATCH_GRID), dtype=torch.long)
    root = int(tokens[0])

    for q, (qy, qx) in enumerate(_quadrant_offsets(8)):
        if not (root >> q) & 1:
            grid[qy:qy + 8, qx:qx + 8] = 8
            continue
        # Descend: the quadrant's four 4x4 regions each have their own token.
        for ry in range(2):
            for rx in range(2):
                r = (qy // 4 + ry) * 4 + (qx // 4 + rx)
                by, bx = ry * 4 + qy, rx * 4 + qx
                cls = int(tokens[1 + r])
                if cls == STOP_4:
                    grid[by:by + 4, bx:bx + 4] = 4
                    continue
                for sq, (sy, sx) in enumerate(_quadrant_offsets(2)):
                    val = 1 if (cls >> sq) & 1 else 2
                    grid[by + sy:by + sy + 2, bx + sx:bx + sx + 2] = val
    return grid


# --------------------------------------------------------------------------- #
# plan  <->  sizes grid                                                       #
# --------------------------------------------------------------------------- #
def sizes_grid_from_plan(plan, patch_grid=PATCH_GRID):
    """Rasterize a (levels, positions, sizes) plan onto the patch-cell grid.

    `positions` are (y, x) patch CENTERS in latent-pixel coords, the convention
    `plan_from_gt_variance` emits: a size-N patch centered at (cy, cx) spans
    latent pixels [cy - N, cy + N) x [cx - N, cx + N), i.e. patch cells
    [(cy - N)/2, ...) of extent N.
    """
    sizes = plan['sizes'].to(torch.long)
    pos = plan['positions'].to(torch.float32)
    grid = torch.zeros((patch_grid, patch_grid), dtype=torch.long)
    for k in range(sizes.shape[0]):
        n = int(sizes[k])
        cy, cx = float(pos[k, 0]), float(pos[k, 1])
        y0 = int(round((cy - n) / 2))       # latent px -> patch cells
        x0 = int(round((cx - n) / 2))
        grid[y0:y0 + n, x0:x0 + n] = n
    if bool((grid == 0).any()):
        raise ValueError("plan does not tile the patch grid")
    return grid


def plan_from_sizes_grid(grid, device=None):
    """Turn a (16, 16) leaf-size grid back into a (levels, positions, sizes) plan.

    Emission order matches `plan_from_gt_variance`: recurse coarsest-first from
    the size-8 patches, row-major over each 2x2 split, emitting a leaf where the
    grid says the patch stops. That keeps generated plans byte-comparable with
    oracle ones for the same structure.
    """
    grid = grid.to(torch.long)
    levels, positions, sizes = [], [], []

    def emit(n, y0, x0):
        levels.append(int(n).bit_length() - 1)        # N=1,2,4,8 -> 0,1,2,3
        # Inverse of sizes_grid_from_plan: cells -> latent-pixel center.
        positions.append((float(y0 * 2 + n), float(x0 * 2 + n)))
        sizes.append(n)

    def recurse(n, y0, x0):
        if int(grid[y0, x0]) == n:
            emit(n, y0, x0)
            return
        m = n // 2
        for dy in (0, m):
            for dx in (0, m):
                recurse(m, y0 + dy, x0 + dx)

    for y0 in range(0, PATCH_GRID, 8):
        for x0 in range(0, PATCH_GRID, 8):
            recurse(8, y0, x0)

    return dict(
        levels=torch.tensor(levels, dtype=torch.long, device=device),
        positions=torch.tensor(positions, dtype=torch.float32, device=device),
        sizes=torch.tensor(sizes, dtype=torch.long, device=device),
    )


def plan_from_structure(tokens, device=None):
    """Convenience: (17,) class sequence -> compressor plan dict."""
    return plan_from_sizes_grid(decode_sizes(tokens), device=device)


def n_tokens_from_sizes_grid(grid):
    """Token count the compressor will emit for this structure.

    A size-N patch occupies N x N cells and is ONE token, so the count is the sum
    of 1/N^2 over cells — computed exactly in integer arithmetic below.
    """
    grid = grid.to(torch.long)
    total = 0
    for n in LEAF_SIZES:
        total += int((grid == n).sum()) // (n * n)
    return total


# --------------------------------------------------------------------------- #
# Smoke test                                                                   #
# --------------------------------------------------------------------------- #
def _smoke_test():
    """Round-trip invariants on random valid trees, plus the two extremes."""
    torch.manual_seed(0)

    def random_grid(gen):
        g = torch.zeros(PATCH_GRID, PATCH_GRID, dtype=torch.long)

        def rec(n, y, x):
            if n == 1 or torch.rand((), generator=gen) < 0.35:
                g[y:y + n, x:x + n] = n
                return
            m = n // 2
            for dy in (0, m):
                for dx in (0, m):
                    rec(m, y + dy, x + dx)

        for y in range(0, PATCH_GRID, 8):
            for x in range(0, PATCH_GRID, 8):
                rec(8, y, x)
        return g

    gen = torch.Generator().manual_seed(0)
    for _ in range(200):
        g = random_grid(gen)
        targets, inputs, mask = encode_sizes(g)

        # tokens -> grid is exact (covered slots are irrelevant to the decode)
        filled = torch.where(targets == IGNORE_INDEX, torch.zeros_like(targets), targets)
        assert torch.equal(decode_sizes(filled), g)

        # grid <-> plan is exact, and the plan tiles the latent
        plan = plan_from_sizes_grid(g)
        assert torch.equal(sizes_grid_from_plan(plan), g)
        assert sum(int((2 * s) ** 2) for s in plan['sizes'].tolist()) == (2 * PATCH_GRID) ** 2
        assert plan['sizes'].shape[0] == n_tokens_from_sizes_grid(g)
        assert (2 ** plan['levels'] == plan['sizes']).all()

        # the root's mask is exactly the set of unsupervised region positions
        assert torch.equal(covered_mask_from_root(int(targets[0])), ~mask[1:])
        assert (targets[1:][~mask[1:]] == IGNORE_INDEX).all()
        assert (targets[1:][mask[1:]] >= 0).all()
        assert inputs[0] == BOS

    # extremes: maximal compression and none at all
    all_coarse = torch.full((PATCH_GRID, PATCH_GRID), 8, dtype=torch.long)
    targets, _, mask = encode_sizes(all_coarse)
    assert int(targets[0]) == 0 and int(mask.sum()) == 1
    assert n_tokens_from_sizes_grid(all_coarse) == 4

    lossless = torch.ones(PATCH_GRID, PATCH_GRID, dtype=torch.long)
    targets, _, mask = encode_sizes(lossless)
    assert int(targets[0]) == 15 and bool(mask.all()) and (targets[1:] == 15).all()
    assert n_tokens_from_sizes_grid(lossless) == PATCH_GRID ** 2

    print("structure smoke test passed (200 random trees + extremes)")


if __name__ == '__main__':
    _smoke_test()
