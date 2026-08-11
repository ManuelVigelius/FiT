"""Diagnostic / dev test for the quadtree (mixed-resolution) upsample placement.

Motivation
----------
The Loss-C learned-upsampler path currently feeds the model one *uniform* low-res
rectangle per image and stretches it over the whole full-res output
(`FiT._upsample_packed`). To render some regions at higher quality than others we
want a *quadtree*: one image is partitioned into rectangular cells, each cell
carrying its own low-res token resolution. The transformer and packing are
unchanged (one image = one document); only the upsample/placement step changes —
each cell's tokens are placed into *its own sub-window* of the full-res grid by
nearest-neighbour block fill (each token covers an (h/k_h)x(w/k_w) block) and
composited.

This script isolates that placement step. It does NOT touch the model: it
reimplements the per-cell windowed grid_sample as a standalone function
(`upsample_quadtree`) operating on a packed token tensor + an explicit cell list,
and checks two properties that must hold before we graft it into the model:

  1. **Degenerate equivalence.** A single full-cell quadtree (one cell covering
     the whole frame at full-res token density) reproduces the identity: the
     output equals the input tokens, laid out on the full-res grid. (And a single
     coarse cell reproduces NN block fill — each token a constant block.)

  2. **Partition / placement correctness.** For a multi-cell quadtree that tiles
     the frame exactly, every full-res output token is written exactly once
     (no gaps, no overlaps), and each cell's window contains the NN block fill of
     *that cell's* tokens (matched against an independent per-window reference).

Cell representation (full-res token units on the H_fr x W_fr grid):
    Cell = (offset, k_h, k_w, y0, x0, h, w)
      offset       : start index of this cell's tokens in the packed sequence
      k_h, k_w     : the cell's own low-res token grid (k_h*k_w tokens)
      y0, x0       : top-left of the cell's window in the full-res grid
      h,  w        : window extent in the full-res grid (k_h<=h, k_w<=w)
    The cells must tile [0,H_fr) x [0,W_fr) with no gaps or overlaps.

Run (CPU is fine):
    python tools/test_quadtree_upsample.py
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# --------------------------------------------------------------------------- #
# The placement function under test (model-free reimplementation).
# --------------------------------------------------------------------------- #
def upsample_quadtree(x, cells, H_fr, W_fr, D):
    """Quadtree nearest-neighbour block fill: place each cell into its window.

    Each token of a cell owns an (h/k_h) x (w/k_w) full-res block and its value
    is copied to every position in that block (no interpolation). A full-res
    (k==window) cell is the identity; a 1-token cell becomes a constant block.

    Args:
        x:     (1, N_total, D) packed low-res tokens for ONE image (one document).
               Each cell occupies a contiguous run [offset, offset + k_h*k_w),
               row-major (w-fast / h-slow), matching the dataset grid convention.
        cells: list of (offset, k_h, k_w, y0, x0, h, w) tiling the full-res frame.
        H_fr, W_fr: full-res token grid size.
        D:     hidden dim.

    Returns:
        x_fr: (1, H_fr*W_fr, D) dense full-res tokens.
        written: (H_fr, W_fr) int tensor counting writes per output cell
                 (returned for the test's partition check; the model would drop it).
    """
    device = x.device
    out = x.new_zeros(D, H_fr, W_fr)
    written = torch.zeros(H_fr, W_fr, dtype=torch.long, device=device)

    for (offset, k_h, k_w, y0, x0, h, w) in cells:
        n = k_h * k_w
        tok = x[0, offset:offset + n]                       # (k_h*k_w, D)
        cell = tok.transpose(0, 1).reshape(D, k_h, k_w)     # (D, k_h, k_w)

        # NN block fill: each token covers an (h/k_h) x (w/k_w) block.
        assert h % k_h == 0 and w % k_w == 0
        rh, rw = h // k_h, w // k_w
        win = cell.repeat_interleave(rh, dim=1).repeat_interleave(rw, dim=2)

        out[:, y0:y0 + h, x0:x0 + w] = win
        written[y0:y0 + h, x0:x0 + w] += 1

    x_fr = out.reshape(D, H_fr * W_fr).transpose(0, 1).unsqueeze(0)  # (1, N_fr, D)
    return x_fr, written


# --------------------------------------------------------------------------- #
# Helpers for the test.
# --------------------------------------------------------------------------- #
def pack_cells(cell_specs, D, dtype=torch.float64):
    """Build a packed token tensor + cell list from (k_h,k_w,y0,x0,h,w) specs.

    Fills each cell's k_h*k_w tokens with random values and assigns contiguous
    offsets. Returns (x, cells, source_blocks) where source_blocks[i] is the
    (D,k_h,k_w) ground-truth content of cell i (for independent re-checking).
    """
    blocks, cells, chunks = [], [], []
    offset = 0
    for (k_h, k_w, y0, x0, h, w) in cell_specs:
        block = torch.randn(D, k_h, k_w, dtype=dtype)
        blocks.append(block)
        tok = block.reshape(D, k_h * k_w).transpose(0, 1)   # (k_h*k_w, D)
        chunks.append(tok)
        cells.append((offset, k_h, k_w, y0, x0, h, w))
        offset += k_h * k_w
    x = torch.cat(chunks, dim=0).unsqueeze(0)               # (1, N_total, D)
    return x, cells, blocks


def report(name, ok):
    print(f"  {'OK  ' if ok else 'FAIL'}  {name}")
    return ok


# --------------------------------------------------------------------------- #
def main():
    torch.manual_seed(0)
    D = 3
    all_ok = True

    # --- Test 1: degenerate single full-res cell == identity ----------------
    print("Test 1: single full-res cell reproduces the input tokens")
    H_fr = W_fr = 8
    x, cells, blocks = pack_cells([(8, 8, 0, 0, 8, 8)], D)
    x_fr, written = upsample_quadtree(x, cells, H_fr, W_fr, D)
    ok = torch.allclose(x_fr, x) and (written == 1).all()
    all_ok &= report("output equals input, every pixel written once", ok)

    # --- Test 2: single coarse cell == NN block fill ------------------------
    print("Test 2: single coarse cell matches NN block fill")
    H_fr = W_fr = 16
    x, cells, blocks = pack_cells([(4, 4, 0, 0, 16, 16)], D)
    x_fr, written = upsample_quadtree(x, cells, H_fr, W_fr, D)
    # 4x4 tokens over a 16x16 window -> each token a constant 4x4 block.
    ref = blocks[0].repeat_interleave(4, dim=1).repeat_interleave(4, dim=2)  # (D,16,16)
    ref_tok = ref.reshape(D, 256).transpose(0, 1).unsqueeze(0).to(x.dtype)
    ok = torch.allclose(x_fr, ref_tok) and (written == 1).all()
    all_ok &= report("coarse cell upsample matches NN block fill", ok)

    # --- Test 3: multi-cell quadtree, exact partition + per-cell content ----
    print("Test 3: 4-quadrant quadtree (mixed resolution) tiles exactly")
    H_fr = W_fr = 16
    # 3 coarse quadrants (k=4 over an 8x8 window) + 1 fine quadrant (k=8).
    specs = [
        (4, 4, 0, 0, 8, 8),    # top-left  coarse
        (4, 4, 0, 8, 8, 8),    # top-right coarse
        (4, 4, 8, 0, 8, 8),    # bot-left  coarse
        (8, 8, 8, 8, 8, 8),    # bot-right fine (native density in its window)
    ]
    x, cells, blocks = pack_cells(specs, D)
    x_fr, written = upsample_quadtree(x, cells, H_fr, W_fr, D)

    # 3a. exact partition: every output token written exactly once.
    part_ok = (written == 1).all()
    report("partition: every full-res token written exactly once", part_ok)

    # 3b. each window holds the NN block fill of its own cell.
    out = x_fr[0].transpose(0, 1).reshape(D, H_fr, W_fr)
    content_ok = True
    for (block, (offset, k_h, k_w, y0, x0, h, w)) in zip(blocks, cells):
        ref_win = block.repeat_interleave(h // k_h, dim=1).repeat_interleave(
            w // k_w, dim=2).to(out.dtype)
        got_win = out[:, y0:y0 + h, x0:x0 + w]
        content_ok &= torch.allclose(got_win, ref_win)
    report("content: each window = NN block fill of its cell", content_ok)
    all_ok &= part_ok and content_ok

    # --- Test 4: the actual model method FiT._upsample_quadtree --------------
    # Validates the graft end-to-end: (4a) a single full-*resolution* cell (k ==
    # frame) reproduces _upsample_packed exactly — at native density both paths
    # are the identity, so this still guards the no-op case; and (4b) the model
    # method agrees with the standalone NN reference on a mixed-resolution
    # quadtree. (A *coarse* single cell no longer matches _upsample_packed: the
    # quadtree path is NN block fill, the packed path is bicubic — intentional.)
    print("Test 4: FiT._upsample_quadtree (model method)")
    from fit.model.fit_model import FiT

    H_fr = W_fr = 16
    # Minimal model — we only call the (param-free) upsample helpers, which take
    # the feature dim D as an explicit argument and never read self.hidden_size,
    # so the model's hidden_size is free to satisfy RoPE's head_dim constraint
    # (head_dim // 2 must be even) independently of the test's D.
    model = FiT(
        depth=1, hidden_size=8, num_heads=1, in_channels=1, patch_size=1,
        num_classes=1, use_upsampler=True, use_size_cond=False,
        online_rope=True, adaln_type='normal',
    ).eval()
    size_fr = torch.tensor([H_fr, W_fr], dtype=torch.int32).view(1, 1, 2)

    # 4a. Single full-*resolution* cell (k == frame): both paths are the identity
    # placement, so the NN quadtree path and bicubic packed path coincide.
    k = H_fr
    tok = torch.randn(1, k * k, D, dtype=torch.float64)
    doc_ids = torch.zeros(1, k * k, dtype=torch.int32)
    size_lr = torch.tensor([k, k], dtype=torch.int32).view(1, 1, 2)
    with torch.no_grad():
        ref = model._upsample_packed(tok, doc_ids, size_lr, size_fr, 1, D)
        cells = [[(0, k, k, 0, 0, H_fr, W_fr)]]
        got = model._upsample_quadtree(tok, doc_ids, size_fr, 1, D, cells)
    # _upsample_packed bicubic-resamples even at native density, leaving ~2e-6
    # float noise; the NN quadtree path is exact identity. Compare loosely.
    reg_ok = torch.allclose(got, ref, atol=1e-5)
    all_ok &= report("single full-res cell == _upsample_packed (identity)", reg_ok)

    # 4b. Mixed quadtree through the model method vs the standalone NN reference
    # (both are exact integer copies — no resampling — so this is exact).
    x, cells_flat, _ = pack_cells(specs, D)               # reuse Test-3 specs
    doc_ids = torch.zeros(1, x.shape[1], dtype=torch.int32)
    cells_model = [cells_flat]                            # one image
    with torch.no_grad():
        got = model._upsample_quadtree(x, doc_ids, size_fr, 1, D, cells_model)
    ref_fr, _ = upsample_quadtree(x, cells_flat, H_fr, W_fr, D)
    match_ok = torch.allclose(got, ref_fr)
    all_ok &= report("model method matches NN block-fill reference", match_ok)

    all_ok &= test_quadtree_grid()
    all_ok &= test_end_to_end()

    print()
    print("ALL PASS" if all_ok else "SOME FAILED")
    return 0 if all_ok else 1


def test_end_to_end():
    """Full mixed-resolution path: sampler -> input builder -> model -> upsample.

    Runs the quadtree sampler against a tiny real FiT (use_upsampler=True) on CPU.
    Checks the pieces connect (no shape/unit errors) and the output shape is the
    full-res latent. Also checks the degenerate all-native quadtree (every token
    its own 1x1 cell) is finite and full-res — i.e. the mixed-res plumbing
    reduces to a sane native-resolution run.

    Uses b=1 and block_mask=None so attention takes the CPU SDPA path: a single
    image is one document, needing no cross-document masking. FlexAttention (the
    multi-image packed path) is unchanged from the existing uniform upsampler and
    only runs on CUDA, so it is exercised by the real generation run, not here.
    """
    from fit.model.fit_model import FiT
    from fit.noise_field_sampler.quadtree import (
        Cell, Quadtree, build_refinement_quadtree, quadtree_grid,
    )
    from fit.noise_field_sampler.noise_field_sampler import sample_upsampler_quadtree
    from fit.scheduler.transport.utils import patchify, unpatchify

    print("Test 6: end-to-end quadtree sampler through a tiny FiT")
    ok_all = True

    torch.manual_seed(0)
    P = 2
    H_fr = W_fr = 8           # grid; full-res latent is 16x16
    C = 4
    model = FiT(
        depth=1, hidden_size=16, num_heads=1, in_channels=C, patch_size=P,
        num_classes=1, use_upsampler=True, use_size_cond=False, learn_sigma=False,
        online_rope=True, adaln_type='normal', class_dropout_prob=0.0,
    ).double().eval()

    b = 1
    y = torch.zeros(b, dtype=torch.long)

    def model_fn_qt(cell_blocks, x_fr_sp, t, qt):
        # No CFG here — single (non-doubled) batch for the test. With n_pack=1
        # the packed sequence is one document, so block_mask=None + the CPU SDPA
        # padding-mask path suffices (no FlexAttention).
        n_pack = x_fr_sp.shape[0]
        seq_lr = qt.n_tokens
        N_total = seq_lr                                    # no padding needed at b=1
        per_cell_tok = [patchify(blk, P) for blk in cell_blocks]
        tok_img = torch.cat(per_cell_tok, dim=1)            # (n_pack, seq_lr, Dtok)
        Dtok = tok_img.shape[-1]
        feat = torch.zeros(1, N_total, Dtok, dtype=tok_img.dtype)
        grid = torch.zeros(1, 2, N_total, dtype=tok_img.dtype)
        mask = torch.zeros(1, N_total, dtype=tok_img.dtype)
        doc_ids = torch.full((1, N_total), -1, dtype=torch.int32)
        g_phys = quadtree_grid(qt, 'cpu', tok_img.dtype)
        off = 0
        for i in range(n_pack):
            feat[0, off:off + seq_lr] = tok_img[i]
            grid[0, :, off:off + seq_lr] = g_phys
            mask[0, off:off + seq_lr] = 1
            doc_ids[0, off:off + seq_lr] = i
            off += seq_lr
        size_lr = torch.tensor([H_fr, W_fr], dtype=torch.int32).view(1, 1, 2).repeat(1, n_pack, 1)
        block_mask = None

        x_fr_tok = patchify(x_fr_sp, P)
        grid_fr = torch.stack([
            torch.arange(W_fr).repeat(H_fr), torch.arange(H_fr).repeat_interleave(W_fr)
        ]).unsqueeze(0).repeat(n_pack, 1, 1)
        mask_fr = torch.ones(n_pack, H_fr * W_fr, dtype=tok_img.dtype)
        size_fr = torch.tensor([H_fr, W_fr], dtype=torch.int32).view(1, 1, 2).repeat(1, n_pack, 1)
        cells = [qt.placement_cells() for _ in range(n_pack)]

        t_pack = t[:1].repeat(1, n_pack)
        v = model(
            feat, t_pack, torch.zeros(1, n_pack, dtype=torch.int), grid, mask, size_lr,
            doc_ids=doc_ids, block_mask=block_mask, x_fullres=x_fr_tok,
            grid_fullres=grid_fr, mask_fullres=mask_fr, size_fullres=size_fr, cells=cells,
        )
        return unpatchify(v, (H_fr * P, W_fr * P), P)

    # 6a. Refinement quadtree (coarse base + sharp center) runs and is full-res.
    qt = build_refinement_quadtree(H_fr, W_fr, base_k=4, refine=[(2, 2, 4, 4, 4)])
    with torch.no_grad():
        out = sample_upsampler_quadtree(
            model_fn_qt, qt, num_steps=3, b=b, d=C, patch_size=P, device='cpu',
            dtype=torch.float64,
        )
    shape_ok = (out.shape == (b, C, H_fr * P, W_fr * P)) and torch.isfinite(out).all()
    ok_all &= report("refinement quadtree sampler runs, output is finite full-res", shape_ok)

    # 6b. Degenerate all-native quadtree (every token a 1x1 cell at full density).
    cells = [Cell(1, 1, yy, xx, 1, 1) for yy in range(H_fr) for xx in range(W_fr)]
    qt_native = Quadtree(H_fr, W_fr, cells); qt_native.validate()
    with torch.no_grad():
        out2 = sample_upsampler_quadtree(
            model_fn_qt, qt_native, num_steps=3, b=b, d=C, patch_size=P, device='cpu',
            dtype=torch.float64,
        )
    native_ok = (out2.shape == (b, C, H_fr * P, W_fr * P)) and torch.isfinite(out2).all()
    ok_all &= report("all-native quadtree sampler runs, output is finite full-res", native_ok)

    return ok_all


def test_quadtree_grid():
    """RoPE physical-center grid invariants (step 2)."""
    from fit.noise_field_sampler.quadtree import (
        Cell, Quadtree, quadtree_grid, build_refinement_quadtree,
    )
    print("Test 5: quadtree physical-center RoPE grid")
    ok_all = True

    # 5a. Uniform full-res quadtree (one 1x1 cell per token) reproduces the
    #     trained integer grid shifted by a global +0.5. Since RoPE is relative,
    #     this is the in-distribution convention: pairwise differences are
    #     exactly the integer-grid differences.
    H = W = 6
    cells = [Cell(1, 1, y, x, 1, 1) for y in range(H) for x in range(W)]
    qt = Quadtree(H, W, cells); qt.validate()
    grid = quadtree_grid(qt, device='cpu')                 # (2, H*W)
    # Reference integer grid (w-fast/h-slow), as _single_grid builds it.
    ref_w = torch.arange(W).repeat(H).float()
    ref_h = torch.arange(H).repeat_interleave(W).float()
    shift_ok = (torch.allclose(grid[0], ref_w + 0.5) and
                torch.allclose(grid[1], ref_h + 0.5))
    ok_all &= report("uniform full-res grid == integer grid + 0.5 (global shift)", shift_ok)

    # 5b. Relative positions are exactly the integer-grid relatives (the +0.5
    #     cancels) — the property that makes the shift harmless for RoPE.
    rel_ok = torch.allclose(grid[0][:, None] - grid[0][None, :],
                            ref_w[:, None] - ref_w[None, :])
    ok_all &= report("pairwise w-differences match integer grid (shift cancels)", rel_ok)

    # 5c. A coarse cell's tokens sit at the window center with physical spacing
    #     h/k_h > 1 — the true physical positions, in the same frame as fine cells.
    qt2 = Quadtree(4, 4, [Cell(2, 2, 0, 0, 4, 4)]); qt2.validate()
    g2 = quadtree_grid(qt2, device='cpu')
    # 2 tokens over a window of extent 4 -> centers at 1.0 and 3.0.
    centers_ok = torch.allclose(torch.unique(g2[1]), torch.tensor([1.0, 3.0]))
    ok_all &= report("coarse 2x2-over-4x4 cell centers at {1.0, 3.0}", centers_ok)

    # 5d. The refinement builder produces a valid partition and token count.
    qt3 = build_refinement_quadtree(
        H_fr=16, W_fr=16, base_k=8,
        refine=[(8, 8, 8, 8, 8)],          # bottom-right quadrant at k=8 (native)
    )
    # base: 8x8 = 64 base cells, minus the 4x4 block the refine window covers
    # (its 8x8 full-res window spans 4x4 base cells), plus the one refine cell.
    builder_ok = (qt3.n_tokens == (64 - 16) + 8 * 8)
    ok_all &= report("refinement builder: valid partition + token count", builder_ok)

    return ok_all


if __name__ == "__main__":
    raise SystemExit(main())
