"""Quadtree resolution maps for mixed-resolution generation.

A quadtree partitions one image's full-res token grid (H_fr x W_fr) into
non-overlapping rectangular **cells**, each carrying its own low-res token
resolution (k_h x k_w). Coarse cells cover a large window with few tokens; fine
cells approach native density (k == window extent). Because the cells tile the
frame exactly, the union of their tokens forms one image = one document for the
transformer (we never split documents — see fit_model.FiT.forward), and the
union of their windows covers the full-res output exactly once.

This module owns three things, deliberately kept free of any model dependency so
they can be unit-tested in isolation:

  * :class:`Cell` / :class:`Quadtree`  — the data structure.
  * :func:`quadtree_grid`              — the *physical-center* RoPE grid for one
                                         quadtree's packed token sequence.
  * :func:`build_refinement_quadtree`  — a simple demo builder (uniform base
                                         resolution + rectangular refine windows).

Coordinate frame
----------------
Everything is in **full-res token units** on the H_fr x W_fr grid. A cell with
window (y0, x0, h, w) covers full-res rows [y0, y0+h) and cols [x0, x0+w). Its
k_h x k_w low-res tokens are evenly spaced *cell centers* inside that window:

    token (a, b)  ->  phys_y = y0 + (a + 0.5) * h / k_h
                      phys_x = x0 + (b + 0.5) * w / k_w

RoPE is relative, so a global +0.5 shift is harmless; for a uniform full-res
quadtree (one 1x1 cell per token) this reduces to the trained integer grid
shifted by +0.5, i.e. the model sees an in-distribution coordinate convention.
For coarse cells the centers are spaced by h/k_h > 1, which is the true physical
spacing — that is what lets coarse and fine tokens share one coordinate frame.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Cell:
    """One quadtree leaf, all coords in full-res token units.

    Attributes:
        k_h, k_w: this cell's low-res token grid (k_h * k_w tokens).
        y0, x0:   top-left of the cell's window in the full-res grid.
        h, w:     window extent in the full-res grid (k_h <= h, k_w <= w).
    """
    k_h: int
    k_w: int
    y0: int
    x0: int
    h: int
    w: int


@dataclass(frozen=True)
class Quadtree:
    """A partition of one H_fr x W_fr full-res frame into cells."""
    H_fr: int
    W_fr: int
    cells: list[Cell]

    @property
    def n_tokens(self) -> int:
        return sum(c.k_h * c.k_w for c in self.cells)

    def validate(self) -> None:
        """Assert the cells tile [0,H_fr) x [0,W_fr) with no gaps or overlaps."""
        covered = torch.zeros(self.H_fr, self.W_fr, dtype=torch.long)
        for c in self.cells:
            assert c.k_h >= 1 and c.k_w >= 1, f"degenerate cell {c}"
            assert c.k_h <= c.h and c.k_w <= c.w, f"cell denser than window: {c}"
            assert 0 <= c.y0 and c.y0 + c.h <= self.H_fr, f"cell out of frame: {c}"
            assert 0 <= c.x0 and c.x0 + c.w <= self.W_fr, f"cell out of frame: {c}"
            covered[c.y0:c.y0 + c.h, c.x0:c.x0 + c.w] += 1
        assert (covered == 1).all(), "cells do not tile the frame exactly"

    def placement_cells(self) -> list[tuple[int, int, int, int, int, int, int]]:
        """List of (offset, k_h, k_w, y0, x0, h, w) for FiT._upsample_quadtree.

        `offset` is the start index of each cell's tokens in this quadtree's
        packed sequence, in cell order (tokens are concatenated cell by cell).
        """
        out = []
        offset = 0
        for c in self.cells:
            out.append((offset, c.k_h, c.k_w, c.y0, c.x0, c.h, c.w))
            offset += c.k_h * c.k_w
        return out


def quadtree_grid(qt: Quadtree, device, dtype=torch.float32) -> torch.Tensor:
    """Physical-center RoPE grid for one quadtree's packed token sequence.

    Returns a (2, n_tokens) tensor [w_coord; h_coord] (w-fast/h-slow within each
    cell, cells concatenated in order) holding each token's physical *center*
    coordinate in full-res token units. This matches the dataset grid layout
    (w first, h second) consumed by VisionRotaryEmbedding.

    Float dtype is required: coarse-cell centers are fractional, and the model
    must run the *online* RoPE path (online_rope=True), which multiplies grid by
    frequencies rather than indexing a cached integer table.
    """
    w_coords = []
    h_coords = []
    for c in qt.cells:
        a = torch.arange(c.k_h, dtype=dtype)                 # row index within cell
        b = torch.arange(c.k_w, dtype=dtype)                 # col index within cell
        phys_y = c.y0 + (a + 0.5) * (c.h / c.k_h)            # (k_h,)
        phys_x = c.x0 + (b + 0.5) * (c.w / c.k_w)            # (k_w,)
        gh, gw = torch.meshgrid(phys_y, phys_x, indexing='ij')
        h_coords.append(gh.reshape(-1))
        w_coords.append(gw.reshape(-1))
    grid = torch.stack([torch.cat(w_coords), torch.cat(h_coords)])
    return grid.to(device=device, dtype=dtype)


def build_refinement_quadtree(
    H_fr: int, W_fr: int, base_k: int, refine: list[tuple[int, int, int, int, int]],
) -> Quadtree:
    """Build a quadtree: a uniform `base_k`-density base, with refine windows.

    The base frame is partitioned into a regular grid of `base_k`-token cells.
    Each `refine` entry (y0, x0, h, w, k) overrides an *axis-aligned, tile-aligned*
    rectangular window with a single higher-resolution cell of token grid
    (k_h, k_w) derived from k. The window must align to the base cell grid so the
    remaining base cells still tile the frame; refine windows must not overlap.

    This is intentionally simple — enough to drive demos ("full image coarse,
    these boxes sharp"). Arbitrary quadtrees can be built by constructing `Cell`s
    directly.

    Args:
        H_fr, W_fr: full-res token grid size.
        base_k:     base cells are (H_fr/base_k_rows ...) — here each base cell is
                    one token-block of size (cell_h, cell_w) carrying base_k**0...
                    Concretely the base is a grid of cells each of window size
                    (bh, bw) with k = the matching coarse token count; see below.
        refine:     list of (y0, x0, h, w, k) refine windows in full-res units.

    Returns:
        A validated Quadtree.
    """
    # Base: tile the frame with cells of window extent (bh, bw). For simplicity
    # the base uses one cell of size (base_k, base_k) tokens per (cell_h, cell_w)
    # window, where (cell_h, cell_w) evenly divide the frame.
    assert H_fr % base_k == 0 and W_fr % base_k == 0, \
        "base_k must divide the frame for the simple builder"
    bh, bw = H_fr // base_k, W_fr // base_k     # window extent of one base token

    # Mark which full-res pixels are claimed by a refine window.
    claimed = torch.zeros(H_fr, W_fr, dtype=torch.bool)
    cells: list[Cell] = []
    for (y0, x0, h, w, k) in refine:
        assert y0 % bh == 0 and x0 % bw == 0 and h % bh == 0 and w % bw == 0, \
            f"refine window {(y0, x0, h, w)} must align to base cells ({bh}x{bw})"
        assert not claimed[y0:y0 + h, x0:x0 + w].any(), "refine windows overlap"
        claimed[y0:y0 + h, x0:x0 + w] = True
        # k is the *token* density across the window's longer mapping; derive
        # per-axis token counts proportional to the window extent.
        k_h = max(1, round(k * h / max(h, w)))
        k_w = max(1, round(k * w / max(h, w)))
        cells.append(Cell(k_h=k_h, k_w=k_w, y0=y0, x0=x0, h=h, w=w))

    # Fill the unclaimed remainder with one base token per (bh, bw) block.
    for iy in range(base_k):
        for ix in range(base_k):
            y0, x0 = iy * bh, ix * bw
            if claimed[y0:y0 + bh, x0:x0 + bw].any():
                continue
            cells.append(Cell(k_h=1, k_w=1, y0=y0, x0=x0, h=bh, w=bw))

    qt = Quadtree(H_fr=H_fr, W_fr=W_fr, cells=cells)
    qt.validate()
    return qt
