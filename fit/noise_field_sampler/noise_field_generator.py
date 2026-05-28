from functools import cmp_to_key

import torch


def farey_fractions_for_denominators(denoms: list[int]) -> torch.Tensor:
    """
    Sorted unique reduced fractions p/q in [0, 1] with q in `denoms`.

    For each denominator q we form (p, q) for p = 0..q vectorized, keep those
    with gcd(p, q) == 1, and add them to a set of (p, q) tuples — the set
    handles dedup implicitly (e.g. 1/2 from q=2 is the same fraction across
    any larger denominator that would reduce to it, but we only ever insert
    reduced forms).

    Returns:
        LongTensor of shape (M, 2) with rows (p, q), sorted by p/q ascending.
    """
    fracs: set[tuple[int, int]] = set()
    for q in denoms:
        p = torch.arange(0, q + 1, dtype=torch.long)
        keep = torch.gcd(p, torch.full_like(p, q)) == 1
        for pi in p[keep].tolist():
            fracs.add((pi, q))

    # Sort by p/q ascending using cross-multiplication (exact, no float)
    def cmp(a: tuple[int, int], b: tuple[int, int]) -> int:
        # a = (p1, q1), b = (p2, q2); compare p1/q1 vs p2/q2 via p1*q2 - p2*q1
        return a[0] * b[1] - b[0] * a[1]

    sorted_fracs = sorted(fracs, key=cmp_to_key(cmp))
    return torch.tensor(sorted_fracs, dtype=torch.long)


def build_atomic_intervals(sizes: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Atomic intervals over the union of bin boundaries for the requested sizes.

    Bin boundaries at resolution k are the fractions i/k for i=0..k, so the
    relevant fractions are reduced p/q with q dividing some k in `sizes`.
    Consecutive pairs of these sorted fractions form atomic intervals; on each
    interval all bin assignments (p*k)//q are constant for every k in `sizes`.

    Returns:
        intervals : LongTensor of shape (J, 4) with rows (p1, q1, p2, q2)
        widths    : float32 tensor of shape (J,) with widths p2/q2 - p1/q1
    """
    # Collect all divisors of any requested size
    divisors = set()
    for k in sizes:
        for q in range(1, k + 1):
            if k % q == 0:
                divisors.add(q)
    denoms = sorted(divisors)

    fracs = farey_fractions_for_denominators(denoms)  # (M, 2), sorted
    p, q = fracs[:, 0], fracs[:, 1]

    intervals = torch.stack([p[:-1], q[:-1], p[1:], q[1:]], dim=1)  # (J, 4)
    # widths = p2/q2 - p1/q1 = (p2*q1 - p1*q2) / (q1*q2)
    num = intervals[:, 2] * intervals[:, 1] - intervals[:, 0] * intervals[:, 3]
    den = intervals[:, 1] * intervals[:, 3]
    widths = (num.to(torch.float64) / den.to(torch.float64)).to(torch.float32)
    return intervals, widths


def build_bin_indices(sizes: list[int], intervals: torch.Tensor) -> torch.Tensor:
    """
    Precompute bin assignments for each requested resolution.

    Returns:
        bin_idx : LongTensor of shape (len(sizes), J) where bin_idx[i, j] is
                  the bin index of atomic interval j at resolution sizes[i].
    """
    p1 = intervals[:, 0]  # (J,)
    q1 = intervals[:, 1]
    ks = torch.tensor(sizes, dtype=torch.long).unsqueeze(1)  # (S, 1)
    return (p1.unsqueeze(0) * ks) // q1.unsqueeze(0)  # (S, J)


def sample_noise_fields_2d(sizes: list[int], d: int, b: int, chunk: int = 512) -> list[torch.Tensor]:
    """
    Sample a consistent family of 2D noise fields at the given resolutions.

    Atomic intervals are built from the union of bin boundaries of the
    requested sizes — i.e. reduced fractions p/q where q divides some k in
    `sizes`. This guarantees that on every atomic interval each (p*k)//q is
    constant, giving cross-resolution sum-consistency.

    Each returned field has unit per-pixel std: the raw scatter_add gives a
    pixel at resolution k variance 1/k² (sum of atomic widths in its bin), so
    we multiply the resolution-k field by k to rescale to per-pixel std 1.
    Sum-consistency across resolutions still holds when corrected for this
    rescaling: avg(fine 2×2 block) == coarse / 2 for a 2× resolution jump.

    Uses streaming over chunks of 2D atomic interval pairs (jx, jy) so that
    memory stays at O(chunk² · b · d + Σ k²) rather than O(J² · b · d).
    Each 2D atomic rectangle has variance  w_jx · w_jy  and is assigned to
    bin (ix, iy) at resolution k via the 1D bin indices.

    Args:
        sizes : list of requested resolutions (each ≥ 1)
        d     : field dimensionality (channels)
        b     : batch size
        chunk : number of 2D atomic pairs to process at once

    Returns:
        List of len(sizes) tensors. Entry i has shape (b, d, sizes[i], sizes[i])
        with per-pixel std 1.
    """
    intervals, widths = build_atomic_intervals(sizes)
    J = intervals.shape[0]
    bin_idx = build_bin_indices(sizes, intervals)  # (S, J)

    # Pre-allocate output fields in float64 to keep the J² accumulation well-conditioned
    fields = [torch.zeros(b, d, k, k, dtype=torch.float64) for k in sizes]
    widths64 = widths.to(torch.float64)

    # Stream over chunks of (jx, jy) pairs
    for jx_start in range(0, J, chunk):
        jx_end = min(jx_start + chunk, J)
        chunk_x = jx_end - jx_start
        w_x = widths64[jx_start:jx_end]                      # (chunk_x,)
        bin_x = bin_idx[:, jx_start:jx_end]                   # (S, chunk_x)

        for jy_start in range(0, J, chunk):
            jy_end = min(jy_start + chunk, J)
            chunk_y = jy_end - jy_start
            w_y = widths64[jy_start:jy_end]                   # (chunk_y,)
            bin_y = bin_idx[:, jy_start:jy_end]                # (S, chunk_y)

            # 2D atomic variances: w_jx * w_jy, shape (chunk_x, chunk_y)
            w_2d = w_x[:, None] * w_y[None, :]                # (cx, cy)

            # Sample independent noise for this chunk in float64: (b, d, cx, cy)
            noise = torch.randn(b, d, chunk_x, chunk_y, dtype=torch.float64) * w_2d.sqrt()

            # Scatter into each requested resolution
            for out_i, k in enumerate(sizes):
                bx = bin_x[out_i]   # (chunk_x,)  values in [0, k)
                by = bin_y[out_i]   # (chunk_y,)  values in [0, k)

                # Flat 2D bin index: ix * k + iy, shape (cx, cy)
                flat_idx = bx[:, None] * k + by[None, :]      # (cx, cy)

                # Flatten spatial dims and scatter_add into (b, d, k*k)
                fields[out_i].view(b, d, -1).scatter_add_(
                    2,
                    flat_idx.reshape(1, 1, -1).expand(b, d, -1),
                    noise.reshape(b, d, -1),
                )

    # Rescale each resolution to per-pixel std 1.
    return [(f * k).to(torch.float32) for f, k in zip(fields, sizes)]


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)
    d, b = 4, 6

    # 2D consistency test
    sizes_2d = [2, 4, 8]
    print(f"\n2D noise field test (sizes={sizes_2d}):")
    fields_2d = sample_noise_fields_2d(sizes_2d, d, b)
    by_size = {k: f for k, f in zip(sizes_2d, fields_2d)}
    for k, f in by_size.items():
        assert f.shape == (b, d, k, k), f"Wrong 2D shape at k={k}: {f.shape}"
    print("  Shape check passed.")

    # Check consistency in 2D: with per-pixel std 1, the fine 2×2 average
    # (sum/4) equals coarse/2 — i.e. 2 * mean_2x2(fine) == coarse.
    print("  Power-of-2 consistency (2D, unit-std fields):")
    for k in sizes_2d:
        if 2 * k not in by_size:
            continue
        coarse = by_size[k]           # (b, d, k, k)
        fine   = by_size[2 * k]       # (b, d, 2k, 2k)
        fine_sum = (fine[..., 0::2, 0::2] + fine[..., 0::2, 1::2] +
                    fine[..., 1::2, 0::2] + fine[..., 1::2, 1::2])
        ok = torch.allclose(fine_sum / 2, coarse, atol=1e-5)
        mark = "✓" if ok else "✗"
        print(f"    {mark}  fields_2d[{k}] == sum_2x2(fields_2d[{2*k}]) / 2")

    # Report raw noise vector sizes for typical input resolutions
    print("\nRaw atomic noise vector sizes (J) for typical input sizes:")
    for s in [64, 128, 256]:
        J = build_atomic_intervals([s])[0].shape[0]
        print(f"  n={s:3d}  =>  J={J:,d}  (1D, √J={J**.5:.1f})    J²={J**2:,d}  (2D, √J²={J:,d})")