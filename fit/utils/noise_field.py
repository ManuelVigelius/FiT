import torch

def farey_sequence(n: int) -> list[tuple[int, int]]:
    """Farey sequence F_n as (numerator, denominator) pairs, including 0/1 and 1/1."""
    fracs = [(0, 1)]
    a, b, c, d = 0, 1, 1, n
    while c <= n:
        fracs.append((c, d))
        k = (n + b) // d
        a, b, c, d = c, d, k * c - a, k * d - b
    return fracs


def build_atomic_intervals(n: int) -> tuple[list[tuple[int, int, int, int]], torch.Tensor]:
    """
    Atomic intervals from F_n.  Each consecutive Farey pair (p1/q1, p2/q2)
    spans width 1/(q1*q2) by the Farey neighbour property.

    Returns:
        intervals : list of (p1, q1, p2, q2)
        widths    : float32 tensor of shape (J,)
    """
    fracs = farey_sequence(n)
    intervals, widths = [], []
    for (p1, q1), (p2, q2) in zip(fracs, fracs[1:]):
        intervals.append((p1, q1, p2, q2))
        widths.append(1.0 / (q1 * q2))
    return intervals, torch.tensor(widths, dtype=torch.float32)

def build_bin_indices(n: int, intervals: list) -> torch.Tensor:
    """
    Precompute bin assignments for all resolutions.

    Returns:
        bin_idx : LongTensor of shape (n, J) where bin_idx[k-1, j] is the
                  bin index of atomic interval j at resolution k.
    """
    J = len(intervals)
    bin_idx = torch.empty(n, J, dtype=torch.long)
    for j, (p1, q1, *_) in enumerate(intervals):
        for k in range(1, n + 1):
            bin_idx[k - 1, j] = (p1 * k) // q1
    return bin_idx


def sample_noise_pair_2d(
    size_a: int,
    size_b: int,
    d: int,
    b: int,
    chunk: int = 512,
) -> dict[int, torch.Tensor]:
    """
    Sample consistent 2D noise fields for exactly two resolutions.

    Uses the shared Farey basis for n = max(size_a, size_b) but only
    materialises output tensors for the two requested sizes.

    Args:
        size_a : first spatial resolution
        size_b : second spatial resolution
        d      : number of channels
        b      : batch size
        chunk  : atomic-interval chunk size for streaming

    Returns:
        {size_a: tensor of shape (b, d, size_a, size_a),
         size_b: tensor of shape (b, d, size_b, size_b)}
    """
    n = max(size_a, size_b)
    intervals, widths = build_atomic_intervals(n)
    J = len(intervals)
    bin_idx = build_bin_indices(n, intervals)  # (n, J)

    sizes = [size_a, size_b]
    fields = {s: torch.zeros(b, d, s, s) for s in sizes}

    for jx_start in range(0, J, chunk):
        jx_end = min(jx_start + chunk, J)
        chunk_x = jx_end - jx_start
        w_x = widths[jx_start:jx_end]
        bin_x = bin_idx[:, jx_start:jx_end]  # (n, chunk_x)

        for jy_start in range(0, J, chunk):
            jy_end = min(jy_start + chunk, J)
            chunk_y = jy_end - jy_start
            w_y = widths[jy_start:jy_end]
            bin_y = bin_idx[:, jy_start:jy_end]  # (n, chunk_y)

            w_2d = w_x[:, None] * w_y[None, :]  # (cx, cy)
            noise = torch.randn(b, d, chunk_x, chunk_y) * w_2d.sqrt()
            noise_flat = noise.reshape(b, d, -1)

            for s in sizes:
                bx = bin_x[s - 1]  # (chunk_x,)
                by = bin_y[s - 1]  # (chunk_y,)
                flat_idx = bx[:, None] * s + by[None, :]  # (cx, cy)
                fields[s].view(b, d, -1).scatter_add_(
                    2,
                    flat_idx.reshape(1, 1, -1).expand(b, d, -1),
                    noise_flat,
                )

    return fields


if __name__ == "__main__":
    torch.manual_seed(0)
    result = sample_noise_pair_2d(4, 8, d=2, b=3)
    for s, f in result.items():
        print(f"size={s}  shape={tuple(f.shape)}")

    # Sum-consistency: field[4] == sum of 2x2 blocks in field[8]
    f4, f8 = result[4], result[8]
    reconstructed = (
        f8[..., 0::2, 0::2] + f8[..., 0::2, 1::2]
        + f8[..., 1::2, 0::2] + f8[..., 1::2, 1::2]
    )
    ok = torch.allclose(reconstructed, f4, atol=1e-5)
    print("Consistency check:", "passed" if ok else "FAILED")
