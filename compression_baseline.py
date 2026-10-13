"""Measure what the quadtree compression alone costs, with no diffusion.

Takes real training latents, builds the GT-variance quadtree for each, runs it
through the compressor's encode -> decode round trip, and writes one image per
sample: original | reconstruction | reconstruction with the tree drawn.

NO PYRAMID, NO LEARNED WEIGHTS
------------------------------
The compressor is built `pyramid_free`, so encoding is pure arithmetic: a
size-N leaf stores the 2x2 QUADRANT MEANS of its 2N x 2N latent footprint, and
`scatter_baseline` expands each mean back over its N x N block. Nothing is
learned, nothing is loaded, and the result is exactly the information the
tokens carry -- no more.

That is the point. With the pyramid decoder in the loop the reconstruction is
a LEARNED function that can hallucinate detail the tokens never encoded, so a
clean-looking output proves nothing about the representation. Dropping it makes
this an honest upper bound on quality: whatever detail is missing here is
detail the transformer never receives.

One consequence worth knowing: a size-1 leaf covers 2x2 latent pixels and
stores 2x2 quadrant means, so it is the IDENTITY -- lossless. Compression
happens only where the tree coarsens, and the tree coarsens only where variance
< threshold, i.e. where there was no detail to lose. At --cap-t 0 the round
trip is therefore near-lossless by construction and will look uncompressed.
Use --cap-t 0.9 (min_leaf 4, a 16x ceiling) to see real artifacts.

Only the GT tree
----------------
The tree comes from the TRUE variance of the latent, capped by
max_res_schedule -- the oracle structure, the best the tree can do for this
image. A prior-sampled tree is not comparable here: it is drawn from the class
label alone and never sees the image, so its fine leaves land where *a* member
of that class tends to have detail, not where THIS image does. The error would
measure spatial misalignment and sampling noise rather than compression. The
prior remains the right source for generation, where no clean latent exists.

Usage:
    python compression_baseline.py --n-images 16
    python compression_baseline.py --n-images 16 --cap-t 0.9
"""

import argparse
import json
import os

import torch
import numpy as np
from PIL import Image
from diffusers.models import AutoencoderKL

from fit.data.in1k_gt_quadtree_latent_dataset import (
    gt_variance_grids, plan_from_gt_variance, min_leaf_size_at,
    normalize_max_res_schedule)
from fit.data.in1k_quadtree_latent_dataset import load_latent

from fit.quadtree_compression.predictive_compressor import (
    PredictiveVarianceCompressor)

from generate_quadtree_images import (
    LATENT_SIZE, C_IN, PATCH_SIZE, PAD_TO_MULTIPLE, VAE_PATH, DATASET_ROOT,
    QUADTREE_THRESHOLD, MAX_RES_SCHEDULE, DENSE_TOKENS,
    pack, scatter_baseline, draw_quadtree,
    _discover_latents, _center_crop, _to_images)


def round_trip(comp, x0, plans, device):
    """Encode x0 onto `plans` and scatter straight back to dense.

    `encode_batch` emits tokens in packed order; `scatter_baseline` consumes
    that same order. In pyramid_free mode `tokens_at_level` returns the
    mean-pooled patch values with no projection, so this whole path is
    deterministic arithmetic -- no parameters are involved in either direction.

    Labels and t are required by `pack` but unused: no transformer runs here,
    and the encoder is unconditional.
    """
    B = x0.shape[0]
    packed = pack(comp, x0, plans,
                  torch.zeros(B, dtype=torch.long, device=device),
                  torch.zeros(B, device=device), device)
    return scatter_baseline(packed["feature"], plans, packed["counts"], device)


def _triptych(orig, rec, plan, gap=4, bg=255):
    """original | reconstruction | reconstruction with the quadtree drawn."""
    panels = [orig, rec, draw_quadtree(rec, plan)]
    h, w, _ = orig.shape
    out = np.full((h, w * 3 + gap * 2, 3), bg, dtype=np.uint8)
    for j, p in enumerate(panels):
        x = j * (w + gap)
        out[:, x:x + w] = p
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-images", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--cap-t", type=float, default=0.0,
                    help="timestep at which max_res_schedule is evaluated")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="/visinf/home/mb_mvigel/compression_baseline")
    ap.add_argument("--dataset", default=DATASET_ROOT)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    schedule = normalize_max_res_schedule(MAX_RES_SCHEDULE)
    min_leaf = min_leaf_size_at(args.cap_t, schedule)
    print(f"Device: {device}  pyramid_free (no learned weights)")
    print(f"cap t={args.cap_t} -> min_leaf={min_leaf} "
          f"(max {(LATENT_SIZE // 2 // min_leaf) ** 2} tokens)")

    vae = AutoencoderKL.from_pretrained(VAE_PATH).to(device).eval()

    # pyramid_free: zero parameters, so there is no checkpoint to load. The
    # token width is 4*C -- the 2x2 quadrant means the encoder emits directly.
    comp = PredictiveVarianceCompressor(
        latent_channels=C_IN, c=256, d=4 * C_IN,
        crop=LATENT_SIZE, pad_to_multiple=PAD_TO_MULTIPLE,
        with_decoder=False, share_weights=True,
        residual_encoder=False, residual_decoder=False,
        pyramid_free=True, patch_size=PATCH_SIZE, load_base_head=False,
    ).to(device).eval()
    n_par = sum(p.numel() for p in comp.parameters())
    if n_par:
        raise RuntimeError(
            f"expected a parameter-free compressor, got {n_par} parameters")

    files = _discover_latents(args.dataset)
    g = torch.Generator(device="cpu").manual_seed(args.seed)
    idx = torch.randint(0, len(files), (args.n_images,), generator=g).tolist()
    print(f"  {len(files)} train latents, sampling {args.n_images}")

    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)

    tok_counts, errs = [], []
    done = 0
    while done < args.n_images:
        bs = min(args.batch_size, args.n_images - done)
        sel = idx[done:done + bs]
        x0 = torch.stack([
            _center_crop(load_latent(files[j])).float() for j in sel
        ]).to(device)
        plans = [dict(zip(("levels", "positions", "sizes"),
                          plan_from_gt_variance(
                              gt_variance_grids(x0[i]),
                              QUADTREE_THRESHOLD, LATENT_SIZE,
                              min_leaf_size=min_leaf)))
                 for i in range(bs)]

        with torch.no_grad():
            rec = round_trip(comp, x0, plans, device)
            orig_img = _to_images(vae, x0)
            rec_img = _to_images(vae, rec)

        # Latent-space error is the clean signal; pixel error also folds in the
        # VAE, which is the same for both and not what we are measuring.
        per_img = (rec - x0).pow(2).mean(dim=(1, 2, 3)).sqrt()
        errs.extend(per_img.tolist())
        tok_counts.extend(int(p["sizes"].shape[0]) for p in plans)

        for k in range(bs):
            i = done + k
            Image.fromarray(
                _triptych(orig_img[k], rec_img[k], plans[k])).save(
                os.path.join(out_dir, f"{i:06d}.png"))

        done += bs
        print(f"  {done}/{args.n_images}", flush=True)

    mt = sum(tok_counts) / len(tok_counts)
    me = sum(errs) / len(errs)
    print(f"\n  mean tokens {mt:.1f} / {DENSE_TOKENS} dense "
          f"= {DENSE_TOKENS / mt:.2f}x compression")
    print(f"  mean latent RMSE {me:.4f}")

    with open(os.path.join(args.out, "stats.json"), "w") as f:
        json.dump(dict(cap_t=args.cap_t,
                       min_leaf=min_leaf, mean_tokens=mt, mean_rmse=me,
                       tokens=tok_counts, rmse=errs), f, indent=2)

    print(f"\nDone. Output under {args.out}")


if __name__ == "__main__":
    main()
