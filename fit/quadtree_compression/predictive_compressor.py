"""Learned, variance-guided compression of latents into packed token sequences.

This is the *preprocessing step* between the frozen variance predictor (which
decides each image's quadtree structure) and the diffusion transformer (which
consumes one packed sequence per step).

Why it exists
-------------
The old path mean-pooled each quadtree leaf (`compress_from_variance`) and fed
the pooled values to a plain linear embed. That throws away all sub-leaf detail
*before* any learned layer sees it, so no downstream "learned downsampling" can
recover it — a size-8 leaf arrives as a single per-channel number. Here the
learned :class:`PyramidEncoder` runs on the FULL-RESOLUTION latent and produces
the level-l token from the real 2^l x 2^l content, so compression is learned
rather than fixed averaging.

The gradient-alignment problem
------------------------------
The encoder is trained, so its forward must sit inside the same backward pass as
the diffusion model's. That means the set of images the encoder runs on has to be
exactly the set of images packed into this step's sequence — otherwise
`zero_grad` would discard encoder gradients for images whose loss has not been
computed yet.

This is achievable because the variance predictor is FROZEN: its forward can run
far ahead, under `no_grad`, on whatever batch size is convenient. From its output
we know every image's exact token count *before* compressing anything. The data
module therefore pools *plans* (structure + token count, no values) and selects
the image subset whose counts sum just under the budget; only then does this
module compress exactly that subset, with grad. The compressor's batch size is
whatever the packing arithmetic produced — 41, 37, whatever — and changes every
step. That is fine: it is a plain batched conv forward.

    frozen predictor  ->  plan pool  ->  select images for THIS sequence
                                              |
                                              v
                              PredictiveVarianceCompressor  (trained)
                                              |
                                              v
                                   packed sequence -> QuadtreeFiT
"""

import math

import torch
import torch.nn as nn

from fit.quadtree_compression.adaptive_patch_pyramid import (
    PyramidEncoder, PyramidDecoder, gather_tokens)
from fit.quadtree_compression.quadtree_compression import (
    LEAF_SIZES, plan_to_masks)

# LEAF_SIZES == (1, 2, 4, 8) -> encoder levels 0..3. A leaf of side N == 2**l is
# one cell of the level-l grid (H / 2^(l+1) per side), so the ladders coincide.
N_LEVELS = len(LEAF_SIZES)


class PredictiveVarianceCompressor(nn.Module):
    """Compress full-resolution latents into packed quadtree token sequences.

    Holds the trained :class:`PyramidEncoder` (and optionally the matching
    :class:`PyramidDecoder`). The frozen variance predictor is NOT held here — it
    lives with the data module, which owns the pooling/packing arithmetic that
    decides which images this module is handed.

    forward(x_t, plans) -> packed sequence dict, ready for QuadtreeFiT.
    """

    def __init__(self, latent_channels=4, c=256, d=1152, crop=32,
                 pad_to_multiple=128, with_decoder=False, c_head=None,
                 t_dim=None, share_weights=False):
        super().__init__()
        self.latent_channels = latent_channels
        self.crop = crop
        self.pad_to_multiple = pad_to_multiple
        self.d = d

        self.encoder = PyramidEncoder(latent_channels, c, d, n_levels=N_LEVELS,
                                      share_weights=share_weights)
        self.decoder = None
        if with_decoder:
            # zero_init_out=False is REQUIRED here: this decoder sits mid-graph,
            # between the transformer and the loss, so a zero output layer would
            # starve the transformer AND the encoder of gradient entirely.
            self.decoder = PyramidDecoder(latent_channels, c, d,
                                          n_levels=N_LEVELS, c_head=c_head,
                                          t_dim=t_dim, share_weights=share_weights,
                                          zero_init_out=False)

    # ------------------------------------------------------------------ #
    # encode                                                              #
    # ------------------------------------------------------------------ #
    def _batch_masks(self, plans, device):
        """Stack per-image plans into batch-wide masks + the packing permutation.

        `plan_to_masks` yields (1, H_l, W_l) masks and a per-image permutation
        from level-major back to recursion order. Concatenating the masks along
        the batch axis is valid — they are plain boolean grids of fixed size, and
        both `gather_tokens` and `scatter_tokens` are documented to index
        (B, H, W) row-major. That lets the whole batch go through the encoder /
        decoder in ONE call instead of B calls at batch size 1.

        The subtlety is ordering. Boolean-mask indexing over a batched mask is
        IMAGE-major within each level (image 0's level-l tokens, then image 1's,
        ...), while the packed sequence is per-image contiguous and level-major
        inside each image. So the permutation has to be built across the whole
        batch, not per image.

        Returns (masks, counts, perm) where:
            masks  list of n_levels (B, H_l, W_l) bool
            counts (B,) python ints, tokens per image
            perm   (N_total,) long, batch-level-major -> packed order, i.e.
                   tokens_packed = tokens_levelmajor[perm]
        """
        per_image, counts = [], []
        for plan in plans:
            m, inv = plan_to_masks(
                plan['levels'], plan['positions'], plan['sizes'],
                self.crop, n_levels=N_LEVELS)
            per_image.append((m, inv))
            counts.append(int(plan['sizes'].shape[0]))

        masks = [torch.cat([m[l] for m, _ in per_image], 0)
                 for l in range(N_LEVELS)]

        # Where each (image, level) block starts in batch-level-major order, and
        # where each image starts in packed order.
        per_level = [[int(m[l].sum()) for m, _ in per_image] for l in range(N_LEVELS)]
        level_base, acc = [], 0
        for l in range(N_LEVELS):
            level_base.append(acc)
            acc += sum(per_level[l])

        perm = torch.empty(acc, dtype=torch.long, device=device)
        img_base = 0
        for i, (_, inv) in enumerate(per_image):
            # positions of image i's level-major tokens inside the batch tensor
            src = torch.cat([
                torch.arange(per_level[l][i], device=device)
                + level_base[l] + sum(per_level[l][:i])
                for l in range(N_LEVELS)])
            # inv is a GATHER: tokens_recursion == tokens_levelmajor[inv],
            # so composing it with `src` gives packed -> batch-level-major.
            perm[img_base:img_base + counts[i]] = src[inv]
            img_base += counts[i]
        return masks, counts, perm

    def encode_batch(self, x_t, plans):
        """Encode a batch of latents, each on its own quadtree plan.

        x_t   : (B, C, H, W) full-resolution noisy latents — NOT pooled.
        plans : list of B dicts with keys `levels`, `positions`, `sizes` (as
                produced by `plan_from_variance`), one per image.

        Returns (tokens, counts) where tokens is (N_total, d) already in packed
        order (image 0's tokens in recursion order, then image 1's, ...) and
        counts is the per-image token count.

        The dense pyramid is computed ONCE for the whole batch (cost ~1.33x the
        finest level) and the gather is a single batched boolean index per level,
        so nothing here loops over images on the GPU.
        """
        feats = self.encoder.features(x_t)          # list of (B, c, H_l, W_l)
        masks, counts, perm = self._batch_masks(plans, x_t.device)

        # No skipping of empty levels here: `perm` is indexed against the full
        # level-0..L-1 concatenation, and an empty level contributes a length-0
        # block that cat handles fine.
        toks = [self.encoder.proj(gather_tokens(feats[l], masks[l]))
                + self.encoder.scale_emb[l] for l in range(N_LEVELS)]
        tokens = torch.cat(toks, 0)[perm]           # -> packed order
        return tokens, counts

    # ------------------------------------------------------------------ #
    # pack                                                                #
    # ------------------------------------------------------------------ #
    def forward(self, x_t, plans, labels, t, x0=None):
        """Compress + pack one training step's images into a single sequence.

        x_t    : (B, C, H, W) noisy latents for exactly the images that belong in
                 this sequence (the data module already did that selection).
        plans  : list of B plan dicts (levels / positions / sizes).
        labels : (B,) class labels.
        t      : (B,) flow-matching timesteps.
        x0     : (B, C, H, W) clean latents, kept for the full-resolution loss.

        Returns the packed dict consumed by QuadtreeFiT (see the data module's
        docstring for field semantics). `feature` carries grad — this is the
        whole point.
        """
        device = x_t.device
        B = x_t.shape[0]
        tokens, counts = self.encode_batch(x_t, plans)
        raw_len = sum(counts)
        N = int(math.ceil(raw_len / self.pad_to_multiple) * self.pad_to_multiple)
        N = max(N, self.pad_to_multiple)

        feat = x_t.new_zeros(1, N, self.d)
        grid = torch.zeros(1, 2, N, device=device)
        tsize = torch.zeros(1, N, dtype=torch.long, device=device)
        mask = torch.zeros(1, N, dtype=torch.uint8, device=device)
        doc = torch.full((1, N), -1, dtype=torch.int32, device=device)
        size = torch.full((1, B, 2), float(self.crop // 2), device=device)

        # `tokens` is already in packed order, so the whole prefix goes in at once
        # — no per-image slice-assign, and one autograd node instead of B.
        feat[0, :raw_len] = tokens
        # positions are patch centers in latent px; /2 -> patch units, where two
        # adjacent size-1 tokens are 1 apart and a size-N token spans N.
        grid[0, :, :raw_len] = torch.cat(
            [p['positions'].to(device) for p in plans], 0).transpose(0, 1) / 2.0
        tsize[0, :raw_len] = torch.cat([p['sizes'].to(device) for p in plans], 0)
        mask[0, :raw_len] = 1
        doc[0, :raw_len] = torch.repeat_interleave(
            torch.arange(B, dtype=torch.int32, device=device),
            torch.tensor(counts, device=device))

        label = torch.as_tensor(labels, device=device).to(torch.int64).reshape(1, B)
        tvec = torch.as_tensor(t, device=device).to(torch.float32).reshape(1, B)

        out = dict(
            feature=feat, grid=grid, tsize=tsize, mask=mask, doc_ids=doc,
            label=label, t=tvec, size=size,
            n_pack=torch.tensor([B], dtype=torch.int32, device=device),
            counts=counts,
        )
        if x0 is not None:
            out['x0'] = x0
            out['x_t_full'] = x_t
        return out

    # ------------------------------------------------------------------ #
    # decode                                                              #
    # ------------------------------------------------------------------ #
    def decode_packed(self, tokens, plans, counts, z, t_emb=None):
        """Reconstruct dense latents from packed model output tokens.

        tokens : (1, N, d) model output (padding included; sliced off here).
        plans  : the same list of plan dicts used to encode.
        counts : per-image token counts (from `forward`).
        z      : (B, C, H, W) noisy latents, for the decoder's long skip.

        Returns x_hat (B, C, H, W).

        ONE decoder call for the whole batch. The decoder is pure dense conv work
        (scatter, Split, Blend, head), so running it B times at batch size 1 —
        as this used to — wasted the GPU on B x n_layers tiny kernel launches and
        built B separate autograd graphs. Batching the masks is legitimate:
        scatter_tokens already indexes (B, H, W), and the quadtree guarantees the
        levels stay disjoint and complete per image, so nothing leaks across the
        batch axis. The blend convs are spatial-only and never cross images.
        """
        assert self.decoder is not None, "compressor built with with_decoder=False"
        masks, _, perm = self._batch_masks(plans, tokens.device)
        raw_len = sum(counts)
        # perm gathers level-major -> packed; invert it to feed the decoder the
        # level-major order its split-by-level expects.
        inv = torch.empty_like(perm)
        inv[perm] = torch.arange(raw_len, device=tokens.device)
        return self.decoder(tokens[0, :raw_len][inv], masks, z, t_emb)
