"""
Adaptive-patch pyramid tokenizer / detokenizer for a diffusion transformer.

Operates in VAE latent space. Patch sizes are {2, 4, 8} latent cells, indexed by
level l in {0, 1, 2}. Level l lives on a grid of shape (H / 2^(l+1), W / 2^(l+1)).

Design commitments:
  - Encoder merges are strictly within-patch: pixel_unshuffle + channel MLP only.
    No spatial kernel ever crosses a patch boundary on the way in.
  - Merge / split weights are SHARED across levels. A level-2 token is
    merge(merge(stem(z))) -- the same function composed. Every patch size trains
    the same parameters, which keeps the token space scale-consistent.
  - Decoder is the synthesis side of a Laplacian pyramid, not a U-Net. Coarse
    tokens are placed on a sparse grid; each upsampling step fills the holes with
    finer tokens. The quadtree guarantees the levels are disjoint and complete.
  - Both encoder and decoder support a RESIDUAL mode in which the learned pyramid
    only ever adds a zero-gated correction on top of an ordinary patch embed /
    per-token readout. See PyramidEncoder.tokens_at_level and
    PyramidDecoder.base_predict. In that mode, at zero compression the pair
    reduces exactly to plain FiT's patchify -> linear -> unpatchify.
  - Decoder blend blocks are ConvNeXt-style (depthwise + inverted bottleneck) and
    DO cross patch boundaries. That is what removes seams.
  - Everything is dense. The level-2 grid is stored full-size even when mostly
    empty; it is tiny. There are no sparse ops and no ragged batching here.

Token ordering convention: boolean-mask indexing over (B, H, W) in row-major
order. gather_tokens and scatter_tokens use the same convention, so a round trip
is order-preserving. The caller is responsible for sequence packing, RoPE
positions from patch centres, and the block-diagonal attention mask.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------- primitives

class ChannelsLastLN(nn.Module):
    """LayerNorm over the channel axis of an (B, C, H, W) tensor."""

    def __init__(self, c, eps=1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(c, eps=eps)

    def forward(self, x):
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


def _patchify_2x2(z):
    """(B, C, H, W) -> (B, 4*C, H/2, W/2), channels ordered (c, p1, p2).

    This is `fit.utils.utils.patchify(z, 2)` kept in spatial layout, and the
    ordering matters: three different 4*C layouts are in play in this repo.

      * `patchify`                -> (c p1 p2)   <- what the PRETRAINED
                                                    `x_embedder` was trained on
      * `F.pixel_unshuffle`       -> (c p1 p2) but with p1/p2 as the fast axes
                                     of a different reshape — NOT the same
      * `compress_from_variance`  -> (p1 p2 c)   <- the quadtree's own legacy
                                                    layout, embed trained from
                                                    scratch so it never mattered

    Residual mode exists to reuse pretrained weights, so this follows `patchify`.
    A permuted layout here would still train, just from a scrambled starting
    point — which is exactly the failure this mode is meant to avoid.
    """
    b, c, h, w = z.shape
    return (z.reshape(b, c, h // 2, 2, w // 2, 2)
             .permute(0, 1, 3, 5, 2, 4)
             .reshape(b, 4 * c, h // 2, w // 2))


def gather_tokens(feat, mask):
    """(B, C, H, W) + (B, H, W) bool  ->  (n_selected, C)."""
    return feat.permute(0, 2, 3, 1)[mask]


def scatter_tokens(tokens, mask, c):
    """(n_selected, C) + (B, H, W) bool  ->  (B, C, H, W), zeros elsewhere."""
    b, h, w = mask.shape
    out = tokens.new_zeros(b, h, w, c)
    out[mask] = tokens
    return out.permute(0, 3, 1, 2)


def icnr_(weight, upscale=2):
    """
    ICNR init for the conv feeding a pixel_shuffle: replicate one kernel across
    all r^2 sub-filters so the block starts as exact nearest-neighbour upsampling.
    Expects weight of shape (r^2 * c_out, c_in, kh, kw).
    """
    o, i, kh, kw = weight.shape
    sub = torch.empty(o // (upscale ** 2), i, kh, kw, device=weight.device)
    nn.init.kaiming_normal_(sub)
    with torch.no_grad():
        weight.copy_(sub.repeat_interleave(upscale ** 2, dim=0))


# ---------------------------------------------------------------- encoder

class Merge(nn.Module):
    """
    2x2 -> 1 merge, applied to a whole grid at once. Non-overlapping by
    construction, so it never mixes across patch boundaries.
    """

    def __init__(self, c, expansion=4):
        super().__init__()
        self.norm = ChannelsLastLN(4 * c)
        self.fc1 = nn.Conv2d(4 * c, expansion * c, 1)
        self.fc2 = nn.Conv2d(expansion * c, c, 1)

    def forward(self, x):
        x = F.pixel_unshuffle(x, 2)
        return self.fc2(F.gelu(self.fc1(self.norm(x))))


class PyramidEncoder(nn.Module):
    """
    Computes ALL pyramid levels densely over the whole latent, then gathers each
    token from the level matching its assigned patch size.

    Dense cost is only ~1.33x the finest level (geometric series), so computing
    levels you mostly discard is cheaper than any ragged alternative.
    """

    def __init__(self, latent_ch, c, d, n_levels=3, share_weights=False,
                 residual=False, pyramid_free=False):
        super().__init__()
        self.n_levels = n_levels
        self.c = c
        self.latent_ch = latent_ch
        self.residual = residual

        # ---- pyramid-free mode ----------------------------------------------
        # The no-learned-compression baseline: tokens are the raw mean-pooled
        # patches, handed straight to the model's own (pretrained) x_embedder.
        # NOTHING here is built or trained -- no stem, no merges, no proj, no
        # gate. Requires d == 4*latent_ch, since there is no projection left to
        # change the width.
        self.pyramid_free = pyramid_free
        if pyramid_free:
            if d != 4 * latent_ch:
                raise ValueError(
                    f'pyramid_free needs d == 4*latent_ch (the mean-pooled patch '
                    f'width), got d={d} and 4*latent_ch={4 * latent_ch}. There is '
                    f'no projection in this mode to reconcile the two.')
            if residual:
                raise ValueError(
                    'pyramid_free and residual are mutually exclusive: residual '
                    'adds a GATED PYRAMID on top of the base path, which is '
                    'exactly what pyramid_free removes.')
            self.stem = self.proj = self.base_proj = None
            self.merges = nn.ModuleList()
            self.scale_emb = self.res_gamma = None
            return

        self.stem = nn.Conv2d(4 * latent_ch, c, 1)
        if share_weights:
            merge = Merge(c)
            self.merges = nn.ModuleList([merge] * (n_levels - 1))
        else:
            self.merges = nn.ModuleList([Merge(c) for _ in range(n_levels - 1)])
        self.proj = nn.Linear(c, d)
        self.scale_emb = nn.Parameter(torch.zeros(n_levels, d))

        # ---- residual mode ---------------------------------------------------
        # The learned pyramid is a CORRECTION on top of the ordinary patch embed,
        # not a replacement for it. `base_proj` is the same 4*C -> d linear the
        # plain model calls `x_embedder.proj`, so a pretrained FiT checkpoint can
        # be copied straight into it (see `load_base_proj`). `res_gamma` is
        # zero-init, which makes the encoder output vanish at step 0: the model
        # then sees EXACTLY the mean-pooled baseline tokens it was pretrained on,
        # and the pyramid only ever earns its way in from there.
        if residual:
            self.base_proj = nn.Linear(4 * latent_ch, d)
            self.res_gamma = nn.Parameter(torch.zeros(n_levels, d))
        else:
            self.base_proj = None
            self.res_gamma = None

    @torch.no_grad()
    def load_base_proj(self, weight, bias=None):
        """Copy a pretrained `x_embedder.proj` into the base projection.

        weight: (d, 4*latent_ch) — the pretrained patch embed matrix.
        """
        assert self.base_proj is not None, "encoder built with residual=False"
        self.base_proj.weight.copy_(weight)
        if bias is not None:
            self.base_proj.bias.copy_(bias)

    def base_features(self, z):
        """z: (B, C, H, W) -> list of (B, 4*C, H_l, W_l) mean-pooled patch values.

        Level l holds the value a plain patchify would give if the whole image
        used patch size 2^(l+1): the 2x2 block of size-2^l leaves, flattened
        row-major to 4*C. That is byte-for-byte the layout
        `compress_from_variance` used to emit, hence what the pretrained
        `x_embedder` expects.
        """
        feats = [_patchify_2x2(z)]
        for _ in range(self.n_levels - 1):
            # avg-pool the LEAF values, then re-block into 2x2 patches. Pooling in
            # (C, H, W) space and re-blocking keeps the (p1, p2, C) row-major
            # ordering intact at every level.
            z = F.avg_pool2d(z, 2)
            feats.append(_patchify_2x2(z))
        return feats

    def features(self, z):
        """z: (B, C, H, W) -> list of (B, c, H/2^(l+1), W/2^(l+1)), fine to coarse.

        Returns a list of None in pyramid_free mode: there is no pyramid, and
        `tokens_at_level` ignores the entry.
        """
        if self.pyramid_free:
            return [None] * self.n_levels
        x = self.stem(F.pixel_unshuffle(z, 2))
        feats = [x]
        for merge in self.merges:
            x = merge(x)
            feats.append(x)
        return feats

    def tokens_at_level(self, feat, base_feat, mask, l):
        """Gather level-l tokens: base patch embed + gated pyramid residual.

        feat      : (B, c, H_l, W_l)    pyramid features for level l, or None
                                        when the pyramid is disabled entirely
        base_feat : (B, 4*C, H_l, W_l)  mean-pooled patch values, or None
        mask      : (B, H_l, W_l) bool
        """
        # `pyramid_free`: emit the mean-pooled patch and nothing else. This is
        # not the same as residual with a zero gate — that still BUILDS and
        # TRAINS the pyramid, and its gate receives gradient immediately, so the
        # pyramid stops being inert after the first optimizer step. For a
        # no-learned-compression baseline the pyramid must not exist at all.
        if self.pyramid_free:
            return gather_tokens(base_feat, mask)
        t = self.proj(gather_tokens(feat, mask)) + self.scale_emb[l]
        if self.residual:
            base = self.base_proj(gather_tokens(base_feat, mask))
            t = base + self.res_gamma[l] * t
        return t

    def forward(self, z, masks):
        """
        masks: list of (B, H_l, W_l) bool, one per level, disjoint and complete
               under the quadtree.

        Returns (tokens, index) where tokens is (N_total, d) and index is a
        (N_total,) long tensor of level ids, in level order 0, 1, 2. Keep this
        ordering -- the decoder's scatter assumes it.
        """
        feats = self.features(z)
        base = self.base_features(z) if self.residual else [None] * self.n_levels
        tokens, levels = [], []
        for l, (f, m) in enumerate(zip(feats, masks)):
            t = self.tokens_at_level(f, base[l], m, l)
            tokens.append(t)
            levels.append(torch.full((t.shape[0],), l, dtype=torch.long, device=t.device))
        return torch.cat(tokens, 0), torch.cat(levels, 0)


# ---------------------------------------------------------------- decoder

class Split(nn.Module):
    """1 -> 2x2 split via inverted bottleneck + pixel_shuffle. Mirror of Merge."""

    def __init__(self, c_in, c_out, expansion=4):
        super().__init__()
        self.norm = ChannelsLastLN(c_in)
        self.fc1 = nn.Conv2d(c_in, expansion * c_out, 1)
        self.fc2 = nn.Conv2d(expansion * c_out, 4 * c_out, 1)
        icnr_(self.fc2.weight, upscale=2)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        x = self.fc2(F.gelu(self.fc1(self.norm(x))))
        return F.pixel_shuffle(x, 2)


class Blend(nn.Module):
    """
    ConvNeXt block with adaLN-Zero timestep modulation. This is the only place a
    kernel crosses patch boundaries -- it is what blends seams between regions of
    different patch size.
    """

    def __init__(self, c, k=7, expansion=4, t_dim=None):
        super().__init__()
        self.dw = nn.Conv2d(c, c, k, padding=k // 2, groups=c)
        self.norm = ChannelsLastLN(c)
        self.fc1 = nn.Conv2d(c, expansion * c, 1)
        self.fc2 = nn.Conv2d(expansion * c, c, 1)
        self.gamma = nn.Parameter(1e-6 * torch.ones(1, c, 1, 1))
        self.mod = None
        if t_dim is not None:
            self.mod = nn.Linear(t_dim, 2 * c)
            nn.init.zeros_(self.mod.weight)
            nn.init.zeros_(self.mod.bias)

    def forward(self, x, t_emb=None):
        h = self.norm(self.dw(x))
        if self.mod is not None and t_emb is not None:
            s, b = self.mod(t_emb).chunk(2, dim=-1)
            h = h * (1 + s[:, :, None, None]) + b[:, :, None, None]
        h = self.fc2(F.gelu(self.fc1(h)))
        return x + self.gamma * h


class PyramidDecoder(nn.Module):
    """
    Scatter tokens into their own level's grid, then walk down: upsample, inject
    the finer tokens into the holes, blend. Finish at full latent resolution with
    a long skip from z_t and predict x_hat.

    The z_t skip matters more than it looks. Since eps = (z_t - alpha_t x) / sigma_t,
    a decoder that can see z_t locally can express eps or v algebraically; the
    tokens only ever need to carry x-information. At low noise x ~= z_t, so the
    head learns a near-identity map and the token just says how to correct it.
    That is what makes coarse patches survivable at low t.
    """

    def __init__(self, latent_ch, c, d, n_levels=3, c_head=None, t_dim=None,
                 share_weights=False, zero_init_out=True, residual=False,
                 patch_size=2):
        super().__init__()
        self.n_levels = n_levels
        self.c = c
        self.latent_ch = latent_ch
        self.residual = residual
        self.patch_size = patch_size
        c_head = c_head or c // 4
        self.c_head = c_head

        # ---- residual mode ---------------------------------------------------
        # Mirror of PyramidEncoder's residual path, on the way out. The BASE
        # prediction is the ordinary per-token readout — d -> patch_size^2 *
        # latent_ch, exactly plain FiT's `final_layer.linear` — reshaped to a
        # patch and repeated across the leaf's footprint. The conv stack below
        # then only supplies a zero-gated correction.
        #
        # Why an explicit repeat rather than leaning on Split's ICNR init: ICNR
        # makes Split start as nearest-neighbour upsampling SPATIALLY, but the
        # vector it replicates has already been through the random `in_proj`
        # (d -> c) and is later mapped by the random `out` (c_head ->
        # latent_ch). Measured at init, the replicated block correlates ~0.06
        # with the token it came from. ICNR replicates the wrong thing; only an
        # explicit base path makes the output start AT the token's own
        # prediction.
        if residual:
            self.base_head = nn.Linear(d, patch_size * patch_size * latent_ch)
            self.res_gamma = nn.Parameter(torch.zeros(n_levels, 1, 1, 1))
        else:
            self.base_head = None
            self.res_gamma = None

        self.in_proj = nn.Linear(d, c)
        # tells the blend convs where the patch-size boundaries are
        self.mask_proj = nn.ModuleList([nn.Conv2d(1, c, 1) for _ in range(n_levels)])

        if share_weights:
            split = Split(c, c)
            self.splits = nn.ModuleList([split] * (n_levels - 1))
        else:
            self.splits = nn.ModuleList([Split(c, c) for _ in range(n_levels - 1)])

        # k=3 at coarse levels: those grids are only a few cells across
        ks = [7 if l == 0 else 3 for l in range(n_levels)]
        self.blends = nn.ModuleList([Blend(c, k=ks[l], t_dim=t_dim)
                                     for l in range(n_levels)])

        self.head_split = Split(c, c_head)
        self.skip_proj = nn.Conv2d(c_head + latent_ch, c_head, 1)
        self.head_blend = Blend(c_head, k=7, t_dim=t_dim)
        self.out = nn.Conv2d(c_head, latent_ch, 1)
        # Zero-init makes x_hat identically zero at init — the usual DiT trick,
        # safe when this decoder terminates one contiguous graph. It is NOT safe
        # when the decoder sits between two separately-optimised modules (as in
        # PredictiveVarianceCompressor): a zero here blocks ALL gradient to the
        # transformer and the encoder upstream, not just to this layer. Pass
        # zero_init_out=False in that setting.
        # Residual mode must NOT zero `out`: the gate `res_gamma` is already
        # zero, and a product of two zero-init factors is a dead end — both
        # factors then receive exactly zero gradient and the pyramid never
        # trains at all. Exactly one of the two may be zero. Keeping `out`
        # random (and the gate zero) is the encoder's arrangement and the one
        # that bootstraps: gamma moves on step 1, which opens the path to `out`.
        if zero_init_out and not residual:
            nn.init.zeros_(self.out.weight)
            nn.init.zeros_(self.out.bias)
        else:
            nn.init.normal_(self.out.weight, std=0.02)
            nn.init.zeros_(self.out.bias)

    def base_predict(self, tokens, masks):
        """Per-token readout, NN-upsampled onto the dense latent grid.

        This is the decoder's identity-like starting point. Each token is mapped
        to one `patch_size x patch_size` patch by `base_head` (the same shape
        plain FiT's `final_layer.linear` produces) and that patch is then
        repeated across the token's whole footprint: a level-l token owns a
        2N x 2N latent region with N == 2^l, i.e. an N x N tiling of the patch.

        At level 0 (N == 1) there is no repetition at all, so with zero
        compression this reduces EXACTLY to plain FiT's readout. For a coarse
        leaf it is a blocky upsample of a single predicted patch — deliberately
        the honest baseline, since a coarse token genuinely carries only one
        patch worth of information. The pyramid's job is to add back the detail
        that blockiness is missing.

        tokens: (N_total, d) in level-major order
        masks:  list of n_levels (B, H_l, W_l) bool
        returns (B, latent_ch, H, W)
        """
        p, C = self.patch_size, self.latent_ch
        counts = [int(m.sum()) for m in masks]
        # `tokens` may already BE patches: when an external head (e.g. a
        # pretrained final_layer) has done the d -> p*p*C readout, applying
        # base_head again would project a second time. Detect that by width and
        # pass through, so the same method serves both wirings.
        flat = tokens if tokens.shape[-1] == p * p * C else self.base_head(tokens)
        per_level = list(torch.split(flat, counts, dim=0))

        B = masks[0].shape[0]
        H = masks[0].shape[1] * p                    # level-0 grid is H/p cells
        out = None
        for l, m in enumerate(masks):
            if counts[l] == 0:
                continue
            n = 2 ** l                               # patch tiling factor
            # scatter the flat patch vectors onto the level-l grid
            g = scatter_tokens(per_level[l], m, p * p * C)   # (B, p*p*C, H_l, W_l)
            b, _, hl, wl = g.shape
            # (p*p*C) -> (C, p, p), then tile n x n and fold into pixels
            g = g.reshape(b, C, p, p, hl, wl)
            g = g.permute(0, 1, 4, 2, 5, 3)          # (B, C, H_l, p, W_l, p)
            g = g.reshape(b, C, hl * p, wl * p)      # patch grid -> pixels
            if n > 1:
                g = g.repeat_interleave(n, dim=2).repeat_interleave(n, dim=3)
            # same for the mask, so levels stay disjoint
            mm = m[:, None].to(g.dtype)
            mm = mm.repeat_interleave(p * n, dim=2).repeat_interleave(p * n, dim=3)
            g = g * mm
            out = g if out is None else out + g
        if out is None:
            out = tokens.new_zeros(B, C, H, H)
        return out

    def forward(self, tokens, masks, z, t_emb=None):
        """
        tokens: (N_total, d) in level order 0, 1, 2 (as produced by the encoder)
        masks:  same list of (B, H_l, W_l) bool used by the encoder
        z:      (B, C, H, W) noisy latent, for the long skip
        returns x_hat: (B, C, H, W)
        """
        base = self.base_predict(tokens, masks) if self.residual else None
        h = self.in_proj(tokens)
        counts = [int(m.sum()) for m in masks]
        per_level = list(torch.split(h, counts, dim=0))

        # start at the coarsest level: a sparse grid, mostly empty
        l = self.n_levels - 1
        f = scatter_tokens(per_level[l], masks[l], self.c)
        f = f + self.mask_proj[l](masks[l][:, None].to(f.dtype))
        f = self.blends[l](f, t_emb)

        # walk down, filling the holes as we go
        for l in range(self.n_levels - 2, -1, -1):
            f = self.splits[l](f)
            m = masks[l]
            placed = scatter_tokens(per_level[l], m, self.c)
            # disjoint by the quadtree: overwrite where this level owns the cell
            f = torch.where(m[:, None], placed, f)
            f = f + self.mask_proj[l](m[:, None].to(f.dtype))
            f = self.blends[l](f, t_emb)

        f = self.head_split(f)                       # -> full latent resolution
        f = self.skip_proj(torch.cat([f, z], dim=1))  # long skip from z_t
        f = self.head_blend(f, t_emb)
        res = self.out(f)
        if not self.residual:
            return res
        # Per-level gate, broadcast back onto the dense grid so each region is
        # gated by the level that owns it. Coarse leaves need the pyramid most,
        # so letting their gate open at its own rate is worth the bookkeeping.
        gate = self._gate_map(masks, res)
        return base + gate * res

    def _gate_map(self, masks, ref):
        """(B, 1, H, W) map of res_gamma[l] over each level's footprint."""
        p = self.patch_size
        gate = torch.zeros_like(ref[:, :1])
        for l, m in enumerate(masks):
            n = 2 ** l
            mm = m[:, None].to(ref.dtype)
            mm = mm.repeat_interleave(p * n, dim=2).repeat_interleave(p * n, dim=3)
            gate = gate + self.res_gamma[l].view(1, 1, 1, 1) * mm
        return gate