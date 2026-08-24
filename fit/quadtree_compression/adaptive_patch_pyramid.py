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

    def __init__(self, latent_ch, c, d, n_levels=3, share_weights=False):
        super().__init__()
        self.n_levels = n_levels
        self.c = c
        self.stem = nn.Conv2d(4 * latent_ch, c, 1)
        if share_weights:
            merge = Merge(c)
            self.merges = nn.ModuleList([merge] * (n_levels - 1))
        else:
            self.merges = nn.ModuleList([Merge(c) for _ in range(n_levels - 1)])
        self.proj = nn.Linear(c, d)
        self.scale_emb = nn.Parameter(torch.zeros(n_levels, d))

    def features(self, z):
        """z: (B, C, H, W) -> list of (B, c, H/2^(l+1), W/2^(l+1)), fine to coarse."""
        x = self.stem(F.pixel_unshuffle(z, 2))
        feats = [x]
        for merge in self.merges:
            x = merge(x)
            feats.append(x)
        return feats

    def forward(self, z, masks):
        """
        masks: list of (B, H_l, W_l) bool, one per level, disjoint and complete
               under the quadtree.

        Returns (tokens, index) where tokens is (N_total, d) and index is a
        (N_total,) long tensor of level ids, in level order 0, 1, 2. Keep this
        ordering -- the decoder's scatter assumes it.
        """
        feats = self.features(z)
        tokens, levels = [], []
        for l, (f, m) in enumerate(zip(feats, masks)):
            t = self.proj(gather_tokens(f, m)) + self.scale_emb[l]
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
                 share_weights=False, zero_init_out=True):
        super().__init__()
        self.n_levels = n_levels
        self.c = c
        c_head = c_head or c // 4
        self.c_head = c_head

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
        if zero_init_out:
            nn.init.zeros_(self.out.weight)
            nn.init.zeros_(self.out.bias)
        else:
            nn.init.normal_(self.out.weight, std=0.02)
            nn.init.zeros_(self.out.bias)

    def forward(self, tokens, masks, z, t_emb=None):
        """
        tokens: (N_total, d) in level order 0, 1, 2 (as produced by the encoder)
        masks:  same list of (B, H_l, W_l) bool used by the encoder
        z:      (B, C, H, W) noisy latent, for the long skip
        returns x_hat: (B, C, H, W)
        """
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
        return self.out(f)