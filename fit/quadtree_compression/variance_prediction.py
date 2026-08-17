"""
Lightweight variance predictor for a latent diffusion process.

Predicts, from a noisy latent x_t and its timestep t, the channel-wise variance
of the *final* image (x0) over the NON-OVERLAPPING quadtree regions of three
stages: 2x2, 4x4 and 8x8 patches.

Architecture (transformer, from our discussion):
  - Patchify the (C, 32, 32) latent with a 2x2 patch embed -> a 16x16 grid of
    tokens; each token corresponds to exactly one non-overlapping 2x2 region.
  - Stage 1: 4 transformer layers, then a prediction head -> per-token log-var
    for the 2x2 regions (16x16 grid).
  - Stage 2: 2x2 average pool over the token grid (tokens now align with the
    4x4 regions, 8x8 grid), 2 more transformer layers, a prediction head -> 4x4.
  - Stage 3: another 2x2 average pool (tokens align with 8x8 regions, 4x4 grid),
    2 more transformer layers, a prediction head -> 8x8.

Design decisions carried over from the conv version:
  - Conditioning on t: FiLM via adaptive LayerNorm (adaLN), computed once as a
    single `temb` shared by all blocks. adaLN-zero init -> blocks start as identity.
  - Norm: RMSNorm (scale-only). adaLN still injects BOTH gamma and beta; the beta
    shift does the mean control that RMSNorm dropped. Do not remove beta.
  - Activation: ReLU^2 in the MLPs.
  - Output: log-variance PER CHANNEL (unconstrained). exp() at readout -> variance.
    The channel-wise MAX is taken only at readout, never in the loss, so each
    channel keeps an honest per-channel likelihood term.
  - Loss: scale-NLL  log_var + s^2 * exp(-log_var)  (Stein / Gaussian scale loss).
  - Head bias init: log(median target) per scale, so training starts at the
    marginal instead of at sigma^2 ~ 1.

Target convention:
    target[s] = empirical variance s^2 of x0 over each non-overlapping N_s x N_s
    region, for every latent channel, laid out on the (H/N_s, W/N_s) region grid.
    Because the three stages live on different grids the targets are a LIST of
    tensors, target[s] of shape (B, latent_channels, H/N_s, W/N_s), one per
    region size in region_sizes=(2, 4, 8). No margins / NaNs are needed since
    the regions tile the latent exactly.
"""

import glob
import math
import os
import subprocess
import tarfile

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from safetensors import safe_open
from torch.utils.data import Dataset, DataLoader

# --------------------------------------------------------------------------- #
# Data: greater_than_256_crop  (Colab)                                        #
# --------------------------------------------------------------------------- #
# The dataset is a folder of .safetensors feature files, same format used in
# latent_region_variance.py / latent_upsample_compare.py:
#     file["feature"]  (2, H, W, 16)  — [0] = un-flipped image
#     file["size"]     (2,)           — (H, W) grid cells
# token width 16 = (c p1 p2), p=2, c=4  →  de-patchify to latent (4, H*2, W*2),
# already scaled by 0.18215.
#
# On Colab we fetch it from Google Drive (public share link) and untar once.
P = 2                 # patch size
LATENT_C = 4          # VAE latent channels
LATENT_SCALE = 0.18215
GDRIVE_FILE_ID = "1JU39tvx1wR0OzymxBQ_44e0lty5t0dZX"     # greater_than_256_crop.tar.gz


def download_dataset(root="/content/data", file_id=GDRIVE_FILE_ID):
    """
    Download + extract greater_than_256_crop.tar.gz from Google Drive.
    Returns the folder that holds the .safetensors files. Idempotent.
    """
    os.makedirs(root, exist_ok=True)
    crop_dir = os.path.join(root, "greater_than_256_crop")
    if glob.glob(os.path.join(crop_dir, "*.safetensors")):
        print(f"dataset already present at {crop_dir}")
        return crop_dir

    tar_path = os.path.join(root, "greater_than_256_crop.tar.gz")
    if not os.path.exists(tar_path):
        print("downloading greater_than_256_crop.tar.gz from Google Drive ...")
        try:
            import gdown
        except ImportError:
            subprocess.run(["pip", "install", "-q", "gdown"], check=True)
            import gdown
        gdown.download(id=file_id, output=tar_path, quiet=False)

    print(f"extracting {tar_path} ...")
    with tarfile.open(tar_path, "r:gz") as tf:
        tf.extractall(root)

    # the tar may extract straight into `root` or into a nested folder; locate it
    if not glob.glob(os.path.join(crop_dir, "*.safetensors")):
        hits = glob.glob(os.path.join(root, "**", "*.safetensors"), recursive=True)
        if not hits:
            raise RuntimeError(f"no .safetensors found after extracting {tar_path}")
        crop_dir = os.path.dirname(hits[0])
    print(f"dataset ready at {crop_dir}  "
          f"({len(glob.glob(os.path.join(crop_dir, '*.safetensors')))} files)")
    return crop_dir


def load_latent(path):
    """De-patchify a features file into a spatial latent (4, H*2, W*2)."""
    with safe_open(path, "pt") as f:
        feat = f.get_tensor("feature")              # (2, H, W, 16)
        size = f.get_tensor("size")
    H, W = int(size[0]), int(size[1])
    x = feat[0].reshape(1, H * W, 16)               # pick un-flipped image
    sp = rearrange(x, "b (h w) (c p1 p2) -> b c (h p1) (w p2)",
                   h=H, w=W, p1=P, p2=P, c=LATENT_C)
    return sp[0]                                     # (4, H*2, W*2)


def region_variance_targets(latent, region_sizes):
    """
    Per-channel variance of x0 over NON-OVERLAPPING (stride N) N x N regions,
    N in region_sizes. Biased / population variance via box means E[x^2]-E[x]^2,
    keeping ALL channels (the model wants a per-channel target; the channel-wise
    max is taken only at readout).

    The three quadtree stages live on different grids, so this returns a LIST of
    tensors, one per region size: target[s] of shape (C, Hs/N, Ws/N). The regions
    tile the latent exactly (Hs, Ws are multiples of every N here), so there are
    no margins and no NaNs.

    latent: (C, Hs, Ws).
    """
    x = latent.float()[None]                         # (1, C, Hs, Ws)
    out = []
    for n in region_sizes:
        mean = F.avg_pool2d(x, n, stride=n)          # E[x]   (1,C,oh,ow)
        mean_sq = F.avg_pool2d(x * x, n, stride=n)   # E[x^2]
        var = (mean_sq - mean * mean).clamp_min(0)[0]   # (C, oh, ow), biased
        out.append(var)
    return out                                       # list of (C, Hs/N, Ws/N)


class LatentVarianceDataset(Dataset):
    """
    Yields (x_t, t, target) for the variance predictor:
        x_t:    (C, H, W)                 noisy latent at time t
        t:      scalar float in [0, 1]    sampled flow-matching time
        target: list of (C, H/N, W/N)     per-channel x0 variance per region size

    The target is a list (one entry per region size in region_sizes) because the
    quadtree stages live on different grids; use `collate_variance` to batch.

    Noising is FLOW-MATCHING / rectified-flow style: a linear interpolant
        x_t = (1 - t) * x0 + t * eps,   t ~ Uniform[0, 1],  eps ~ N(0, I)
    so t=0 is the clean latent and t=1 is pure noise. t is sampled uniformly
    (all noise levels equally likely) and drawn fresh every __getitem__, so
    across epochs each latent is seen at many noise levels. Latents are
    cropped/padded to `crop` so we can batch them. t is passed to the model as-is
    in [0, 1]; the sinusoidal embedding rescales it internally.
    """

    def __init__(self, crop_dir, region_sizes=(2, 4, 8),
                 crop=32, seed=0):
        self.paths = sorted(glob.glob(os.path.join(crop_dir, "*.safetensors")))
        if not self.paths:
            raise RuntimeError(f"no .safetensors in {crop_dir}")
        self.region_sizes = tuple(region_sizes)
        self.crop = crop
        self._gen = torch.Generator().manual_seed(seed)

    def __len__(self):
        return len(self.paths)

    def _center_crop(self, latent):
        """Center-crop (or reflect-pad) a (C, Hs, Ws) latent to (C, crop, crop)."""
        c = self.crop
        _, Hs, Ws = latent.shape
        # pad if smaller
        ph, pw = max(0, c - Hs), max(0, c - Ws)
        if ph or pw:
            latent = F.pad(latent[None], (0, pw, 0, ph), mode="reflect")[0]
            _, Hs, Ws = latent.shape
        top = (Hs - c) // 2
        left = (Ws - c) // 2
        return latent[:, top:top + c, left:left + c]

    def __getitem__(self, i):
        x0 = self._center_crop(load_latent(self.paths[i]))   # (C, crop, crop)
        target = region_variance_targets(x0, self.region_sizes)

        # flow-matching forward process: linear interpolant x0 <-> noise
        t = torch.rand(1, generator=self._gen)               # Uniform[0, 1]
        eps = torch.randn(x0.shape, generator=self._gen)
        x_t = (1 - t) * x0 + t * eps
        return x_t.float(), t.squeeze(0).float(), [g.float() for g in target]


def collate_variance(batch):
    """
    Collate (x_t, t, target_list) items. The target is a list of per-scale
    tensors on different grids, so we stack each scale separately and return a
    list of batched tensors target[s] of shape (B, C, H/N, W/N).
    """
    xs, ts, targets = zip(*batch)
    x = torch.stack(xs)
    t = torch.stack(ts)
    n_scales = len(targets[0])
    target = [torch.stack([item[s] for item in targets]) for s in range(n_scales)]
    return x, t, target


def make_loader(crop_dir=None, *, batch_size=32, region_sizes=(2, 4, 8),
                crop=32, num_workers=2, download=True, shuffle=True):
    """Convenience: (download →) dataset → DataLoader ready for `train`."""
    if crop_dir is None:
        crop_dir = download_dataset() if download else "/content/data/greater_than_256_crop"
    ds = LatentVarianceDataset(crop_dir, region_sizes=region_sizes, crop=crop)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, drop_last=True, pin_memory=True,
                      collate_fn=collate_variance)


# --------------------------------------------------------------------------- #
# Time embedding                                                              #
# --------------------------------------------------------------------------- #
def sinusoidal_embedding(t, dim, max_period=10_000, time_scale=1000.0):
    """
    Flow-matching time embedding. t: (B,) continuous in [0, 1] -> (B, dim).

    t is scaled by `time_scale` before the sinusoids so the geometric frequency
    ladder (periods 2*pi*time_scale ... 2*pi/max_period*time_scale) actually
    resolves the [0, 1] interval — an un-scaled [0, 1] input would leave the
    low-frequency channels nearly constant. With time_scale=1000 this matches
    the resolution of integer DDPM timesteps in [0, 1000].
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, device=t.device) / half
    )
    args = (t.float() * time_scale)[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim)
        )

    def forward(self, t):
        return self.mlp(sinusoidal_embedding(t, self.dim))


# --------------------------------------------------------------------------- #
# Norm + activation                                                           #
# --------------------------------------------------------------------------- #
class RMSNorm(nn.Module):
    """RMSNorm over the last (channel) dim. No affine: adaLN supplies scale/shift."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


def relu2(x):
    return F.relu(x).pow(2)


# --------------------------------------------------------------------------- #
# 2D axial RoPE                                                               #
# --------------------------------------------------------------------------- #
def axial_rope_tables(g, head_dim, device=None, dtype=torch.float32,
                      max_period=10_000):
    """
    Build cos/sin tables for 2D axial RoPE over a g x g token grid.

    The per-head dim is split in half: the first half is rotated by the token's
    row coordinate, the second half by its column coordinate (indices 0..g-1 per
    axis, this stage's own grid). Returns cos, sin each of shape (g*g, head_dim),
    laid out to match tokens flattened row-major (as in `x.flatten(2)`).
    """
    assert head_dim % 4 == 0, "head_dim must be divisible by 4 for axial RoPE"
    half = head_dim // 2                             # dims per axis
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(0, half, 2, device=device,
                                              dtype=dtype) / half
    )                                                # (half/2,)
    coord = torch.arange(g, device=device, dtype=dtype)
    # row-major token order: row index varies slow, col index varies fast
    rows = coord[:, None].expand(g, g).reshape(-1)   # (g*g,)
    cols = coord[None, :].expand(g, g).reshape(-1)   # (g*g,)
    ang_r = rows[:, None] * freqs[None]              # (g*g, half/2)
    ang_c = cols[:, None] * freqs[None]
    ang = torch.cat([ang_r, ang_c], dim=-1)          # (g*g, half)
    ang = torch.cat([ang, ang], dim=-1)              # (g*g, head_dim), dup for pairs
    return ang.cos(), ang.sin()


def apply_rope(x, cos, sin):
    """
    Rotate the last dim of x by the RoPE angles. x: (B, heads, N, head_dim);
    cos/sin: (N, head_dim). Uses the rotate-half convention.
    """
    d = x.shape[-1]
    x1, x2 = x[..., : d // 2], x[..., d // 2 :]
    rot = torch.cat([-x2, x1], dim=-1)
    return x * cos + rot * sin


# --------------------------------------------------------------------------- #
# Transformer block with adaLN-zero conditioning                              #
# --------------------------------------------------------------------------- #
def modulate(x, gamma, beta):
    """FiLM on token features. x: (B, N, C); gamma/beta: (B, C)."""
    return x * (1 + gamma[:, None]) + beta[:, None]


class TransformerAdaLNBlock(nn.Module):
    """
    Pre-norm transformer block (MHSA + ReLU^2 MLP) with adaLN-zero conditioning.

    t injects, per sub-layer, a (gamma, beta) FiLM pair plus an output gate alpha.
    adaLN-zero: the modulation projection starts at 0, so gamma=beta=alpha=0 and
    the block is the identity at init (DiT-style, stable start).

    Attention is explicit MHSA (Q/K/V Linears + scaled_dot_product_attention) so
    2D axial RoPE can rotate Q and K before the attention product; position is
    supplied as (cos, sin) tables built per stage from that stage's token grid.
    """

    def __init__(self, dim, t_dim, num_heads=6, mlp_ratio=4):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.norm1 = RMSNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.norm2 = RMSNorm(dim)
        hidden = mlp_ratio * dim
        self.mlp1 = nn.Linear(dim, hidden)
        self.mlp2 = nn.Linear(hidden, dim)
        # -> gamma1, beta1, alpha1, gamma2, beta2, alpha2
        self.ada = nn.Linear(t_dim, 6 * dim)
        nn.init.zeros_(self.ada.weight)
        nn.init.zeros_(self.ada.bias)

    def _attn(self, h, cos, sin):                    # h: (B, N, C)
        B, N, _ = h.shape
        qkv = self.qkv(h).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)             # (3, B, heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        out = F.scaled_dot_product_attention(q, k, v)   # (B, heads, N, head_dim)
        out = out.transpose(1, 2).reshape(B, N, self.num_heads * self.head_dim)
        return self.proj(out)

    def forward(self, x, temb, cos, sin):            # x: (B, N, C)
        g1, b1, a1, g2, b2, a2 = self.ada(temb).chunk(6, dim=-1)
        h = modulate(self.norm1(x), g1, b1)
        h = self._attn(h, cos, sin)
        x = x + a1[:, None] * h
        h = modulate(self.norm2(x), g2, b2)
        h = self.mlp2(relu2(self.mlp1(h)))
        x = x + a2[:, None] * h
        return x


# --------------------------------------------------------------------------- #
# Prediction head                                                             #
# --------------------------------------------------------------------------- #
class VarianceHead(nn.Module):
    """
    Per-token log-variance head for one quadtree stage. A pre-norm Linear that
    maps each token (B, N, dim) to latent_channels log-variances.

    Zero weight -> at init every token predicts a constant = the marginal (the
    bias). Gradients still reach the trunk from step 1 (DiT-style zero-init final
    layer); harmless one-step delay, stable start.
    """

    def __init__(self, dim, latent_channels, median_target=0.5):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.proj = nn.Linear(dim, latent_channels)
        nn.init.zeros_(self.proj.weight)
        nn.init.constant_(self.proj.bias, math.log(median_target))

    def forward(self, x):                            # x: (B, N, dim)
        return self.proj(self.norm(x))               # (B, N, latent_channels)


# --------------------------------------------------------------------------- #
# Model                                                                       #
# --------------------------------------------------------------------------- #
class VariancePredictor(nn.Module):
    """
    Three-stage quadtree transformer. A 2x2 patch embed turns a (C, H, W) latent
    into an (H/2)x(W/2) token grid, then:

        stage 0:  depths[0] transformer layers -> head -> 2x2 regions
        stage 1:  2x2 avg-pool tokens, depths[1] layers -> head -> 4x4 regions
        stage 2:  2x2 avg-pool tokens, depths[2] layers -> head -> 8x8 regions

    forward returns a list of log_var tensors, one per region size, each of shape
    (B, latent_channels, H/N, W/N).
    """

    def __init__(
        self,
        latent_channels=4,
        region_sizes=(2, 4, 8),
        dim=192,
        depths=(4, 2, 2),                # transformer layers per stage
        num_heads=6,
        t_dim=128,
        median_targets=None,             # per-scale medians for bias init
    ):
        super().__init__()
        assert tuple(region_sizes) == (2, 4, 8), \
            "this model is specialised to the 2x2 / 4x4 / 8x8 quadtree stages"
        assert len(depths) == len(region_sizes)
        self.latent_channels = latent_channels
        self.region_sizes = tuple(region_sizes)
        self.n_scales = len(region_sizes)

        self.time = TimeEmbedding(t_dim)
        # 2x2 patch embed: each token == one non-overlapping 2x2 region.
        self.patch = nn.Conv2d(latent_channels, dim, kernel_size=P, stride=P)
        # Position is injected via 2D axial RoPE inside attention (no learned
        # pos-embed); the head_dim must be divisible by 4 for the axial split.
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.stages = nn.ModuleList(
            nn.ModuleList(
                TransformerAdaLNBlock(dim, t_dim, num_heads=num_heads)
                for _ in range(d)
            )
            for d in depths
        )

        # Token merges between stages: a depthwise (channel-wise) 2x2 stride-2
        # conv — the 4-children -> 1 spatial linear merge, block-diagonal over
        # channels, so it costs the same as avg-pool but can learn a spatial
        # pattern other than the plain mean (e.g. retain some cross-child spread,
        # which a mean discards). Init to 1/4 so it *starts as* avg-pool: the
        # coarse target's mean is exactly the mean of its four children, keeping
        # the warm-start target-consistent. One merge per coarsening step.
        self.merges = nn.ModuleList(
            nn.Conv2d(dim, dim, kernel_size=2, stride=2, groups=dim)
            for _ in range(len(depths) - 1)
        )
        for m in self.merges:
            nn.init.constant_(m.weight, 0.25)
            nn.init.zeros_(m.bias)

        if median_targets is None:
            defaults = {2: 0.3600, 4: 0.4551, 8: 0.5394}
            median_targets = [defaults.get(n, 0.5) for n in self.region_sizes]
        self.heads = nn.ModuleList(
            VarianceHead(dim, latent_channels, median_target=m)
            for m in median_targets
        )

    def forward(self, x, t):
        """
        x: (B, latent_channels, H, W), t: (B,) ->
        list of log_var[s] of shape (B, latent_channels, H/N_s, W/N_s).
        """
        temb = self.time(t)
        x = self.patch(x)                            # (B, dim, g, g), g = H/2
        B, dim, g, _ = x.shape

        outs = []
        for s, stage in enumerate(self.stages):
            if s > 0:                                # coarsen tokens 2x2 -> next stage
                x = self.merges[s - 1](x)            # depthwise 2x2 stride-2 merge
                g = x.shape[-1]
            # per-stage 2D axial RoPE tables from this stage's g x g grid
            cos, sin = axial_rope_tables(g, self.head_dim,
                                         device=x.device, dtype=x.dtype)
            tok = x.flatten(2).transpose(1, 2)       # (B, g*g, dim)
            for blk in stage:
                tok = blk(tok, temb, cos, sin)
            x = tok.transpose(1, 2).reshape(B, dim, g, g)

            log_var = self.heads[s](tok)             # (B, g*g, C)
            log_var = log_var.transpose(1, 2).reshape(B, self.latent_channels, g, g)
            outs.append(log_var)
        return outs


# --------------------------------------------------------------------------- #
# Loss + readout                                                              #
# --------------------------------------------------------------------------- #
def variance_nll_loss(log_var, target, scale_weights=None, clamp=(-12.0, 12.0)):
    """
    Scale-NLL (Stein loss):  log_var + s^2 * exp(-log_var), per channel.
    target = s^2 (per-channel variance).

    log_var, target: LISTS of per-scale tensors, each (B, C, H/N, W/N) (the
    quadtree stages live on different grids). scale_weights: optional
    (n_scales,) per-scale weights.

    The mean is taken over all elements of all scales (each scale weighted by
    scale_weights[s]); larger regions have fewer tokens, so this naturally gives
    every stage's tokens equal per-token weight.
    """
    lo, hi = clamp
    total = log_var[0].new_zeros(())
    denom = log_var[0].new_zeros(())
    for s, (lv, tgt) in enumerate(zip(log_var, target)):
        lv = lv.clamp(lo, hi)
        per = lv + tgt * torch.exp(-lv)
        w = 1.0 if scale_weights is None else float(scale_weights[s])
        total = total + w * per.sum()
        denom = denom + w * per.numel()
    return total / denom.clamp(min=1.0)


def dof_scale_weights(region_sizes, normalize=True):
    """
    Per-scale weight k/2 with k = N^2 - 1 dof (larger regions -> more reliable s^2).
    NOTE: this leans on the within-region-Gaussian assumption, which is the
    shakiest one here, and it heavily up-weights large regions (k ranges 3..63).
    Default training uses uniform weights; enable this only if it helps in eval.
    """
    k = torch.tensor([n * n - 1 for n in region_sizes], dtype=torch.float)
    w = k / 2
    return w / w.mean() if normalize else w


@torch.no_grad()
def predict_variance(model, x, t):
    """
    Readout: per-channel variance -> channel-wise max, per stage. Returns a list
    of tensors, one per region size, each (B, H/N, W/N).
    """
    log_var = model(x, t)
    return [lv.exp().max(dim=1).values for lv in log_var]


# --------------------------------------------------------------------------- #
# Training                                                                     #
# --------------------------------------------------------------------------- #
def train_step(model, optimizer, x, t, target, scale_weights=None, grad_clip=1.0):
    model.train()
    log_var = model(x, t)
    loss = variance_nll_loss(log_var, target, scale_weights)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    if grad_clip is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    return loss.item()


@torch.no_grad()
def oracle_loss(loader, scale_weights=None, n_batches=20):
    """
    Loss floor: the scale-NLL when the model predicts every variance perfectly,
    i.e. log_var = log(s^2), giving per-element loss log(s^2) + 1. Because most
    targets are < 1, this floor is NEGATIVE — so a negative training loss is
    expected, not a bug. Report the gap to this floor, not the sign of the loss.
    """
    tot, cnt = 0.0, 0
    for b, (_, _, target) in enumerate(loader):
        log_var = [torch.log(g.clamp_min(1e-8)) for g in target]   # perfect prediction
        tot += variance_nll_loss(log_var, target, scale_weights).item()
        cnt += 1
        if b + 1 >= n_batches:
            break
    return tot / max(cnt, 1)


def train(model, loader, *, epochs=10, lr=3e-4, weight_decay=1e-4,
          use_dof_weights=False, device="cuda"):
    """
    `loader` yields (x, t, target):
        x:      (B, latent_channels, 32, 32)  noisy latents x_t
        t:      (B,)                           flow-matching time in [0, 1]
        target: list of (B, latent_channels, 32/N, 32/N)  per-channel s^2 per stage
    """
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * len(loader))
    sw = dof_scale_weights(model.region_sizes) if use_dof_weights else None

    floor = oracle_loss(loader, sw)
    print(f"oracle loss (perfect prediction floor): {floor:.4f}  "
          f"-> a negative training loss is expected; watch the gap to this.")

    for epoch in range(epochs):
        running = 0.0
        for x, t, target in loader:
            x, t = x.to(device), t.to(device)
            target = [g.to(device) for g in target]
            running += train_step(model, opt, x, t, target, sw)
            sched.step()
        avg = running / len(loader)
        print(f"epoch {epoch:3d}  loss {avg:.4f}  (gap to floor {avg - floor:+.4f})")
    return model


# --------------------------------------------------------------------------- #
# Entrypoints                                                                 #
# --------------------------------------------------------------------------- #
def smoke_test():
    """Random-data check: shapes + a few training steps run."""
    torch.manual_seed(0)
    B, C, H, W = 8, 4, 32, 32
    region_sizes = (2, 4, 8)

    model = VariancePredictor(latent_channels=C, region_sizes=region_sizes)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params: {n_params/1e6:.2f}M")

    x = torch.randn(B, C, H, W)
    t = torch.rand(B)
    log_var = model(x, t)
    print("log_var:", [tuple(lv.shape) for lv in log_var])      # per-stage (B,C,H/N,W/N)
    print("readout:", [tuple(v.shape) for v in predict_variance(model, x, t)])

    # fake per-channel, per-stage targets around ~0.5 on each stage's grid
    target = [torch.rand(B, C, H // n, W // n) * 0.5 + 0.3 for n in region_sizes]
    print("init loss:", variance_nll_loss(log_var, target).item())

    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    for i in range(5):
        print(f"step {i}: {train_step(model, opt, x, t, target):.4f}")


def run_training(args):
    """Download greater_than_256_crop, then train on it (Colab)."""
    region_sizes = (2, 4, 8)
    device = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device={device}")

    loader = make_loader(crop_dir=args.crop_dir, batch_size=args.batch_size,
                         region_sizes=region_sizes, num_workers=args.num_workers,
                         download=not args.no_download)

    # sanity: inspect one batch's shapes
    xb, tb, yb = next(iter(loader))
    print(f"batch  x_t={tuple(xb.shape)}  t={tuple(tb.shape)}  "
          f"target={[tuple(g.shape) for g in yb]}")

    model = VariancePredictor(latent_channels=LATENT_C, region_sizes=region_sizes)
    print(f"params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    train(model, loader, epochs=args.epochs, lr=args.lr,
          use_dof_weights=args.dof_weights, device=device)

    if args.save:
        torch.save(model.state_dict(), args.save)
        print(f"saved model -> {args.save}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train", action="store_true",
                    help="train on greater_than_256_crop (default: smoke test)")
    ap.add_argument("--crop-dir", default=None,
                    help="dataset folder; if unset, download from Google Drive")
    ap.add_argument("--no-download", action="store_true",
                    help="skip the Drive download (use an existing --crop-dir)")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--dof-weights", action="store_true")
    ap.add_argument("--save", default=None, help="path to save state_dict")
    args = ap.parse_args()

    if args.train:
        run_training(args)
    else:
        smoke_test()