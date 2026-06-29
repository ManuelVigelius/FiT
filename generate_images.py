"""
Image generation script: generates images per (model, schedule) pair using
the noise-field sampler with CFG=4.0.

Each schedule is a sequence of grid sizes that the noise-field sampler runs at
progressively, drawing consistent noise fields internally.

Generated images are saved as PNG under:
  <OUTPUT_DIR>/<ckpt_name>/noise_field_<idx>/<idx:06d>.png

All configuration lives in the CONFIG block below — no CLI arguments needed.
"""


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import json
import math
import shutil
import sys
import zipfile
from itertools import combinations_with_replacement
from pathlib import Path

import torch

from PIL import Image
from safetensors.torch import load_file
from diffusers.models import AutoencoderKL

# ─────────────────────────────── CONFIG ─────────────────────────────────────

# Scale schedules: all length-4 combinations (with replacement) from the digit set
# that end at 16 (full-res), each interpolated to 16 steps.
def _make_schedules():
    digits = [1, 6, 11, 16]

    def interpolate(seq, target_len=16):
        n = len(seq)
        segments = target_len - 1
        gaps = n - 1
        steps_per_gap = segments // gaps
        result = []
        for i in range(gaps):
            a, b = seq[i], seq[i + 1]
            for j in range(steps_per_gap):
                result.append(int(round(a + j * (b - a) / steps_per_gap)))
        result.append(seq[-1])
        return result

    return {
        idx: interpolate(x)
        for idx, x in enumerate(combinations_with_replacement(digits, 4))
        if x[-1] == digits[-1]
    }

SCHEDULES = {0: [16] * 16}

# ── Quadtree (mixed-resolution) demo (Loss-C / use_upsampler models only) ────
# When USE_QUADTREE is True, the Loss-C path ignores SCHEDULES and instead runs
# the mixed-resolution sampler: the whole frame is conditioned at base_k token
# density, with the listed refine windows rendered at higher density. Windows
# are in *grid (token)* units on the H_fr×W_fr full-res grid and must align to
# the base cell grid (see build_refinement_quadtree).
USE_QUADTREE = True
QUADTREE_STEPS = 16          # number of Euler steps
QUADTREE_PER_LEAF_SIGMA = True  # noise each cell at its own level (vs one global)
QUADTREE = dict(
    base_k=8,                # base density over the full 16×16 grid (coarse)
    refine=[                 # (y0, x0, h, w, k) in grid units; here: sharp center box
        (4, 4, 8, 8, 8),
    ],
)

# Number of images to generate per (checkpoint, schedule) pair.
N_IMAGES = 16

# Batch size for the generation loop (single GPU).
BATCH_SIZE = 64

# Classifier-free guidance scale.
CFG_SCALE = 4.0

# ImageNet class range — labels are sampled uniformly from [0, NUM_CLASSES).
NUM_CLASSES = 1000

# Target latent resolution (matches training).  The full-res grid is
# TARGET_LEN_PIX // (patch_size * vae_scale) = 256 // (2 * 8) = 16.
TARGET_LEN_PIX = 256

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# VAE checkpoint — must be locally available.
VAE_PATH = "stabilityai/sd-vae-ft-ema"

# Output directory for generated images.
OUTPUT_DIR = "/visinf/projects_students/mb_mvigel/images"

# Global seed for reproducibility.
GLOBAL_SEED = 42

# Base model config (shared by all checkpoints).
_BASE_MODEL_CFG = dict(
    context_size=256,
    patch_size=2,
    in_channels=4,
    hidden_size=1152,
    depth=36,
    num_heads=16,
    mlp_ratio=4.0,
    class_dropout_prob=0.1,
    num_classes=1000,
    learn_sigma=False,
    use_swiglu=True,
    use_swiglu_large=False,
    q_norm="layernorm",
    k_norm="layernorm",
    qk_norm_weight=False,
    rel_pos_embed="rope",
    online_rope=True,
    adaln_type="lora",
    adaln_lora_dim=288,
    use_size_cond=True,
)

# Checkpoints to generate from.
# For accelerate checkpoints, set `dir` to the checkpoint-NNNN directory;
# the script will resolve model.safetensors / ema_model.safetensors inside it.
# For plain .safetensors files (e.g. baseline), set `dir` to the file path directly.
CHECKPOINTS = [ # cluster paths
    # dict(
    #     name="baseline",
    #     dir="/visinf/projects_students/mb_mvigel/checkpoints/model_ema.safetensors",
    #     loss_type="baseline",
    # ),
    dict(
        name="loss_a_8k_ema",
        dir="/visinf/projects_students/mb_mvigel/workdir/fitv2_xl_cluster_a/checkpoints/checkpoint-8000",
        loss_type="A",
        use_ema=True,
    ),
    dict(
        name="loss_c_10k_ema",
        dir="/visinf/projects_students/mb_mvigel/workdir/fitv2_xl_cluster_c/checkpoints/checkpoint-8000",
        loss_type="C",
        use_ema=True,
    ),
    # dict(
    #     name="loss_a_8k_train",
    #     dir="/visinf/projects_students/mb_mvigel/workdir/fitv2_xl_cluster_a/checkpoints/checkpoint-8000",
    #     loss_type="A",
    #     use_ema=False,
    # ),
]


# CHECKPOINTS = [ # colab
#     dict(
#         name="baseline",
#         dir="/content/drive/MyDrive/FiT/inference_weights/checkpoint-baseline",
#         loss_type="A",
#         use_ema=True
#     ),
#     dict(
#         name="loss_a_8k_ema",
#         dir="/content/drive/MyDrive/FiT/inference_weights/checkpoint-8000",
#         loss_type="A",
#         use_ema=True,
#     ),
#     # dict(
#     #     name="loss_a_8k_train",
#     #     dir="/content/drive/MyDrive/FiT/inference_weights/checkpoint-8000",
#     #     loss_type="A",
#     #     use_ema=False,
#     # ),
# ]

# ─────────────────────────────────────────────────────────────────────────────

PATCH_SIZE = 2
VAE_SCALE  = 8   # SD VAE 8× spatial downsampling
C_IN       = 4   # VAE latent channels
PAD_TO_MULTIPLE = 128  # packed-sequence padding (matches packed_collate_fn)

sys.path.insert(0, str(Path(__file__).parent))

from fit.model.fit_model import FiT
from fit.noise_field_sampler.noise_field_sampler import (
    sample as noise_field_sample,
    sample_upsampler as noise_field_sample_upsampler,
    sample_upsampler_quadtree as noise_field_sample_upsampler_quadtree,
)
from fit.noise_field_sampler.quadtree import build_refinement_quadtree
from fit.scheduler.transport.utils import patchify, unpatchify


# ──────────────────────────── helpers ────────────────────────────────────────

def model_cfg_for(loss_type: str) -> dict:
    cfg = dict(_BASE_MODEL_CFG)
    cfg["use_size_cond"] = (loss_type not in ("baseline", "virtual_resize"))
    # Loss C builds the learned-upsampler tail (up_proj, fr_embedder, up_blocks;
    # the prediction head is shared with the low-res path). Without this the
    # checkpoint's upsampler weights would be
    # reported as unexpected keys and the model would run the plain low-res head.
    cfg["use_upsampler"] = (loss_type == "C")
    return cfg


def load_model(ckpt_path: str, cfg: dict, device: str) -> FiT:
    model = FiT(**cfg)
    state = load_file(ckpt_path, device="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  [warn] {len(missing)} missing keys (first 5: {missing[:5]})")
    if unexpected:
        print(f"  [warn] {len(unexpected)} unexpected keys (first 5: {unexpected[:5]})")
    return model.to(device).eval()


def make_grid_and_mask(H_g: int, W_g: int, B: int, device: torch.device, dtype: torch.dtype):
    """Build grid, mask, and size tensors for a given grid shape."""
    # ── identical to IN1kLatentDataset.__getitem__ grid construction ──────────
    hs = torch.arange(H_g, dtype=dtype)
    ws = torch.arange(W_g, dtype=dtype)
    gh, gw = torch.meshgrid(hs, ws, indexing='ij')
    grid = torch.stack([gw.reshape(-1), gh.reshape(-1)])
    # ─────────────────────────────────────────────────────────────────────────
    grid = grid.unsqueeze(0).repeat(B, 1, 1).to(device=device, dtype=dtype)
    mask = torch.ones(B, H_g * W_g, device=device, dtype=dtype)
    size = torch.tensor((H_g, W_g), dtype=torch.int32, device=device).repeat(B, 1).unsqueeze(1)
    return grid, mask, size


def _single_grid(H_g: int, W_g: int, device, dtype) -> torch.Tensor:
    """(2, H_g*W_g) integer grid, w-fast/h-slow — matches the dataset convention."""
    hs = torch.arange(H_g, dtype=dtype)
    ws = torch.arange(W_g, dtype=dtype)
    gh, gw = torch.meshgrid(hs, ws, indexing='ij')
    return torch.stack([gw.reshape(-1), gh.reshape(-1)]).to(device=device, dtype=dtype)


def build_upsampler_inputs(x_lr_sp, x_fr_sp, y_full, k_grid, H_fr, W_fr, device, dtype):
    """Assemble the model's two-resolution forward inputs for one CFG-doubled batch.

    Each batch element is one image in the pack (n_pack = batch size). The
    low-res side is packed into a single B=1 sequence (doc_ids + block_mask);
    the full-res side is a dense (n_pack, N_fr, ·) batch — exactly the layout
    packed_collate_fn produces for return_fullres data.

    All sizes here are *grid* (token) sizes, not spatial: a k_grid×k_grid grid
    has k_grid² tokens and corresponds to a (k_grid·p)×(k_grid·p) latent.

    Args:
        x_lr_sp: (n_pack, C, k_grid*p, k_grid*p)   low-res noisy latent (spatial)
        x_fr_sp: (n_pack, C, H_fr*p, W_fr*p)       full-res noisy latent (spatial)
        y_full:  (n_pack,)                         labels (cond + null concatenated)
        k_grid:  low-res grid size
    Returns dict of model_kwargs plus the packed low-res token tensor `x`.
    """
    from torch.nn.attention.flex_attention import create_block_mask

    n_pack = x_lr_sp.shape[0]
    seq_lr = k_grid * k_grid
    N_total = math.ceil((n_pack * seq_lr) / PAD_TO_MULTIPLE) * PAD_TO_MULTIPLE

    # --- packed low-res tokens ---
    x_lr_tok = patchify(x_lr_sp, PATCH_SIZE)          # (n_pack, seq_lr, p**2*C)
    Dtok = x_lr_tok.shape[-1]
    feat = torch.zeros(1, N_total, Dtok, device=device, dtype=dtype)
    grid = torch.zeros(1, 2, N_total, device=device, dtype=torch.long)
    mask = torch.zeros(1, N_total, device=device, dtype=dtype)
    doc_ids = torch.full((1, N_total), -1, device=device, dtype=torch.int32)
    size_lr = torch.zeros(1, n_pack, 2, device=device, dtype=torch.int32)
    t_pack = torch.zeros(1, n_pack, device=device, dtype=dtype)  # filled by caller's t
    y_pack = y_full.to(torch.int).view(1, n_pack)

    g_lr = _single_grid(k_grid, k_grid, device, torch.long)     # (2, seq_lr)
    offset = 0
    for img_idx in range(n_pack):
        feat[0, offset:offset + seq_lr] = x_lr_tok[img_idx]
        grid[0, :, offset:offset + seq_lr] = g_lr
        mask[0, offset:offset + seq_lr] = 1
        doc_ids[0, offset:offset + seq_lr] = img_idx
        size_lr[0, img_idx] = torch.tensor([k_grid, k_grid], device=device)
        offset += seq_lr

    def doc_mask_mod(b, h, q_idx, kv_idx):
        return doc_ids[b, q_idx] == doc_ids[b, kv_idx]
    block_mask = create_block_mask(doc_mask_mod, 1, None, N_total, N_total, device=device)

    # --- dense full-res inputs ---
    x_fr_tok = patchify(x_fr_sp, PATCH_SIZE)          # (n_pack, N_fr, p**2*C)
    grid_fr = _single_grid(H_fr, W_fr, device, torch.long)        # (2, N_fr)
    grid_fr = grid_fr.unsqueeze(0).repeat(n_pack, 1, 1)          # (n_pack, 2, N_fr)
    mask_fr = torch.ones(n_pack, H_fr * W_fr, device=device, dtype=dtype)
    size_fr = torch.tensor([H_fr, W_fr], device=device, dtype=torch.int32)
    size_fr = size_fr.view(1, 1, 2).repeat(1, n_pack, 1)         # (1, n_pack, 2)

    return dict(
        x=feat, grid=grid, mask=mask, size=size_lr, doc_ids=doc_ids,
        block_mask=block_mask, y=y_pack, t_pack=t_pack,
        x_fullres=x_fr_tok, grid_fullres=grid_fr, mask_fullres=mask_fr,
        size_fullres=size_fr,
    )


def build_upsampler_inputs_quadtree(cell_blocks, qt, x_fr_sp, y_full, H_fr, W_fr, device, dtype):
    """Quadtree counterpart of :func:`build_upsampler_inputs`.

    Every image in the pack shares one quadtree `qt`. Per image, the low-res
    tokens are the concatenation of its cells' patchified blocks (cell order),
    forming one document; RoPE uses the physical-center grid so coarse and fine
    cells live in one coordinate frame. The full-res side is dense as before.

    Args:
        cell_blocks: list (len = len(qt.cells)) of (n_pack, C, k_h*p, k_w*p)
            low-res noisy spatial latents, one per cell, already at the cell's
            own resolution (built by the sampler).
        qt:    Quadtree shared by all n_pack images.
        x_fr_sp: (n_pack, C, H_fr*p, W_fr*p) full-res noisy latent.
        y_full:  (n_pack,) labels (cond + null concatenated).
    Returns dict of model_kwargs (including `cells` for FiT._upsample_quadtree).
    """
    from torch.nn.attention.flex_attention import create_block_mask
    from fit.noise_field_sampler.quadtree import quadtree_grid

    n_pack = x_fr_sp.shape[0]
    seq_lr = qt.n_tokens                                    # tokens per image
    N_total = math.ceil((n_pack * seq_lr) / PAD_TO_MULTIPLE) * PAD_TO_MULTIPLE

    # Per-image packed low-res tokens: concatenate each cell's patchified tokens
    # in cell order (matches Quadtree.placement_cells offsets).
    per_cell_tok = [patchify(blk, PATCH_SIZE) for blk in cell_blocks]  # each (n_pack, k_h*k_w, Dtok)
    tok_img = torch.cat(per_cell_tok, dim=1)               # (n_pack, seq_lr, Dtok)
    Dtok = tok_img.shape[-1]

    feat = torch.zeros(1, N_total, Dtok, device=device, dtype=dtype)
    grid = torch.zeros(1, 2, N_total, device=device, dtype=dtype)   # float: physical coords
    mask = torch.zeros(1, N_total, device=device, dtype=dtype)
    doc_ids = torch.full((1, N_total), -1, device=device, dtype=torch.int32)
    # RoPE scale uses size.max over the pack; the physical grid spans the full
    # frame, so report each image's size as the full-res size.
    size_lr = torch.tensor([H_fr, W_fr], dtype=torch.int32, device=device)
    size_lr = size_lr.view(1, 1, 2).repeat(1, n_pack, 1)   # (1, n_pack, 2)
    y_pack = y_full.to(torch.int).view(1, n_pack)
    t_pack = torch.zeros(1, n_pack, device=device, dtype=dtype)

    g_phys = quadtree_grid(qt, device, dtype)              # (2, seq_lr) physical centers
    offset = 0
    for img_idx in range(n_pack):
        feat[0, offset:offset + seq_lr] = tok_img[img_idx]
        grid[0, :, offset:offset + seq_lr] = g_phys
        mask[0, offset:offset + seq_lr] = 1
        doc_ids[0, offset:offset + seq_lr] = img_idx
        offset += seq_lr

    def doc_mask_mod(b, h, q_idx, kv_idx):
        return doc_ids[b, q_idx] == doc_ids[b, kv_idx]
    block_mask = create_block_mask(doc_mask_mod, 1, None, N_total, N_total, device=device)

    # --- dense full-res inputs (unchanged) ---
    x_fr_tok = patchify(x_fr_sp, PATCH_SIZE)               # (n_pack, N_fr, Dtok)
    grid_fr = _single_grid(H_fr, W_fr, device, torch.long)
    grid_fr = grid_fr.unsqueeze(0).repeat(n_pack, 1, 1)
    mask_fr = torch.ones(n_pack, H_fr * W_fr, device=device, dtype=dtype)
    size_fr = torch.tensor([H_fr, W_fr], device=device, dtype=torch.int32)
    size_fr = size_fr.view(1, 1, 2).repeat(1, n_pack, 1)

    # One cell list per image (all share the same quadtree).
    cells = [qt.placement_cells() for _ in range(n_pack)]

    return dict(
        x=feat, grid=grid, mask=mask, size=size_lr, doc_ids=doc_ids,
        block_mask=block_mask, y=y_pack, t_pack=t_pack,
        x_fullres=x_fr_tok, grid_fullres=grid_fr, mask_fullres=mask_fr,
        size_fullres=size_fr, cells=cells,
    )

# ──────────────────────────── main ───────────────────────────────────────────

def main():
    if os.path.isdir(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Record the schedules used so they travel inside the output zip.
    schedules_path = os.path.join(OUTPUT_DIR, "schedules.json")
    with open(schedules_path, "w") as f:
        json.dump({f"noise_field_{idx:03d}": grid_sizes
                   for idx, grid_sizes in SCHEDULES.items()}, f, indent=2)
    print(f"Wrote schedules to {schedules_path}")

    torch.manual_seed(GLOBAL_SEED)
    print(f"Device: {DEVICE}")
    print(f"Generating {N_IMAGES} images per (checkpoint, schedule) pair")
    print(f"CFG={CFG_SCALE}")

    # Full-res grid for 256×256 images: latent = 32×32, grid = 16×16.
    H_fr = TARGET_LEN_PIX // (PATCH_SIZE * VAE_SCALE)
    W_fr = TARGET_LEN_PIX // (PATCH_SIZE * VAE_SCALE)
    print(f"Full-res grid: {H_fr}×{W_fr}")

    # Pre-generate fixed labels shared across all checkpoints. The noise-field
    # sampler draws its own noise fields internally (seeded per batch below).
    all_labels = torch.randint(0, NUM_CLASSES, (N_IMAGES,), device=DEVICE)
    print(f"Fixed labels pre-generated (seed={GLOBAL_SEED}).")

    # VAE for decoding.
    print(f"\nLoading VAE from {VAE_PATH} …")
    vae = AutoencoderKL.from_pretrained(VAE_PATH).to(DEVICE).eval()

    for ckpt_cfg in CHECKPOINTS:
        ckpt_name = ckpt_cfg["name"]
        ckpt_dir  = ckpt_cfg["dir"]
        loss_type = ckpt_cfg["loss_type"]
        use_ema   = ckpt_cfg.get("use_ema", None)  # None → plain file, True/False → accel dir

        # Resolve the actual .safetensors file.
        if os.path.isfile(ckpt_dir):
            # Plain file (e.g. baseline model_ema.safetensors).
            ckpt_path = ckpt_dir
        else:
            # Accelerate checkpoint directory: pick ema or train weights.
            fname = "model_1.safetensors" if use_ema else "model.safetensors"
            ckpt_path = os.path.join(ckpt_dir, fname)

        print(f"\n{'='*60}")
        print(f"Checkpoint: {ckpt_name}  loss={loss_type}")
        print(f"  {ckpt_path}")
        if not os.path.isfile(ckpt_path):
            print("  [skip] file not found")
            continue

        model = load_model(ckpt_path, model_cfg_for(loss_type), DEVICE)
        dtype = next(model.parameters()).dtype

        # --- noise-field sampler (progressive resolution with consistent noise) ---
        for sched_idx, grid_sizes in SCHEDULES.items():
            out_dir = os.path.join(OUTPUT_DIR, ckpt_name, f"noise_field_{sched_idx:03d}")
            os.makedirs(out_dir, exist_ok=True)
            print(f"\n  Noise-field {sched_idx:03d} {grid_sizes}  →  {out_dir}")

            generated = 0
            while generated < N_IMAGES:
                bs = min(BATCH_SIZE, N_IMAGES - generated)
                y = all_labels[generated:generated + bs]
                y_null = torch.full_like(y, NUM_CLASSES)
                y_full = torch.cat([y, y_null], 0)        # (2*bs,) cond + null

                # Deterministic per batch: sampler draws its own noise fields internally.
                # The sampler treats schedule entries as spatial sizes; convert from
                # packed grid sizes by multiplying with PATCH_SIZE.
                torch.manual_seed(GLOBAL_SEED + generated)

                if loss_type == "C" and USE_QUADTREE:
                    # Mixed-resolution (quadtree) learned-upsampler path. The
                    # state stays full-res; the low-res conditioning is built
                    # per-cell at varying densities. model_fn receives the list
                    # of per-cell spatial blocks + the quadtree and returns the
                    # full-res velocity.
                    qt = build_refinement_quadtree(
                        H_fr, W_fr, QUADTREE["base_k"], QUADTREE["refine"]
                    )

                    def model_fn_qt(cell_blocks, x_fr_sp, t, qt):
                        # CFG-double the full-res input, the per-cell blocks, and
                        # the labels (cond + null), mirroring build_upsampler_inputs.
                        x_fr2 = torch.cat([x_fr_sp, x_fr_sp], 0)
                        cell_blocks2 = [torch.cat([cb, cb], 0) for cb in cell_blocks]
                        kw = build_upsampler_inputs_quadtree(
                            cell_blocks2, qt, x_fr2, y_full, H_fr, W_fr, DEVICE, dtype
                        )
                        n_pack = x_fr2.shape[0]
                        t_pack = t[:1].repeat(1, n_pack)
                        v = model(
                            kw['x'], t_pack, kw['y'], kw['grid'], kw['mask'],
                            kw['size'], doc_ids=kw['doc_ids'], block_mask=kw['block_mask'],
                            x_fullres=kw['x_fullres'], grid_fullres=kw['grid_fullres'],
                            mask_fullres=kw['mask_fullres'], size_fullres=kw['size_fullres'],
                            cells=kw['cells'],
                        )                                   # (n_pack, N_fr, p**2*C)
                        v_cond, v_uncond = v.chunk(2, dim=0)
                        v_pred = v_uncond + CFG_SCALE * (v_cond - v_uncond)
                        return unpatchify(v_pred, (H_fr * PATCH_SIZE, W_fr * PATCH_SIZE), PATCH_SIZE)

                    x1_sp = noise_field_sample_upsampler_quadtree(
                        model_fn_qt,
                        qt,
                        num_steps=QUADTREE_STEPS,
                        b=bs,
                        d=C_IN,
                        patch_size=PATCH_SIZE,
                        per_leaf_sigma=QUADTREE_PER_LEAF_SIGMA,
                        device=DEVICE,
                        dtype=dtype,
                    )
                elif loss_type == "C":
                    # Learned-upsampler model: the integration state stays full-res;
                    # the schedule size only sets the resolution of the low-res
                    # conditioning input. model_fn receives both resolutions and
                    # always returns a full-res velocity.
                    def model_fn(x_lr_sp, x_fr_sp, t, k):
                        # k is a *spatial* size (schedule entries are spatial);
                        # convert to the low-res grid size for the packed branch.
                        k_grid = k // PATCH_SIZE
                        # CFG-double both resolutions and the labels.
                        x_lr2 = torch.cat([x_lr_sp, x_lr_sp], 0)
                        x_fr2 = torch.cat([x_fr_sp, x_fr_sp], 0)
                        kw = build_upsampler_inputs(
                            x_lr2, x_fr2, y_full, k_grid, H_fr, W_fr, DEVICE, dtype
                        )
                        n_pack = x_lr2.shape[0]
                        t_pack = t[:1].repeat(1, n_pack)   # shared timestep per image
                        v = model(
                            kw['x'], t_pack, kw['y'], kw['grid'], kw['mask'],
                            kw['size'], doc_ids=kw['doc_ids'], block_mask=kw['block_mask'],
                            x_fullres=kw['x_fullres'], grid_fullres=kw['grid_fullres'],
                            mask_fullres=kw['mask_fullres'], size_fullres=kw['size_fullres'],
                        )                                   # (n_pack, N_fr, p**2*C)
                        v_cond, v_uncond = v.chunk(2, dim=0)
                        v_pred = v_uncond + CFG_SCALE * (v_cond - v_uncond)
                        return unpatchify(v_pred, (H_fr * PATCH_SIZE, W_fr * PATCH_SIZE), PATCH_SIZE)

                    x1_sp = noise_field_sample_upsampler(
                        model_fn,
                        scale_schedule=[g * PATCH_SIZE for g in grid_sizes],
                        b=bs,
                        d=C_IN,
                        device=DEVICE,
                        dtype=dtype,
                    )
                else:
                    def model_fn(x_sp: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
                        H_g = x_sp.shape[-2] // PATCH_SIZE
                        W_g = x_sp.shape[-1] // PATCH_SIZE
                        grid, mask, size = make_grid_and_mask(H_g, W_g, bs, DEVICE, dtype)
                        x_tok = patchify(x_sp, PATCH_SIZE)
                        v = model(
                            torch.cat([x_tok, x_tok], 0),
                            torch.cat([t, t], 0),
                            torch.cat([y, y_null], 0),
                            torch.cat([grid, grid], 0),
                            torch.cat([mask, mask], 0),
                            torch.cat([size, size], 0),
                        )
                        v_cond, v_uncond = v.chunk(2, dim=0)
                        v_pred = v_uncond + CFG_SCALE * (v_cond - v_uncond)
                        return unpatchify(v_pred, (H_g * PATCH_SIZE, W_g * PATCH_SIZE), PATCH_SIZE)

                    x1_sp = noise_field_sample(
                        model_fn,
                        scale_schedule=[g * PATCH_SIZE for g in grid_sizes],
                        b=bs,
                        d=C_IN,
                        device=DEVICE,
                        dtype=dtype,
                    )

                with torch.no_grad():
                    imgs = vae.decode(x1_sp / vae.config.scaling_factor).sample
                imgs = torch.clamp(127.5 * imgs + 128.0, 0, 255)
                imgs = imgs.permute(0, 2, 3, 1).to(torch.uint8).cpu().numpy()
                for i, img_arr in enumerate(imgs):
                    Image.fromarray(img_arr).save(os.path.join(out_dir, f"{generated + i:06d}.png"))
                generated += bs
                print(f"    {generated}/{N_IMAGES}", end="\r", flush=True)
            print(f"    {N_IMAGES}/{N_IMAGES}  done")

        del model

    zip_path = OUTPUT_DIR.rstrip("/") + ".zip"
    print(f"\nZipping {OUTPUT_DIR}/ → {zip_path} …")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for fpath in sorted(Path(OUTPUT_DIR).rglob("*")):
            if fpath.is_file():
                zf.write(fpath, fpath.relative_to(OUTPUT_DIR))
    print(f"Saved {zip_path}")


if __name__ == "__main__":
    main()
