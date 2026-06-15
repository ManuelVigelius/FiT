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

sys.path.insert(0, str(Path(__file__).parent))

from fit.model.fit_model import FiT
from fit.noise_field_sampler.noise_field_sampler import sample as noise_field_sample
from fit.scheduler.transport.utils import patchify, unpatchify


# ──────────────────────────── helpers ────────────────────────────────────────

def model_cfg_for(loss_type: str) -> dict:
    cfg = dict(_BASE_MODEL_CFG)
    cfg["use_size_cond"] = (loss_type not in ("baseline", "virtual_resize"))
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

                # Deterministic per batch: sampler draws its own noise fields internally.
                # The sampler treats schedule entries as spatial sizes; convert from
                # packed grid sizes by multiplying with PATCH_SIZE.
                torch.manual_seed(GLOBAL_SEED + generated)
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
