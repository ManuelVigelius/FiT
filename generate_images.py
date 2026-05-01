"""
Image generation script: generates 128 images per (model, compression) pair using
a 10-step Euler ODE sampler with CFG=4.0.

The sampling mirrors the full-res velocity error evaluation in eval_losses.py:
  - For baseline / virtual_resize: the model runs at full resolution (16×16 grid),
    noise is sampled at full-res, and the trajectory integrates v_pred at full-res.
  - For loss A/B: the model runs at the compressed grid with size conditioning;
    the predicted clean latent is bilinearly upsampled to full-res before decoding.
  - For loss C: as A/B but the ResNet upsampler refines the predicted clean latent.

Generated images are saved as PNG under:
  <OUTPUT_DIR>/<ckpt_name>/grid_<g>x<g>/<idx:06d>.png

All configuration lives in the CONFIG block below — no CLI arguments needed.
"""

import os
import sys
import zipfile
from pathlib import Path

import torch
import torch.nn.functional as F

from PIL import Image
from safetensors.torch import load_file
from diffusers.models import AutoencoderKL

# ─────────────────────────────── CONFIG ─────────────────────────────────────

# Compression grid sizes to generate at.  Each entry is a square grid side-length.
# The full-res grid for 256×256 images is 16×16 (spatial latent 32×32, patch 2×2).
COMPRESSIONS = [2, 8, 16]

# Number of images to generate per (checkpoint, compression) pair.
N_IMAGES = 4

# Batch size for the generation loop (single GPU).
BATCH_SIZE = 32

# Number of Euler ODE steps.
N_STEPS = 10

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
OUTPUT_DIR = "generated_images"

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
CHECKPOINTS = [
    dict(
        name="baseline",
        dir="/content/drive/MyDrive/FiT/inference_weights/checkpoint-baseline",
        loss_type="baseline",
    ),
    dict(
        name="loss_a_8k",
        dir="/content/drive/MyDrive/FiT/inference_weights/checkpoint-8000-bs8k",
        loss_type="A",
    ),
    dict(
        name="loss_c_6k",
        dir="/content/drive/MyDrive/FiT/inference_weights/checkpoint-6000-bs8k-lossc",
        loss_type="C",
    ),
]

# ─────────────────────────────────────────────────────────────────────────────

PATCH_SIZE = 2
VAE_SCALE  = 8   # SD VAE 8× spatial downsampling
C_IN       = 4   # VAE latent channels

sys.path.insert(0, str(Path(__file__).parent))

from fit.model.fit_model import FiT
from fit.scheduler.transport.utils import patchify, unpatchify, spatial_resize as _spatial_resize


# ──────────────────────────── helpers ────────────────────────────────────────

def spatial_resize(x: torch.Tensor, H: int, W: int,
                   H_out: int, W_out: int,
                   patch_size: int = PATCH_SIZE,
                   mode: str = 'bilinear') -> torch.Tensor:
    if H == H_out and W == W_out:
        return x
    return _spatial_resize(x, H, W, H_out, W_out, patch_size, mode)


def spatial_resize_sp(x_sp: torch.Tensor,
                      H_out: int, W_out: int,
                      patch_size: int = PATCH_SIZE) -> torch.Tensor:
    """Bilinear resize of an unpatchified spatial tensor (B, C, H*p, W*p)."""
    if x_sp.shape[-2] == H_out * patch_size and x_sp.shape[-1] == W_out * patch_size:
        return x_sp
    return F.interpolate(x_sp.float(), size=(H_out * patch_size, W_out * patch_size),
                         mode="bilinear", align_corners=True).to(x_sp.dtype)


def model_cfg_for(loss_type: str) -> dict:
    cfg = dict(_BASE_MODEL_CFG)
    cfg["use_size_cond"] = (loss_type not in ("baseline", "virtual_resize"))
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
    grid_h = torch.arange(H_g, dtype=torch.long)
    grid_w = torch.arange(W_g, dtype=torch.long)
    grid = torch.meshgrid(grid_w, grid_h, indexing='xy')
    grid = torch.cat([grid[0].reshape(1, -1),
                      grid[1].reshape(1, -1)], dim=0)
    grid = grid.repeat(B, 1, 1).to(device=device, dtype=dtype)
    mask = torch.ones(B, H_g * W_g, device=device, dtype=dtype)
    size = torch.tensor((H_g, W_g), dtype=torch.int32, device=device).repeat(B, 1).unsqueeze(1)
    return grid, mask, size

# ──────────────── Euler ODE sampler (10 steps, noise→data) ───────────────────

@torch.no_grad()
def euler_sample(
    model: FiT,
    z: torch.Tensor,              # (B, seq, 16)  initial noise at t=1
    y: torch.Tensor,              # (B,) class labels
    H_g: int, W_g: int,          # generation grid
    H_fr: int, W_fr: int,        # full-res grid
    n_steps: int,
    cfg_scale: float,
    loss_type: str,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Run a simple Euler ODE from t=1 (pure noise) to t=0 (clean image).

    For baseline the trajectory runs at full resolution (H_fr×W_fr).
    For A/B/C the trajectory runs at the compressed grid (H_g×W_g).

    Returns the predicted clean latent at full-res as a spatial tensor
    (B, C_IN, H_fr*p, W_fr*p).
    """
    p = PATCH_SIZE
    B = z.shape[0]
    r = H_g / H_fr  # downsampling ratio (1.0 for baseline)

    use_fr = (loss_type in ("baseline", "C"))
    H_run = H_fr if use_fr else H_g
    W_run = W_fr if use_fr else W_g

    y_null = torch.full_like(y, NUM_CLASSES)

    # Loss C: trajectory lives at full-res in spatial form; LR grid used only for model input.
    # Other losses: trajectory lives in token form at H_run×W_run.
    if loss_type == "C":
        xt_fr_sp = z  # (B, C_IN, H_fr*p, W_fr*p) full-res spatial noise
        grid_lr, mask_lr, size_lr = make_grid_and_mask(H_g, W_g, B, device, dtype)
    else:
        x = z  # (B, H_run*W_run, p**2*C_IN) token-form noise
        grid, mask, size = make_grid_and_mask(H_run, W_run, B, device, dtype)

    # Forward time: t=0 (noise) → t=1 (data), matching the training convention
    # where the model sees t=0 as noise and t=1 as clean data.
    ts = torch.linspace(0.0, 1.0, n_steps + 1, device=device, dtype=dtype)

    for i in range(n_steps):
        t_cur  = ts[i]
        t_next = ts[i + 1]
        dt     = t_next - t_cur  # positive

        sigma = (1.0 - t_cur).expand(B).to(dtype)          # (B,) noise weight, 1→0

        # Noise-corrected timestep fed to the model.
        # For compressed grids the effective SNR shifts; sigma_inj corrects for it.
        # At full resolution (r==1) sigma_inj == sigma, so t_model == t_cur.
        if r < 1.0:
            sigma_inj = sigma / (r + sigma * (1.0 - r))
        else:
            sigma_inj = sigma
        t_model = 1.0 - sigma_inj                           # (B,) == t_cur for baseline

        if loss_type == "C":
            x_lr_sp = F.interpolate(xt_fr_sp.float(), size=(H_g * p, W_g * p),
                                    mode='area').to(xt_fr_sp.dtype)
            x_lr = patchify(x_lr_sp, p)
            v = model(
                torch.cat([x_lr, x_lr], 0),
                torch.cat([t_model, t_model], 0),
                torch.cat([y, y_null], 0),
                torch.cat([grid_lr, grid_lr], 0),
                torch.cat([mask_lr, mask_lr], 0),
                torch.cat([size_lr, size_lr], 0),
            )
        else:
            v = model(
                torch.cat([x, x], 0),
                torch.cat([t_model, t_model], 0),
                torch.cat([y, y_null], 0),
                torch.cat([grid, grid], 0),
                torch.cat([mask, mask], 0),
                torch.cat([size, size], 0),
            )
        v_cond, v_uncond = v.chunk(2, dim=0)

        # CFG on first 3*patch**2 channels only, matching model.forward_with_cfg
        C_cfg = 3 * p * p
        v_pred = torch.cat([
            v_uncond[:, :, :C_cfg] + cfg_scale * (v_cond[:, :, :C_cfg] - v_uncond[:, :, :C_cfg]),
            v_cond[:, :, C_cfg:],
        ], dim=2)

        if loss_type == "C":
            v_lr_sp    = unpatchify(v_pred, (H_g * p, W_g * p), p)
            v_lr_up_sp = F.interpolate(v_lr_sp.float(), size=(H_fr * p, W_fr * p),
                                       mode='bilinear', align_corners=True).to(v_lr_sp.dtype)
            v_pred_fr_sp = model.upsampler(v_lr_up_sp, xt_fr_sp)
            xt_fr_sp     = xt_fr_sp + dt * v_pred_fr_sp
        else:
            x = x + dt * v_pred

    # ── convert final state to full-res spatial latent ───────────────────────
    if loss_type == "C":
        return unpatchify(patchify(xt_fr_sp, p), (H_fr * p, W_fr * p), p)

    if use_fr:
        return unpatchify(x, (H_fr * p, W_fr * p), p)

    # Loss A/B: upsample the clean prediction (not the velocity).
    x1_hat_lr_sp = unpatchify(x, (H_run * p, W_run * p), p)
    return spatial_resize_sp(x1_hat_lr_sp, H_fr, W_fr)


# ──────────────────────────── main ───────────────────────────────────────────

def main():
    torch.manual_seed(GLOBAL_SEED)
    print(f"Device: {DEVICE}")
    print(f"Generating {N_IMAGES} images per (checkpoint, compression) pair")
    print(f"CFG={CFG_SCALE}, steps={N_STEPS}")

    # Full-res grid for 256×256 images: latent = 32×32, grid = 16×16.
    H_fr = TARGET_LEN_PIX // (PATCH_SIZE * VAE_SCALE)
    W_fr = TARGET_LEN_PIX // (PATCH_SIZE * VAE_SCALE)
    print(f"Full-res grid: {H_fr}×{W_fr}")

    # VAE for decoding.
    print(f"\nLoading VAE from {VAE_PATH} …")
    vae = AutoencoderKL.from_pretrained(VAE_PATH).to(DEVICE).eval()

    for ckpt_cfg in CHECKPOINTS:
        ckpt_name = ckpt_cfg["name"]
        ckpt_dir  = ckpt_cfg["dir"]
        loss_type = ckpt_cfg["loss_type"]

        ckpt_path = os.path.join(ckpt_dir, "model_1.safetensors")
        print(f"\n{'='*60}")
        print(f"Checkpoint: {ckpt_name}  loss={loss_type}")
        print(f"  {ckpt_path}")
        if not os.path.isfile(ckpt_path):
            print("  [skip] file not found")
            continue

        model = load_model(ckpt_path, model_cfg_for(loss_type), DEVICE)
        dtype = next(model.parameters()).dtype

        for g in COMPRESSIONS:
            H_g = g; W_g = g
            out_dir = os.path.join(OUTPUT_DIR, ckpt_name, f"grid_{g}x{g}")
            os.makedirs(out_dir, exist_ok=True)
            print(f"\n  Grid {g}×{g}  →  {out_dir}")

            generated = 0
            while generated < N_IMAGES:
                bs = min(BATCH_SIZE, N_IMAGES - generated)

                # Random labels.
                y = torch.randint(0, NUM_CLASSES, (bs,), device=DEVICE)

                # Initial noise.
                # Loss C: full-res spatial tensor (B, C_IN, H_fr*p, W_fr*p).
                # Baseline: token form at full-res. A/B: token form at compressed grid.
                if loss_type == "C":
                    z = torch.randn(bs, C_IN, H_fr * PATCH_SIZE, W_fr * PATCH_SIZE,
                                    device=DEVICE, dtype=dtype)
                else:
                    use_fr_noise = (loss_type == "baseline")
                    H_noise = H_fr if use_fr_noise else H_g
                    W_noise = W_fr if use_fr_noise else W_g
                    z = torch.randn(bs, H_noise * W_noise, PATCH_SIZE**2 * C_IN,
                                    device=DEVICE, dtype=dtype)

                # Run Euler sampler → (B, C_IN, H_fr*p, W_fr*p) spatial latent.
                x1_sp = euler_sample(
                    model=model,
                    z=z,
                    y=y,
                    H_g=H_g, W_g=W_g,
                    H_fr=H_fr, W_fr=W_fr,
                    n_steps=N_STEPS,
                    cfg_scale=CFG_SCALE,
                    loss_type=loss_type,
                    device=DEVICE,
                    dtype=dtype,
                )  # (B, 4, 32, 32)

                # Decode with VAE.
                with torch.no_grad():
                    imgs = vae.decode(x1_sp / vae.config.scaling_factor).sample
                imgs = torch.clamp(127.5 * imgs + 128.0, 0, 255)
                imgs = imgs.permute(0, 2, 3, 1).to(torch.uint8).cpu().numpy()

                for i, img_arr in enumerate(imgs):
                    idx = generated + i
                    Image.fromarray(img_arr).save(os.path.join(out_dir, f"{idx:06d}.png"))

                generated += bs
                print(f"    {generated}/{N_IMAGES}", end="\r", flush=True)

            print(f"    {N_IMAGES}/{N_IMAGES}  done")

        del model

    zip_path = OUTPUT_DIR + ".zip"
    print(f"\nZipping {OUTPUT_DIR}/ → {zip_path} …")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for fpath in sorted(Path(OUTPUT_DIR).rglob("*")):
            if fpath.is_file():
                zf.write(fpath, fpath.relative_to(OUTPUT_DIR))
    print(f"Saved {zip_path}")


if __name__ == "__main__":
    main()
