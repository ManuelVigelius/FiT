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
from einops import rearrange
from PIL import Image
from safetensors.torch import load_file
from diffusers.models import AutoencoderKL

# ─────────────────────────────── CONFIG ─────────────────────────────────────

# Compression grid sizes to generate at.  Each entry is a square grid side-length.
# The full-res grid for 256×256 images is 16×16 (spatial latent 32×32, patch 2×2).
COMPRESSIONS = [2, 4, 6, 8, 10, 12, 14, 16]

# Number of images to generate per (checkpoint, compression) pair.
N_IMAGES = 128

# Batch size for the generation loop (single GPU).
BATCH_SIZE = 128

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
    use_sit=True,
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
        name="loss_c_8k",
        dir="/content/drive/MyDrive/FiT/inference_weights/checkpoint-8000-bs8k-lossc",
        loss_type="C",
    ),
]

# ─────────────────────────────────────────────────────────────────────────────

PATCH_SIZE = 2
VAE_SCALE  = 8   # SD VAE 8× spatial downsampling
C_IN       = 4   # VAE latent channels

sys.path.insert(0, str(Path(__file__).parent))

from fit.model.fit_model import FiT


# ──────────────────────────── helpers ────────────────────────────────────────

def spatial_resize(x: torch.Tensor, H: int, W: int,
                   H_out: int, W_out: int,
                   patch_size: int = PATCH_SIZE, C_in: int = C_IN,
                   mode: str = 'bilinear') -> torch.Tensor:
    """Resize a patchified token sequence (B, H*W, patch_size**2*C_in)."""
    p = patch_size
    sp = rearrange(x, "b (h w) (p1 p2 c) -> b c (h p1) (w p2)",
                   h=H, w=W, p1=p, p2=p, c=C_in)
    kwargs = {} if mode == 'area' else {'align_corners': True}
    sp = F.interpolate(sp.float(), size=(H_out * p, W_out * p),
                       mode=mode, **kwargs).to(x.dtype)
    return rearrange(sp, "b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=p, p2=p)


def spatial_resize_sp(x_sp: torch.Tensor,
                      H_out: int, W_out: int,
                      patch_size: int = PATCH_SIZE) -> torch.Tensor:
    """Bilinear resize of an unpatchified spatial tensor (B, C, H*p, W*p)."""
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
    seq = H_g * W_g
    hs = torch.arange(H_g, dtype=dtype, device=device)
    ws = torch.arange(W_g, dtype=dtype, device=device)
    gh, gw = torch.meshgrid(hs, ws, indexing="ij")
    grid = torch.zeros(B, 2, seq, dtype=dtype, device=device)
    grid[:, 0] = gh.reshape(-1).unsqueeze(0).expand(B, -1)
    grid[:, 1] = gw.reshape(-1).unsqueeze(0).expand(B, -1)
    mask = torch.ones(B, seq, dtype=torch.uint8, device=device)
    size = torch.tensor([[H_g, W_g]], dtype=torch.int32, device=device).expand(B, -1).unsqueeze(1)
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

    use_fr = (loss_type == "baseline")
    H_run = H_fr if use_fr else H_g
    W_run = W_fr if use_fr else W_g

    y_null = torch.full_like(y, NUM_CLASSES)
    grid, mask, size = make_grid_and_mask(H_run, W_run, B, device, dtype)
    mask_f = mask.float()

    x = z  # (B, seq_run, 16), t=1 pure noise
    # Loss C tracks the full-res spatial trajectory in parallel.
    if loss_type == "C":
        xt_fr_sp = spatial_resize_sp(
            rearrange(x, "b (h w) (p1 p2 c) -> b c (h p1) (w p2)",
                      h=H_g, w=W_g, p1=p, p2=p, c=C_IN),
            H_fr, W_fr,
        )

    ts = torch.linspace(1.0, 0.0, n_steps + 1, device=device, dtype=dtype)

    for i in range(n_steps):
        t_cur  = ts[i]
        t_next = ts[i + 1]
        dt     = t_next - t_cur  # negative

        sigma = (1.0 - t_cur).expand(B).to(dtype)          # (B,)

        # Noise-corrected timestep fed to the model (matches eval_losses.py).
        if r < 1.0:
            sigma_inj = sigma / (r + sigma * (1.0 - r))
        else:
            sigma_inj = sigma
        t_model = 1.0 - sigma_inj                           # (B,)

        v = model(
            torch.cat([x, x], 0),
            torch.cat([t_model, t_model], 0),
            torch.cat([y, y_null], 0),
            torch.cat([grid, grid], 0),
            torch.cat([mask_f, mask_f], 0),
            torch.cat([size, size], 0),
        )
        v_cond, v_uncond = v.chunk(2, dim=0)
        v_pred_lr = v_uncond + cfg_scale * (v_cond - v_uncond)

        if loss_type == "C":
            v_lr_sp      = rearrange(v_pred_lr, "b (h w) (p1 p2 c) -> b c (h p1) (w p2)",
                                     h=H_g, w=W_g, p1=p, p2=p, c=C_IN)
            v_lr_up_sp   = spatial_resize_sp(v_lr_sp, H_fr, W_fr)
            v_pred_fr_sp = model.upsampler(v_lr_up_sp, xt_fr_sp)
            xt_fr_sp     = xt_fr_sp + dt * v_pred_fr_sp

        x = x + dt * v_pred_lr

    # ── convert final state to full-res spatial latent ───────────────────────
    if use_fr:
        return rearrange(x, "b (h w) (p1 p2 c) -> b c (h p1) (w p2)",
                         h=H_fr, w=W_fr, p1=p, p2=p, c=C_IN)

    if loss_type == "C":
        return xt_fr_sp

    # Loss A/B: upsample the clean prediction (not the velocity).
    x1_hat_lr_sp = rearrange(x, "b (h w) (p1 p2 c) -> b c (h p1) (w p2)",
                              h=H_g, w=W_g, p1=p, p2=p, c=C_IN)
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
    vae = AutoencoderKL.from_pretrained(VAE_PATH, local_files_only=True).to(DEVICE).eval()

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

                # Initial noise at the generation grid.
                use_fr_noise = loss_type in ("baseline", "virtual_resize")
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
