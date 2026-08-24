"""
Evaluation script: measures velocity loss and image-MSE at evenly-spaced
timesteps, for each compression level, for both normal and EMA weights.

All configuration lives in the CONFIG block below — no CLI arguments needed.

Metrics computed at each (t, compression_grid) pair:
  1. vel_loss_lr   – MSE between v_pred and v_target at the low-res grid
  2. img_mse_lr    – MSE between x1_hat (recovered from v_pred via the ICPlan
                     formula x1_hat = xt + (1-t)*v_pred) and the true low-res
                     latent x1, evaluated at the low-res grid
  3. img_mse_fr    – full-res latent error at timestep t:
                     (x1_fr - (t*x1_fr + (1-t)*upsample(x0_lr - v_pred_lr)))**2

For loss_type='virtual_resize', only img_mse_fr is computed:
  The full-res latent is spatially compressed to the low-res grid and back
  (virtual resize), then noised and passed through the full-res model.
  The predicted clean latent is compared to the original full-res latent.
  This mirrors the virtual-resize condition in virtual_vs_real_resize_experiment.py.
"""

import os
import sys
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from fit.utils.utils import patchify, unpatchify, spatial_resize
from torch.utils.data import DataLoader, Subset

# ─────────────────────────────── CONFIG ─────────────────────────────────────

# Dataset root (same layout as training).
DATA_PATH = "/visinf/projects_students/mb_mvigel/datasets"

# Use the last N samples from the dataset for evaluation.
N_EVAL_SAMPLES = 128

# Compression grid sizes to evaluate.  Each entry is a square grid side-length
# (must be even; the latent patch size is 2, so spatial size = grid*2).
# The dataset resize_range used during training was [2, 16].
COMPRESSIONS = [2, 4, 6, 8, 10, 12, 14, 16]

# Number of evenly-spaced timesteps in (0, 1) to evaluate at.
N_TIMESTEPS = 20

# Batch size for the DataLoader (single GPU / CPU evaluation).
BATCH_SIZE = 256

# Dataset target_len (must match what was used for pre-computing latents).
TARGET_LEN = 256

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Base model config shared by all checkpoints.
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

# Checkpoints to evaluate.
# Each entry is a dict with:
#   name      – label used in results and printed output
#   dir       – folder containing model_1.safetensors (EMA weights)
#   loss_type – 'baseline', 'A', or 'B'
#               'baseline' → no size conditioning
#               'A'/'B'    → size conditioning
#
# Only model_1.safetensors (EMA) is evaluated.  Missing files are skipped.
CHECKPOINTS = [
    dict(
        name="baseline",
        dir="/visinf/projects_students/mb_mvigel/checkpoints/model_ema.safetensors",
        loss_type="baseline",
    ),
    dict(
        name="baseline_virtual_resize",
        dir="/visinf/projects_students/mb_mvigel/checkpoints/model_ema.safetensors",
        loss_type="virtual_resize",
    ),
    dict(
        name="zero_velocity",
        loss_type="zero_velocity_model",
    ),
    dict(
        name="loss_a_8k_ema",
        dir="/visinf/projects_students/mb_mvigel/workdir/fitv2_xl_cluster_a/checkpoints/checkpoint-8000",
        loss_type="A",
        use_ema=True,
    ),
    dict(
        name="loss_a_8k_train",
        dir="/visinf/projects_students/mb_mvigel/workdir/fitv2_xl_cluster_a/checkpoints/checkpoint-8000",
        loss_type="A",
        use_ema=False,
    ),
]

# Output file for results (JSON).
OUTPUT_JSON = "/visinf/projects_students/mb_mvigel/eval_losses_results.json"
# ─────────────────────────────────────────────────────────────────────────────

sys.path.insert(0, str(Path(__file__).parent))

from fit.model.fit_model import FiT
from fit.data.in1k_latent_dataset import IN1kLatentDataset


# ──────────────────────────── helpers ────────────────────────────────────────



def sample_paired_noise(
    B: int,
    H_fr: int, W_fr: int,
    H_lr: int, W_lr: int,
    sigma: torch.Tensor,
    sigma_inj: torch.Tensor,
    x1_fr: torch.Tensor,
    x1_lr: torch.Tensor,
    p: int = 2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample synchronized, isotropic noise at FR and LR via a common SCM grid.

    Draws i.i.d. Gaussian noise at the smallest grid whose size is a multiple of
    both H_fr and H_lr (assuming square grids), then area-pools down to each target
    resolution. This keeps the two noise vectors correlated and isotropic regardless
    of the ratio between sizes.

    Raises ValueError if the SCM grid is more than 8x larger than the FR grid.

    Returns:
        xt_fr    – noisy full-res tokens  (B, H_fr*W_fr, C), mixed with sigma
        xt_lr    – noisy low-res tokens   (B, H_lr*W_lr, C), mixed with sigma_inj
        x0_fr    – noise at FR resolution (B, H_fr*W_fr, C)
        x0_lr    – noise at LR resolution (B, H_lr*W_lr, C)
    """
    H_scm = math.lcm(H_fr, H_lr)
    W_scm = math.lcm(W_fr, W_lr)

    if H_scm > 8 * H_fr or W_scm > 8 * W_fr:
        raise ValueError(
            f"SCM grid ({H_scm}x{W_scm}) is more than 8x larger than FR grid "
            f"({H_fr}x{W_fr}). Reduce the size difference between FR and LR."
        )

    C = x1_fr.shape[-1]
    x0_scm = torch.randn(B, H_scm * W_scm, C, dtype=x1_fr.dtype, device=x1_fr.device)

    # Area-pooling by factor s reduces std by 1/s; rescale each output to std 1.
    x0_fr = spatial_resize(x0_scm, H_scm, W_scm, H_fr, W_fr, p=p, mode='area') * (H_scm / H_fr)
    x0_lr = spatial_resize(x0_scm, H_scm, W_scm, H_lr, W_lr, p=p, mode='area') * (H_scm / H_lr)

    sigma_fr  = sigma.view(B, 1, 1)
    sigma_lr  = sigma_inj.view(B, 1, 1)

    xt_fr = (1.0 - sigma_fr) * x1_fr + sigma_fr * x0_fr
    xt_lr = (1.0 - sigma_lr) * x1_lr + sigma_lr * x0_lr

    return xt_fr, xt_lr, x0_fr, x0_lr


class ZeroVelocityModel:
    """Always predicts zero velocity — used to cross-check the zero_velocity baseline."""
    def __call__(self, x, *args, **kwargs):
        return torch.zeros_like(x)  # args/kwargs intentionally ignored


def spatial_resize_sp(x_sp: torch.Tensor,
                      H_out: int, W_out: int,
                      patch_size: int = 2) -> torch.Tensor:
    """Bilinear resize of an already-unpatchified spatial tensor (B, C, H*p, W*p)."""
    return F.interpolate(x_sp.float(), size=(H_out * patch_size, W_out * patch_size),
                         mode="bilinear", align_corners=True).to(x_sp.dtype)

def load_model(ckpt_path: str, cfg: dict, device: str) -> FiT:
    """Instantiate FiT and load weights from a .safetensors file."""
    model = FiT(**cfg)
    state = load_file(ckpt_path, device="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  [warn] {len(missing)} missing keys (first 5: {missing[:5]})")
    if unexpected:
        print(f"  [warn] {len(unexpected)} unexpected keys (first 5: {unexpected[:5]})")
    model = model.to(device).eval()
    return model


def build_dataset(data_path: str, target_len: int) -> IN1kLatentDataset:
    """Dataset with full-res enabled so we can evaluate both resolutions."""
    return IN1kLatentDataset(
        root_dir=data_path,
        target_len=target_len,
        random="crop",          # matches training (picks the crop file variant)
        resize_range=None,      # we override grid size manually per compression
        return_fullres=True,
    )


def model_cfg_for(loss_type: str) -> dict:
    """Build a model config dict for a given loss type.

    baseline / virtual_resize → no size conditioning
    A / B                     → size conditioning
    """
    cfg = dict(_BASE_MODEL_CFG)
    cfg["use_size_cond"] = (loss_type not in ("baseline", "virtual_resize"))
    return cfg


def get_last_n_subset(dataset: IN1kLatentDataset, n: int) -> Subset:
    total = len(dataset)
    indices = list(range(max(0, total - n), total))
    return Subset(dataset, indices)


@torch.no_grad()
def evaluate_at_compression(
    model: FiT,
    dataloader: DataLoader,
    grid_size: int,
    timesteps: torch.Tensor,
    device: str,
    patch_size: int = 2,
) -> dict:
    """
    For a fixed grid size (compression) and a set of timesteps, compute the
    three metrics averaged over the dataloader.

    Returns a dict keyed by float(t) → {vel_loss_lr, img_mse_lr, img_mse_fr}.
    """
    p = patch_size

    # Accumulators: sum and count per timestep.
    T = len(timesteps)
    sums = {
        "vel_loss_lr": torch.zeros(T),
        "img_mse_lr":  torch.zeros(T),
        "img_mse_fr":  torch.zeros(T),
    }
    counts = torch.zeros(T)

    for batch in dataloader:
        # ── unpack batch ────────────────────────────────────────────────────
        # feature / grid / mask are at the original (full-res) resolution because
        # we set resize_range=None.  We manually downsample to `grid_size`.
        feat_fr = batch["feature"].to(device)          # (B, target_len, 16)
        mask_fr_raw = batch["mask"].to(device)         # (B, target_len)  uint8
        size_fr = batch["size_fullres"].to(device)     # (B, 1, 2)
        grid_fr_raw = batch["grid"].to(device)         # (B, 2, target_len)
        label = batch["label"].to(device).long().squeeze(-1)  # (B,)

        # size_fullres gives the actual grid dims of the stored latent.
        H_fr = int(size_fr[0, 0, 0])
        W_fr = int(size_fr[0, 0, 1])
        seq_fr = H_fr * W_fr

        # Extract valid full-res tokens.
        x1_fr = feat_fr[:, :seq_fr, :]                # (B, N_fr, 16)
        mask_fr = mask_fr_raw[:, :seq_fr].unsqueeze(-1).float()  # (B, N_fr, 1)

        # ── build low-res (compressed) x1 ───────────────────────────────────
        g = grid_size
        if g == H_fr:
            # Already at the requested resolution.
            x1_lr = x1_fr
            mask_lr = mask_fr
            H_lr = H_fr; W_lr = W_fr
        else:
            # Bilinear downsample from full-res spatial to g×g.
            x1_lr = spatial_resize(x1_fr, H_fr, W_fr, g, g)
            H_lr = g; W_lr = g
            seq_lr = H_lr * W_lr
            mask_lr = torch.ones(x1_lr.shape[0], seq_lr, 1, device=device, dtype=x1_lr.dtype)

        seq_lr = H_lr * W_lr

        # ── build model kwargs for low-res forward ──────────────────────────
        B = x1_lr.shape[0]
        hs = torch.arange(H_lr, dtype=x1_lr.dtype, device=device)
        ws = torch.arange(W_lr, dtype=x1_lr.dtype, device=device)
        gh, gw = torch.meshgrid(hs, ws, indexing="ij")
        # grid[0]=width coords, grid[1]=height coords (matches dataset and rope convention)
        grid_lr = torch.stack([gw.reshape(-1), gh.reshape(-1)]).unsqueeze(0).expand(B, -1, -1)

        mask_lr_seq = torch.ones(B, seq_lr, dtype=torch.uint8, device=device)

        size_lr_t = torch.tensor([[H_lr, W_lr]], dtype=torch.int32, device=device).expand(B, -1).unsqueeze(1)

        model_kwargs = dict(
            y=label,
            grid=grid_lr,
            mask=mask_lr_seq,
            size=size_lr_t,
        )

        # Downsampling ratio for noise correction (matches _forward_unpacked).
        # When the low-res grid is smaller than full-res, injecting noise at
        # sigma would yield a different effective SNR after bilinear downsampling.
        # The training code corrects for this via:
        #   sigma_inj = sigma * r / (1 - sigma * (1 - r))
        # where r = H_lr / H_fr.  At full resolution (r == 1) sigma_inj = sigma.
        r = H_lr / H_fr

        # ── loop over timesteps ─────────────────────────────────────────────
        for ti, t_val in enumerate(timesteps):
            t = t_val.expand(B).to(device).to(x1_lr.dtype)

            # Compute noise-corrected injection sigma (scalar → (B,)).
            sigma = 1.0 - t                                              # (B,)
            sigma_inj = sigma * r / (1.0 - sigma * (1.0 - r))           # (B,)

            _xt_fr, xt_lr_valid, _x0_fr, x0_lr = sample_paired_noise(
                B, H_fr, W_fr, H_lr, W_lr, sigma, sigma_inj, x1_fr, x1_lr, p=p,
            )

            sigma_exp = sigma_inj.view(B, 1, 1)
            ut_lr = x1_lr - x0_lr                                        # (B, seq_lr, 16)

            v_pred_lr = model(xt_lr_valid, 1 - sigma_inj, **model_kwargs)

            # ── low-res metrics (shared) ─────────────────────────────────────
            x1_hat_lr = xt_lr_valid + sigma_exp * v_pred_lr
            vel_loss_lr = ((v_pred_lr - ut_lr) * mask_lr).pow(2).mean(dim=[1, 2]).mean()
            img_mse_lr  = ((x1_hat_lr - x1_lr) * mask_lr).pow(2).mean(dim=[1, 2]).mean()

            # ── full-res latent metric ───────────────────────────────────────
            # Error at timestep t between the full-res target latent and a
            # reconstruction that mixes the clean full-res latent (weight t) with
            # the upsampled (noise - velocity) residual (weight 1 - t):
            #   (x1_fr - (t * x1_fr + (1 - t) * upsample(x0_lr - v_pred_lr))) ** 2
            resid_lr_sp = unpatchify(x0_lr - v_pred_lr, (H_lr * p, W_lr * p), p)
            resid_up = patchify(spatial_resize_sp(resid_lr_sp, H_fr, W_fr), p)
            t_fr = t.view(B, 1, 1)
            x1_hat_fr = t_fr * x1_fr + (1.0 - t_fr) * resid_up
            img_mse_fr = ((x1_fr - x1_hat_fr) * mask_fr).pow(2).mean(dim=[1, 2]).mean()

            # Accumulate (each batch contributes B samples).
            sums["vel_loss_lr"][ti] += vel_loss_lr.item() * B
            sums["img_mse_lr"][ti]  += img_mse_lr.item() * B
            sums["img_mse_fr"][ti]  += img_mse_fr.item() * B
            counts[ti] += B

    # Average over samples.
    results = {}
    for ti, t_val in enumerate(timesteps):
        key = f"{t_val.item():.4f}"
        n = counts[ti].item()
        results[key] = {
            "vel_loss_lr": sums["vel_loss_lr"][ti].item() / n,
            "img_mse_lr":  sums["img_mse_lr"][ti].item()  / n,
            "img_mse_fr":  sums["img_mse_fr"][ti].item()  / n,
        }
    return results


@torch.no_grad()
def evaluate_at_compression_virtual_resize(
    model: FiT,
    dataloader: DataLoader,
    grid_size: int,
    timesteps: torch.Tensor,
    device: str,
    patch_size: int = 2,
) -> dict:
    """
    Virtual-resize evaluation for the baseline model.

    The full-res model is run on a virtually-resized input — the clean latent is
    compressed to grid_size and back (up(down(image))) and noised at timestep t:
        model_input = t * up(down(image)) + (1 - t) * noise
    The predicted full-res velocity is then downsampled to the g×g grid
    (velocity = down(velocity_full_res)) and the metrics are computed against
    the low-res target velocity / latent:

        vel_loss_lr = (down(image) - down(noise) - velocity) ** 2
        img_mse_lr  = (down(image)
                       - (t*down(image) + (1-t)*(down(noise) - velocity))) ** 2
        img_mse_fr  = (image
                       - (t*image + (1-t)*up(down(noise) - velocity))) ** 2

    Returns a dict keyed by float(t) → {vel_loss_lr, img_mse_lr, img_mse_fr}.
    """
    p = patch_size

    T = len(timesteps)
    sums = {
        "vel_loss_lr": torch.zeros(T),
        "img_mse_lr":  torch.zeros(T),
        "img_mse_fr":  torch.zeros(T),
    }
    counts = torch.zeros(T)

    def down(x):
        return spatial_resize(x, H_fr, W_fr, g, g)

    def up(x):
        return spatial_resize(x, g, g, H_fr, W_fr)

    for batch in dataloader:
        feat_fr = batch["feature"].to(device)          # (B, target_len, 16)
        size_fr = batch["size_fullres"].to(device)     # (B, 1, 2)
        label = batch["label"].to(device).long().squeeze(-1)  # (B,)

        H_fr = int(size_fr[0, 0, 0])
        W_fr = int(size_fr[0, 0, 1])
        seq_fr = H_fr * W_fr

        x1_fr = feat_fr[:, :seq_fr, :]                # (B, N_fr, 16)
        mask_fr = torch.ones(x1_fr.shape[0], seq_fr, 1, device=device, dtype=x1_fr.dtype)

        B = x1_fr.shape[0]
        g = grid_size

        # ── virtual resize: compress x1_fr to g×g and back to full-res ────────
        x1_lr = down(x1_fr)                           # (B, N_lr, 16) — down(image)
        x1_virtual = up(x1_lr)                        # (B, N_fr, 16) — up(down(image))

        mask_lr = torch.ones(B, g * g, 1, device=device, dtype=x1_fr.dtype)

        # ── build model kwargs for full-res forward (no size conditioning) ─────
        hs = torch.arange(H_fr, dtype=x1_fr.dtype, device=device)
        ws = torch.arange(W_fr, dtype=x1_fr.dtype, device=device)
        gh, gw = torch.meshgrid(hs, ws, indexing="ij")
        # grid[0]=width coords, grid[1]=height coords (matches dataset and rope convention)
        grid_fr = torch.stack([gw.reshape(-1), gh.reshape(-1)]).unsqueeze(0).expand(B, -1, -1)

        mask_fr_seq = torch.ones(B, seq_fr, dtype=torch.uint8, device=device)

        size_fr_t = torch.tensor([[H_fr, W_fr]], dtype=torch.int32, device=device).expand(B, -1).unsqueeze(1)

        model_kwargs = dict(
            y=label,
            grid=grid_fr,
            mask=mask_fr_seq,
            size=size_fr_t,
        )

        for ti, t_val in enumerate(timesteps):
            t = t_val.expand(B).to(device).to(x1_fr.dtype)
            t_e = t.view(B, 1, 1)

            x0_fr = torch.randn_like(x1_fr)           # full-res noise
            x0_lr = down(x0_fr)                        # down(noise)

            # ── model input: t * up(down(image)) + (1 - t) * noise ────────────
            model_input = t_e * x1_virtual + (1.0 - t_e) * x0_fr
            v_pred_fr = model(model_input, t, **model_kwargs)  # (B, N_fr, 16)

            # ── velocity = down(velocity_full_res) ────────────────────────────
            velocity = down(v_pred_fr)                # (B, N_lr, 16)

            # ── low-res velocity error ────────────────────────────────────────
            target_velocity = x1_lr - x0_lr           # down(image) - down(noise)
            vel_loss_lr = ((target_velocity - velocity) * mask_lr).pow(2).mean(dim=[1, 2]).mean()

            # ── low-res image error ───────────────────────────────────────────
            recon_lr = t_e * x1_lr + (1.0 - t_e) * (x0_lr - velocity)
            img_mse_lr = ((x1_lr - recon_lr) * mask_lr).pow(2).mean(dim=[1, 2]).mean()

            # ── full-res image error ──────────────────────────────────────────
            recon_fr = t_e * x1_fr + (1.0 - t_e) * up(x0_lr - velocity)
            img_mse_fr = ((x1_fr - recon_fr) * mask_fr).pow(2).mean(dim=[1, 2]).mean()

            sums["vel_loss_lr"][ti] += vel_loss_lr.item() * B
            sums["img_mse_lr"][ti]  += img_mse_lr.item() * B
            sums["img_mse_fr"][ti]  += img_mse_fr.item() * B
            counts[ti] += B

    results = {}
    for ti, t_val in enumerate(timesteps):
        key = f"{t_val.item():.4f}"
        n = counts[ti].item()
        results[key] = {
            "vel_loss_lr": sums["vel_loss_lr"][ti].item() / n,
            "img_mse_lr":  sums["img_mse_lr"][ti].item()  / n,
            "img_mse_fr":  sums["img_mse_fr"][ti].item()  / n,
        }
    return results




# ──────────────────────────── main ───────────────────────────────────────────

def main():
    print(f"Device: {DEVICE}")

    # Evenly-spaced timesteps in (0, 1) — exclude exact 0 and 1.
    timesteps = torch.linspace(0.0, 1.0, N_TIMESTEPS + 2)[1:-1]
    print(f"Evaluating at {N_TIMESTEPS} timesteps: "
          f"{[f'{t:.3f}' for t in timesteps.tolist()]}")

    # Dataset (full-res, deterministic).
    print(f"\nLoading dataset from {DATA_PATH} …")
    full_dataset = build_dataset(DATA_PATH, TARGET_LEN)
    subset = get_last_n_subset(full_dataset, N_EVAL_SAMPLES)
    print(f"  {len(full_dataset)} total samples → using last {len(subset)}")

    dataloader = DataLoader(
        subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=(DEVICE == "cuda"),
        drop_last=False,
    )

    all_results = {}

    for ckpt_cfg in CHECKPOINTS:
        ckpt_name = ckpt_cfg["name"]
        loss_type = ckpt_cfg["loss_type"]

        print(f"\n{'='*60}")

        if loss_type == "zero_velocity_model":
            print(f"Checkpoint: {ckpt_name}  loss={loss_type}  (ZeroVelocityModel stub)")
            model = ZeroVelocityModel()
        else:
            ckpt_dir  = ckpt_cfg["dir"]
            use_ema   = ckpt_cfg.get("use_ema", None)  # None → plain file
            model_cfg = model_cfg_for(loss_type)

            # Resolve the actual .safetensors file (same logic as generate_images.py).
            if os.path.isfile(ckpt_dir):
                ckpt_path = ckpt_dir
                weight_name = "plain"
            else:
                fname = "model_1.safetensors" if use_ema else "model.safetensors"
                ckpt_path = os.path.join(ckpt_dir, fname)
                weight_name = "ema" if use_ema else "train"

            print(f"Checkpoint: {ckpt_name}  loss={loss_type}  weights={weight_name}")
            print(f"  {ckpt_path}")
            if not os.path.isfile(ckpt_path):
                print(f"  [skip] file not found")
                continue

            model = load_model(ckpt_path, model_cfg, DEVICE)

        results_for_run = {}

        for g in COMPRESSIONS:
            print(f"\n  Compression grid={g}×{g}  (seq_len={g*g} tokens, "
                  f"spatial={g*2}×{g*2} latent pixels)")

            if loss_type == "virtual_resize":
                per_t = evaluate_at_compression_virtual_resize(
                    model=model,
                    dataloader=dataloader,
                    grid_size=g,
                    timesteps=timesteps,
                    device=DEVICE,
                )
                results_for_run[f"grid_{g}x{g}"] = per_t

                header = f"{'t':>8}  {'vel_lr':>10}  {'img_lr':>10}  {'img_fr':>10}"
                print(f"    {header}")
                print(f"    {'-'*len(header)}")
                for t_key, vals in per_t.items():
                    print(f"    {float(t_key):8.4f}  "
                          f"{vals['vel_loss_lr']:10.6f}  "
                          f"{vals['img_mse_lr']:10.6f}  "
                          f"{vals['img_mse_fr']:10.6f}")
            else:
                per_t = evaluate_at_compression(
                    model=model,
                    dataloader=dataloader,
                    grid_size=g,
                    timesteps=timesteps,
                    device=DEVICE,
                )
                results_for_run[f"grid_{g}x{g}"] = per_t

                header = f"{'t':>8}  {'vel_lr':>10}  {'img_lr':>10}  {'img_fr':>10}"
                print(f"    {header}")
                print(f"    {'-'*len(header)}")
                for t_key, vals in per_t.items():
                    print(f"    {float(t_key):8.4f}  "
                          f"{vals['vel_loss_lr']:10.6f}  "
                          f"{vals['img_mse_lr']:10.6f}  "
                          f"{vals['img_mse_fr']:10.6f}")

        all_results[ckpt_name] = results_for_run
        del model

    # Save to JSON.
    with open(OUTPUT_JSON, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
