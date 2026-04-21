"""
Evaluation script: measures velocity loss and image-MSE at evenly-spaced
timesteps, for each compression level, for both normal and EMA weights.

All configuration lives in the CONFIG block below — no CLI arguments needed.

Metrics computed at each (t, compression_grid) pair:
  1. vel_loss_lr   – MSE between v_pred and v_target at the low-res grid
  2. vel_loss_fr   – MSE between upsampled-v_pred and v_target at full-res
                     (bilinear upsample of the low-res prediction; meaningful
                      even for Loss A/B/C checkpoints as a diagnostic)
  3. img_mse_lr    – MSE between x1_hat (recovered from v_pred via the ICPlan
                     formula x1_hat = xt + (1-t)*v_pred) and the true low-res
                     latent x1, evaluated at the low-res grid
  4. img_mse_fr    – same but upsampled to full-res and compared to x1_fr

For loss_type='virtual_resize', only img_mse_fr is computed:
  The full-res latent is spatially compressed to the low-res grid and back
  (virtual resize), then noised and passed through the full-res model.
  The predicted clean latent is compared to the original full-res latent.
  This mirrors the virtual-resize condition in virtual_vs_real_resize_experiment.py.
"""

import os
import sys
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from einops import rearrange
from safetensors.torch import load_file
from torch.utils.data import DataLoader, Subset

# ─────────────────────────────── CONFIG ─────────────────────────────────────

# Dataset root (same layout as training).
DATA_PATH = "datasets/imagenet1k_latents_256_sd_vae_ft_ema"

# Use the last N samples from the dataset for evaluation.
N_EVAL_SAMPLES = 256

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
# use_upsampler is overridden per-checkpoint for Loss C.
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

# Checkpoints to evaluate.
# Each entry is a dict with:
#   name      – label used in results and printed output
#   dir       – folder containing model_1.safetensors (EMA weights)
#   loss_type – 'baseline', 'A', 'B', or 'C'
#               'baseline' → no size conditioning, no upsampler
#               'A'/'B'    → size conditioning, no upsampler
#               'C'        → size conditioning + ResNet upsampler
#
# Only model_1.safetensors (EMA) is evaluated.  Missing files are skipped.
CHECKPOINTS = [
    dict(
        name="baseline",
        dir="/content/drive/MyDrive/FiT/inference_weights/checkpoint-baseline",
        loss_type="baseline",
    ),
    dict(
        name="baseline_virtual_resize",
        dir="/content/drive/MyDrive/FiT/inference_weights/checkpoint-baseline",
        loss_type="virtual_resize",
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

# Output file for results (JSON).
OUTPUT_JSON = "eval_losses_results.json"
# ─────────────────────────────────────────────────────────────────────────────

sys.path.insert(0, str(Path(__file__).parent))

from fit.model.fit_model import FiT
from fit.data.in1k_latent_dataset import IN1kLatentDataset


# ──────────────────────────── helpers ────────────────────────────────────────

def spatial_resize(x: torch.Tensor, H: int, W: int,
                   H_out: int, W_out: int,
                   patch_size: int = 2, C_in: int = 4) -> torch.Tensor:
    """Bilinear resize of a patchified token sequence.

    Args:
        x:        (B, H*W, patch_size**2 * C_in) token sequence
        H, W:     input grid dims
        H_out, W_out: output grid dims
        patch_size, C_in: patch and channel dims (must match x)
    Returns:
        (B, H_out*W_out, patch_size**2 * C_in) token sequence
    """
    p = patch_size
    sp = rearrange(x, "b (h w) (p1 p2 c) -> b c (h p1) (w p2)",
                   h=H, w=W, p1=p, p2=p, c=C_in)
    sp = F.interpolate(sp.float(), size=(H_out * p, W_out * p),
                       mode="bilinear", align_corners=True).to(x.dtype)
    return rearrange(sp, "b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=p, p2=p)


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

    baseline / virtual_resize → no size conditioning, no upsampler
    A / B                     → size conditioning, no upsampler
    C                         → size conditioning + ResNet upsampler
    """
    cfg = dict(_BASE_MODEL_CFG)
    cfg["use_size_cond"] = (loss_type not in ("baseline", "virtual_resize"))
    cfg["use_upsampler"] = (loss_type == "C")
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
    C_in: int = 4,
    use_resnet_upsampler: bool = False,
) -> dict:
    """
    For a fixed grid size (compression) and a set of timesteps, compute the
    four metrics averaged over the dataloader.

    Returns a dict keyed by float(t) → {vel_loss_lr, vel_loss_fr,
                                         img_mse_lr, img_mse_fr}.
    """
    p = patch_size

    # Accumulators: sum and count per timestep.
    T = len(timesteps)
    sums = {
        "vel_loss_lr": torch.zeros(T),
        "vel_loss_fr": torch.zeros(T),
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
        grid_lr = torch.zeros(B, 2, TARGET_LEN, dtype=x1_lr.dtype, device=device)
        grid_lr[:, 0, :seq_lr] = gh.reshape(-1).unsqueeze(0).expand(B, -1)
        grid_lr[:, 1, :seq_lr] = gw.reshape(-1).unsqueeze(0).expand(B, -1)

        mask_lr_seq = torch.zeros(B, TARGET_LEN, dtype=torch.uint8, device=device)
        mask_lr_seq[:, :seq_lr] = 1

        feat_lr_padded = torch.zeros(B, TARGET_LEN, 16, dtype=x1_lr.dtype, device=device)
        feat_lr_padded[:, :seq_lr] = x1_lr

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
        #   sigma_inj = sigma / (r + sigma * (1 - r))
        # where r = H_lr / H_fr.  At full resolution (r == 1) sigma_inj = sigma.
        r = H_lr / H_fr

        # ── loop over timesteps ─────────────────────────────────────────────
        for ti, t_val in enumerate(timesteps):
            t = t_val.expand(B).to(device).to(x1_lr.dtype)

            # Compute noise-corrected injection sigma (scalar → (B,)).
            sigma = 1.0 - t                                       # (B,)
            if r < 1.0:
                sigma_inj = sigma / (r + sigma * (1.0 - r))      # (B,)
            else:
                sigma_inj = sigma                                  # no correction needed

            # Build noisy latent using sigma_inj (ICPlan with correction).
            x0_lr = torch.randn_like(x1_lr)
            sigma_inj_exp = sigma_inj.view(B, 1, 1)
            alpha_inj_exp = (1.0 - sigma_inj_exp)
            xt_lr_valid = alpha_inj_exp * x1_lr + sigma_inj_exp * x0_lr  # (B, seq_lr, 16)

            if use_resnet_upsampler:
                # ── Loss C: mirror _loss_c exactly ──────────────────────────
                # Full-res i.i.d. noise; xt_lr derived by downsampling xt_fr;
                # plain sigma = 1-t (no injection correction).
                sigma_exp = (1.0 - t).view(B, 1, 1)

                x0_fr = torch.randn_like(x1_fr)                 # (B, N_fr, 16)
                xt_fr = (1.0 - sigma_exp) * x1_fr + sigma_exp * x0_fr

                xt_fr_sp = rearrange(xt_fr, "b (h w) (p1 p2 c) -> b c (h p1) (w p2)",
                                      h=H_fr, w=W_fr, p1=p, p2=p, c=C_in)
                xt_lr_c = spatial_resize(xt_fr, H_fr, W_fr, H_lr, W_lr)  # downsample xt_fr

                # FiT forward at low-res.
                xt_lr_c_padded = torch.zeros(B, TARGET_LEN, 16, dtype=x1_lr.dtype, device=device)
                xt_lr_c_padded[:, :seq_lr] = xt_lr_c
                v_pred_lr_c = model(xt_lr_c_padded, t, **model_kwargs)[:, :seq_lr, :]

                x1_lr_hat_c = xt_lr_c + sigma_exp * v_pred_lr_c  # (B, N_lr, 16)

                # ── metric 1 (Loss C): velocity MSE at low-res ──────────────
                x0_lr_c = spatial_resize(x0_fr, H_fr, W_fr, H_lr, W_lr)
                ut_lr = x1_lr - x0_lr_c
                vel_loss_lr = ((v_pred_lr_c - ut_lr) * mask_lr).pow(2).mean(dim=[1, 2]).mean()

                # ── metric 3 (Loss C): image MSE at low-res ─────────────────
                img_mse_lr = ((x1_lr_hat_c - x1_lr) * mask_lr).pow(2).mean(dim=[1, 2]).mean()

                # Upsample predicted clean LR latent and run ResNet.
                x1_lr_up_c_sp = rearrange(x1_lr_hat_c, "b (h w) (p1 p2 c) -> b c (h p1) (w p2)",
                                           h=H_lr, w=W_lr, p1=p, p2=p, c=C_in)
                x1_lr_up_c_sp = spatial_resize_sp(x1_lr_up_c_sp, H_fr, W_fr)
                v_pred_sp_fr = model.upsampler(x1_lr_up_c_sp, xt_fr_sp)  # (B, 4, H_sp, W_sp)

                ut_fr = x1_fr - x0_fr                           # (B, N_fr, 16)
                x1_hat_sp_fr = xt_fr_sp + sigma_exp.view(B, 1, 1, 1) * v_pred_sp_fr
            else:
                # ── Loss A/B path ────────────────────────────────────────────
                xt_padded = torch.zeros(B, TARGET_LEN, 16, dtype=x1_lr.dtype, device=device)
                xt_padded[:, :seq_lr] = xt_lr_valid
                v_pred_lr = model(xt_padded, 1 - sigma_inj, **model_kwargs)[:, :seq_lr, :]

                ut_lr = x1_lr - x0_lr                                # (B, seq_lr, 16)

                # ── metric 1: velocity MSE at low-res ───────────────────────
                vel_loss_lr = ((v_pred_lr - ut_lr) * mask_lr).pow(2).mean(dim=[1, 2]).mean()

                # ── metric 3: image MSE at low-res ──────────────────────────
                # x1_hat = xt + sigma_inj * v_pred (uses actual noise coeff, not t)
                x1_hat_lr = xt_lr_valid + sigma_inj_exp * v_pred_lr  # (B, seq_lr, 16)
                img_mse_lr = ((x1_hat_lr - x1_lr) * mask_lr).pow(2).mean(dim=[1, 2]).mean()

                # Approximate full-res noise by upsampling x0_lr.
                ut_fr = x1_fr - spatial_resize(x0_lr, H_lr, W_lr, H_fr, W_fr)

                v_pred_sp_fr = rearrange(spatial_resize(v_pred_lr, H_lr, W_lr, H_fr, W_fr),
                                         "b (h w) (p1 p2 c) -> b c (h p1) (w p2)",
                                         h=H_fr, w=W_fr, p1=p, p2=p, c=C_in)
                x1_hat_sp_fr = rearrange(spatial_resize(x1_hat_lr, H_lr, W_lr, H_fr, W_fr),
                                         "b (h w) (p1 p2 c) -> b c (h p1) (w p2)",
                                         h=H_fr, w=W_fr, p1=p, p2=p, c=C_in)

            v_pred_fr = rearrange(v_pred_sp_fr,
                                  "b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=p, p2=p)
            x1_hat_fr = rearrange(x1_hat_sp_fr, "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
                                   p1=p, p2=p)                  # (B, N_fr, 16)

            # ── metric 4: image MSE at full-res ─────────────────────────────
            img_sq_fr = ((x1_hat_fr - x1_fr) * mask_fr) ** 2
            img_mse_fr = img_sq_fr.mean(dim=[1, 2]).mean()

            # ── metric 2: velocity MSE at full-res ──────────────────────────
            vel_sq_fr = ((v_pred_fr - ut_fr) * mask_fr) ** 2
            vel_loss_fr = vel_sq_fr.mean(dim=[1, 2]).mean()

            # Accumulate (each batch contributes B samples).
            sums["vel_loss_lr"][ti] += vel_loss_lr.item() * B
            sums["vel_loss_fr"][ti] += vel_loss_fr.item() * B
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
            "vel_loss_fr": sums["vel_loss_fr"][ti].item() / n,
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
    C_in: int = 4,
) -> dict:
    """
    Virtual-resize evaluation for the baseline model.

    For each sample the full-res latent is spatially compressed to grid_size
    and bilinearly upsampled back to full-res (virtual resize).  The result is
    noised and passed through the full-res model.  The predicted clean latent is
    compared to the original full-res latent via img_mse_fr.

    This mirrors the 'virtual resize' condition from
    virtual_vs_real_resize_experiment.py.

    Returns a dict keyed by float(t) → {img_mse_fr}.
    """
    p = patch_size

    T = len(timesteps)
    sums = {"img_mse_fr": torch.zeros(T)}
    counts = torch.zeros(T)

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
        x1_virtual = spatial_resize(
            spatial_resize(x1_fr, H_fr, W_fr, g, g),
            g, g, H_fr, W_fr,
        )                                            # (B, N_fr, 16) — virtually resized

        # ── build model kwargs for full-res forward (no size conditioning) ─────
        hs = torch.arange(H_fr, dtype=x1_fr.dtype, device=device)
        ws = torch.arange(W_fr, dtype=x1_fr.dtype, device=device)
        gh, gw = torch.meshgrid(hs, ws, indexing="ij")
        grid_fr = torch.zeros(B, 2, TARGET_LEN, dtype=x1_fr.dtype, device=device)
        grid_fr[:, 0, :seq_fr] = gh.reshape(-1).unsqueeze(0).expand(B, -1)
        grid_fr[:, 1, :seq_fr] = gw.reshape(-1).unsqueeze(0).expand(B, -1)

        mask_fr_seq = torch.zeros(B, TARGET_LEN, dtype=torch.uint8, device=device)
        mask_fr_seq[:, :seq_fr] = 1

        feat_fr_padded = torch.zeros(B, TARGET_LEN, 16, dtype=x1_fr.dtype, device=device)

        size_fr_t = torch.tensor([[H_fr, W_fr]], dtype=torch.int32, device=device).expand(B, -1).unsqueeze(1)

        model_kwargs = dict(
            y=label,
            grid=grid_fr,
            mask=mask_fr_seq,
            size=size_fr_t,
        )

        for ti, t_val in enumerate(timesteps):
            t = t_val.expand(B).to(device).to(x1_fr.dtype)
            sigma = (1.0 - t).view(B, 1, 1)

            # Noise the virtually-resized latent at full-res (no sigma correction
            # needed — the model runs at its native resolution).
            x0_fr = torch.randn_like(x1_virtual)
            xt_virtual = (1.0 - sigma) * x1_virtual + sigma * x0_fr

            feat_fr_padded[:, :seq_fr] = xt_virtual
            xt_padded = feat_fr_padded.clone()

            v_pred_padded = model(xt_padded, t, **model_kwargs)  # (B, target_len, 16)
            v_pred_fr = v_pred_padded[:, :seq_fr, :]             # (B, N_fr, 16)

            # Recover predicted clean latent: x1_hat = xt + sigma * v_pred
            x1_hat_fr = xt_virtual + sigma * v_pred_fr           # (B, N_fr, 16)

            # img_mse_fr: compare to the original (uncompressed) full-res latent
            img_sq_fr = ((x1_hat_fr - x1_fr) * mask_fr) ** 2
            img_mse_fr = img_sq_fr.mean(dim=[1, 2]).mean()

            sums["img_mse_fr"][ti] += img_mse_fr.item() * B
            counts[ti] += B

    results = {}
    for ti, t_val in enumerate(timesteps):
        key = f"{t_val.item():.4f}"
        n = counts[ti].item()
        results[key] = {"img_mse_fr": sums["img_mse_fr"][ti].item() / n}
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
        ckpt_dir  = ckpt_cfg["dir"]
        loss_type = ckpt_cfg["loss_type"]
        model_cfg = model_cfg_for(loss_type)

        weight_files = {
            "ema": os.path.join(ckpt_dir, "model_1.safetensors"),
        }

        for weight_name, ckpt_path in weight_files.items():
            run_key = f"{ckpt_name}/{weight_name}"
            print(f"\n{'='*60}")
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

                    header = f"{'t':>8}  {'img_fr':>10}"
                    print(f"    {header}")
                    print(f"    {'-'*len(header)}")
                    for t_key, vals in per_t.items():
                        print(f"    {float(t_key):8.4f}  "
                              f"{vals['img_mse_fr']:10.6f}")
                else:
                    per_t = evaluate_at_compression(
                        model=model,
                        dataloader=dataloader,
                        grid_size=g,
                        timesteps=timesteps,
                        device=DEVICE,
                        use_resnet_upsampler=(loss_type == "C"),
                    )
                    results_for_run[f"grid_{g}x{g}"] = per_t

                    header = f"{'t':>8}  {'vel_lr':>10}  {'vel_fr':>10}  {'img_lr':>10}  {'img_fr':>10}"
                    print(f"    {header}")
                    print(f"    {'-'*len(header)}")
                    for t_key, vals in per_t.items():
                        print(f"    {float(t_key):8.4f}  "
                              f"{vals['vel_loss_lr']:10.6f}  "
                              f"{vals['vel_loss_fr']:10.6f}  "
                              f"{vals['img_mse_lr']:10.6f}  "
                              f"{vals['img_mse_fr']:10.6f}")

            all_results[run_key] = results_for_run

    # Save to JSON.
    with open(OUTPUT_JSON, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
