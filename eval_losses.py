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
BATCH_SIZE = 16

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

    baseline → no size conditioning, no upsampler
    A / B    → size conditioning, no upsampler
    C        → size conditioning + ResNet upsampler
    """
    cfg = dict(_BASE_MODEL_CFG)
    cfg["use_size_cond"] = (loss_type != "baseline")
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
        label = batch["label"].to(device)              # (B,)

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
            x1_sp_fr = rearrange(x1_fr, "b (h w) (p1 p2 c) -> b c (h p1) (w p2)",
                                  h=H_fr, w=W_fr, p1=p, p2=p, c=C_in)
            x1_sp_lr = F.interpolate(x1_sp_fr.float(), size=(g * p, g * p),
                                      mode="bilinear", align_corners=True).to(x1_fr.dtype)
            x1_lr = rearrange(x1_sp_lr, "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
                               p1=p, p2=p)
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

            # Pad xt_lr to target_len for the model.
            xt_padded = torch.zeros(B, TARGET_LEN, 16, dtype=x1_lr.dtype, device=device)
            xt_padded[:, :seq_lr] = xt_lr_valid

            # Model forward (velocity prediction); t (not sigma_inj) is passed
            # to the timestep embedder, matching the training convention.
            v_pred_padded = model(xt_padded, t, **model_kwargs)  # (B, target_len, 16)
            v_pred_lr = v_pred_padded[:, :seq_lr, :]             # (B, seq_lr, 16)

            # True velocity target (ICPlan: ut = x1 - x0, unchanged by correction).
            ut_lr = x1_lr - x0_lr                                # (B, seq_lr, 16)

            # ── metric 1: velocity MSE at low-res ───────────────────────────
            vel_sq_lr = ((v_pred_lr - ut_lr) * mask_lr) ** 2    # (B, seq_lr, 16)
            vel_loss_lr = vel_sq_lr.mean(dim=[1, 2]).mean()      # scalar

            # ── metric 3: image MSE at low-res ──────────────────────────────
            # x1_hat = xt + sigma_inj * v_pred  (matches _loss_b: uses the
            # actual noise coefficient used to form xt, not the embedding t)
            x1_hat_lr = xt_lr_valid + sigma_inj_exp * v_pred_lr  # (B, seq_lr, 16)
            img_sq_lr = ((x1_hat_lr - x1_lr) * mask_lr) ** 2
            img_mse_lr = img_sq_lr.mean(dim=[1, 2]).mean()

            # ── upsample prediction to full-res for fr metrics ───────────────
            # Unpatchify low-res x1_hat → spatial → bilinear upsample → repatchify.
            x1_hat_sp_lr = rearrange(x1_hat_lr, "b (h w) (p1 p2 c) -> b c (h p1) (w p2)",
                                      h=H_lr, w=W_lr, p1=p, p2=p, c=C_in)
            x1_hat_sp_fr = F.interpolate(x1_hat_sp_lr.float(), size=(H_fr * p, W_fr * p),
                                          mode="bilinear", align_corners=True).to(x1_lr.dtype)
            x1_hat_fr = rearrange(x1_hat_sp_fr, "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
                                   p1=p, p2=p)                  # (B, N_fr, 16)

            # ── metric 4: image MSE at full-res ─────────────────────────────
            img_sq_fr = ((x1_hat_fr - x1_fr) * mask_fr) ** 2
            img_mse_fr = img_sq_fr.mean(dim=[1, 2]).mean()

            # ── derive full-res velocity from prediction for vel_loss_fr ────
            # ut_fr is noise sampled consistently with the lr pair.
            # We reuse x0_lr upsampled as an approximation of full-res noise;
            # this is the same approximation made in Loss B training.
            x0_fr_up_sp = F.interpolate(
                rearrange(x0_lr, "b (h w) (p1 p2 c) -> b c (h p1) (w p2)",
                          h=H_lr, w=W_lr, p1=p, p2=p, c=C_in).float(),
                size=(H_fr * p, W_fr * p), mode="bilinear", align_corners=True
            ).to(x1_lr.dtype)
            x0_fr_approx = rearrange(x0_fr_up_sp,
                                     "b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=p, p2=p)
            ut_fr = x1_fr - x0_fr_approx                        # (B, N_fr, 16)

            # v_pred_fr is our bilinearly upsampled velocity prediction.
            v_pred_sp_lr = rearrange(v_pred_lr, "b (h w) (p1 p2 c) -> b c (h p1) (w p2)",
                                      h=H_lr, w=W_lr, p1=p, p2=p, c=C_in)
            v_pred_sp_fr = F.interpolate(v_pred_sp_lr.float(), size=(H_fr * p, W_fr * p),
                                          mode="bilinear", align_corners=True).to(x1_lr.dtype)
            v_pred_fr = rearrange(v_pred_sp_fr,
                                  "b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=p, p2=p)

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
                per_t = evaluate_at_compression(
                    model=model,
                    dataloader=dataloader,
                    grid_size=g,
                    timesteps=timesteps,
                    device=DEVICE,
                )
                results_for_run[f"grid_{g}x{g}"] = per_t

                # Print a summary table for this compression.
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
