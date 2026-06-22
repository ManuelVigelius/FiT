"""Diagnostic: compare the *initial* velocity loss of the pretrained low-res
model against the learned-upsampler model initialized exactly as at the start of
Loss-C training.

Motivation
----------
Loss C starts at ~20 even though the upsampler is deliberately initialized to
reproduce the base model at native resolution. This script isolates whether that
is a bug or expected, by measuring both losses on the *same* synthetic latents
and the *same* noise family / timesteps the dataset uses.

What it measures
----------------
* Base model: plain velocity loss (ICPlan xt = t*x1 + (1-t)*x0, target ut = x1-x0)
  at a single grid size, averaged over N_TIMESTEPS timesteps x N_SAMPLES samples.
* Upsampler model: the exact `_loss_upsampler` path (full-res velocity target),
  for N_COMPRESSIONS different low-res grid sizes (the "compression sizes"),
  over the same timesteps.

Both models load the same pretrained checkpoint; the upsampler model is built
with use_upsampler=True so initialize_weights() applies the identity up_proj /
zero fr_embedder / shared-head init used at training start.

The comparison is *relative* (random x1 stands in for real latents); it answers
"does the upsampler path start near the base loss, or far above it?" — not the
absolute data loss.

Run (CPU is fine, just slower):
    python tools/test_upsampler_init_loss.py --ckpt checkpoints/fitv2_xl.safetensors
"""

import argparse
import os
import torch

from fit.model.fit_model import FiT
from fit.noise_field_sampler.noise_field_generator import sample_noise_fields_2d


# Matches configs/fitv2/config_fitv2_xl_colab_c.yaml network_config.params,
# minus use_upsampler (set per-model below).
BASE_CFG = dict(
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

N_SAMPLES = 128
N_TIMESTEPS = 10
# Low-res "compression" grid sizes to probe for the upsampler. The full-res grid
# is fixed at FULLRES_G; smaller low-res grids = more compression. Includes the
# no-op case (low-res == full-res), where the upsampler is designed to reproduce
# the base model exactly.
FULLRES_G = 16
COMPRESSION_GS = [16, 14, 12, 10, 8, 6, 4, 2]   # 8 sizes
assert len(COMPRESSION_GS) == 8

# Base model native grid for its own velocity loss baseline.
BASE_G = 16


def _grid(H_g, W_g, device, dtype):
    """(2, H_g*W_g) integer grid, w-fast/h-slow — matches the dataset convention."""
    hs = torch.arange(H_g, dtype=dtype)
    ws = torch.arange(W_g, dtype=dtype)
    gh, gw = torch.meshgrid(hs, ws, indexing="ij")
    return torch.stack([gw.reshape(-1), gh.reshape(-1)]).to(device=device, dtype=dtype)


def _timesteps():
    """Even sweep across (0, 1), avoiding the exact endpoints."""
    return torch.linspace(0.05, 0.95, N_TIMESTEPS)


@torch.no_grad()
def base_model_loss(model, device):
    """Plain velocity loss for the pretrained model at its native grid.

    For each sample we draw x1 ~ N(0, I) as a latent stand-in, x0 from the same
    consistent-noise family used in training, and average the velocity MSE over
    a sweep of timesteps.
    """
    H_g = W_g = BASE_G
    N = H_g * W_g
    grid = _grid(H_g, W_g, device, torch.long).unsqueeze(0)           # (1, 2, N)
    mask = torch.ones(1, N, device=device)
    size = torch.tensor([[[H_g, W_g]]], dtype=torch.int32, device=device)  # (1,1,2)

    ts = _timesteps()
    total, count = 0.0, 0
    for s in range(N_SAMPLES):
        x1 = torch.randn(N, 16, device=device)
        nf, = sample_noise_fields_2d([H_g], d=16, b=1)
        x0 = nf[0].permute(1, 2, 0).reshape(N, 16).to(device)
        y = torch.randint(0, 1000, (1,), device=device)
        for t_val in ts:
            t = t_val.to(device)
            xt = (t * x1 + (1.0 - t) * x0).unsqueeze(0)               # (1, N, 16)
            ut = (x1 - x0).unsqueeze(0)                               # (1, N, 16)
            v = model(xt, t.reshape(1), y, grid, mask, size=size)    # (1, N, 16)
            loss = ((v - ut) ** 2).mean()
            total += float(loss); count += 1
    return total / count


@torch.no_grad()
def upsampler_loss_for_size(model, lr_g, device):
    """`_loss_upsampler`-equivalent loss for one low-res (compression) grid size.

    Packed batch of a single image (n_pack=1) so the test runs on CPU without
    FlexAttention: doc_ids all 0 + all-valid mask reproduces document attention.
    The full-res target is ut_fr = x1_fr - x0_fr; the model predicts the base
    velocity (bicubically upsampled) plus the (zero-init) full-res enrichment.
    """
    H_lr = W_lr = lr_g
    H_fr = W_fr = FULLRES_G
    N_lr = H_lr * W_lr
    N_fr = H_fr * W_fr

    # Packed low-res inputs (B=1, n_pack=1).
    grid_lr = _grid(H_lr, W_lr, device, torch.long).unsqueeze(0)      # (1, 2, N_lr)
    mask_lr = torch.ones(1, N_lr, device=device)
    size_lr = torch.tensor([[[H_lr, W_lr]]], dtype=torch.int32, device=device)  # (1,1,2)
    doc_ids = torch.zeros(1, N_lr, dtype=torch.long, device=device)

    # Dense full-res fields (n_pack=1 row).
    grid_fr = _grid(H_fr, W_fr, device, torch.long).unsqueeze(0)      # (1, 2, N_fr)
    mask_fr = torch.ones(1, N_fr, device=device)
    size_fr = torch.tensor([[[H_fr, W_fr]]], dtype=torch.int32, device=device)  # (1, n_pack, 2)

    ts = _timesteps()
    total, count = 0.0, 0
    for s in range(N_SAMPLES):
        # Consistent cross-resolution noise + matching clean latents.
        x1_lr = torch.randn(N_lr, 16, device=device)
        x1_fr = torch.randn(N_fr, 16, device=device)
        nf_lr, nf_fr = sample_noise_fields_2d([H_lr, H_fr], d=16, b=1)
        x0_lr = nf_lr[0].permute(1, 2, 0).reshape(N_lr, 16).to(device)
        x0_fr = nf_fr[0].permute(1, 2, 0).reshape(N_fr, 16).to(device)
        y = torch.randint(0, 1000, (1, 1), device=device)            # (1, n_pack)
        for t_val in ts:
            t = t_val.to(device)
            xt_lr = (t * x1_lr + (1.0 - t) * x0_lr).unsqueeze(0)      # (1, N_lr, 16)
            xt_fr = (t * x1_fr + (1.0 - t) * x0_fr).unsqueeze(0)      # (1, N_fr, 16)
            ut_fr = (x1_fr - x0_fr).unsqueeze(0)                      # (1, N_fr, 16)
            t_pack = t.reshape(1, 1)                                  # (1, n_pack)

            v_fr = model(
                xt_lr, t_pack, y, grid_lr, mask_lr, size=size_lr,
                doc_ids=doc_ids, block_mask=None,
                x_fullres=xt_fr, grid_fullres=grid_fr,
                mask_fullres=mask_fr, size_fullres=size_fr,
            )                                                        # (1, N_fr, 16)

            m = mask_fr[..., None]
            sq = ((v_fr - ut_fr) * m) ** 2
            denom = m.sum().clamp(min=1)
            loss = sq.sum() / denom
            total += float(loss); count += 1
    return total / count


def main():
    global N_SAMPLES
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/fitv2_xl.safetensors")
    ap.add_argument("--gpu", type=int, default=None,
                    help="Physical GPU index to use (e.g. 0, 1, 2). Pins via "
                         "CUDA_VISIBLE_DEVICES so the run only touches that GPU. "
                         "Omit to use the default visible GPU; pass --device cpu "
                         "to force CPU.")
    ap.add_argument("--device", default=None,
                    help="Override device ('cuda' or 'cpu'). Defaults to cuda "
                         "when a GPU is available, else cpu.")
    ap.add_argument("--samples", type=int, default=N_SAMPLES)
    args = ap.parse_args()

    # Pin the GPU *before* any CUDA initialization. After this, the chosen
    # physical GPU is the only visible device and is addressed as cuda:0.
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    N_SAMPLES = args.samples
    torch.manual_seed(0)

    print(f"Device: {device} | samples: {N_SAMPLES} | timesteps: {N_TIMESTEPS}")
    print(f"Checkpoint: {args.ckpt}\n")

    # --ckpt none builds randomly-initialized models: loss numbers are then
    # meaningless, but it lets you smoke-test the forward/shape plumbing without
    # the (cluster-only) checkpoint.
    ckpt = None if args.ckpt.lower() == "none" else args.ckpt
    if ckpt is None:
        print("[warn] no checkpoint — random init, loss values are NOT meaningful\n")

    print("Building base (low-res) model ...")
    base = FiT(pretrain_ckpt=ckpt, **BASE_CFG).to(device).eval()

    print("Building upsampler model (init = training start) ...")
    up = FiT(pretrain_ckpt=ckpt, use_upsampler=True, **BASE_CFG).to(device).eval()

    print("\n=== Base model velocity loss (native 16x16 grid) ===")
    base_loss = base_model_loss(base, device)
    print(f"  base velocity loss = {base_loss:.4f}")

    print("\n=== Upsampler model velocity loss (full-res target, by compression) ===")
    print(f"  full-res grid fixed at {FULLRES_G}x{FULLRES_G}")
    print(f"  {'low-res':>8} | {'loss':>10}")
    print(f"  {'-'*8}-+-{'-'*10}")
    for lr_g in COMPRESSION_GS:
        l = upsampler_loss_for_size(up, lr_g, device)
        tag = "  (no-op: lr==fr)" if lr_g == FULLRES_G else ""
        print(f"  {lr_g:>6}^2 | {l:>10.4f}{tag}")

    print("\nInterpretation:")
    print("  - If the no-op row (lr==fr) is ~base loss, the upsampler init is")
    print("    sound and the high training loss comes from the full-res target")
    print("    being genuinely harder than the low-res velocity (expected).")
    print("  - If even the no-op row is ~20, the init/forward is broken.")


if __name__ == "__main__":
    main()
