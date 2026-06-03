"""
FID measurement script for the noise-field schedules 19, 29, 33, 34 and the
standard full-resolution Euler integrator.

For each (checkpoint, sampler) pair this script:
  1. generates N_IMAGES images with the given sampler,
  2. extracts pytorch-fid InceptionV3 pool3 (2048-d) activations,
  3. computes the Fréchet Inception Distance against an ADM reference
     statistics file (the canonical ImageNet-256
     `VIRTUAL_imagenet256_labeled.npz`, which stores `mu` and `sigma`).

The samplers measured are:
  - "euler_fr"          : standard full-res Euler ODE (generate_images.euler_sample_fr)
  - "noise_field_019"   : noise-field sampler, schedule [1,4,7,10,13,16,...,16]
  - "noise_field_029"   : noise-field sampler, schedule [6,8,10,12,14,16,...,16]
  - "noise_field_033"   : noise-field sampler, schedule [11,12,13,14,15,16,...,16]
  - "noise_field_034"   : noise-field sampler, schedule [16,16,...,16]

All configuration lives in the CONFIG block below — no CLI arguments needed.

Requires `pytorch-fid` (pip install pytorch-fid) for the FID-standard
InceptionV3 weights, plus scipy for the matrix sqrt.

Results are printed and written to <OUTPUT_DIR>/fid_results.json.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from diffusers.models import AutoencoderKL

sys.path.insert(0, str(Path(__file__).parent))

# Reuse the generation machinery already validated in generate_images.py.
import generate_images as gi
from generate_images import (
    BATCH_SIZE,
    CFG_SCALE,
    C_IN,
    DEVICE,
    NUM_CLASSES,
    PATCH_SIZE,
    TARGET_LEN_PIX,
    VAE_SCALE,
    euler_sample_fr,
    load_model,
    make_grid_and_mask,
    model_cfg_for,
)
from fit.noise_field_sampler.noise_field_sampler import sample as noise_field_sample
from fit.scheduler.transport.utils import patchify, unpatchify

# ─────────────────────────────── CONFIG ─────────────────────────────────────

# Noise-field schedules to evaluate, by index in generate_images._make_schedules().
# (Values inlined here so this script does not depend on that dict being enabled.)
NOISE_FIELD_SCHEDULES = {
    19: [1, 4, 7, 10, 13, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16],
    29: [6, 8, 10, 12, 14, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16],
    33: [11, 12, 13, 14, 15, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16],
    34: [16] * 16,
}

# Number of generated images per (checkpoint, sampler). For a meaningful FID
# this should be large (e.g. 10000-50000); keep it small only for smoke tests.
N_IMAGES = 10000

# Euler steps for the standard full-res sampler.
N_STEPS = 16

# Global seed for reproducibility.
GLOBAL_SEED = 42

# Path to the ADM reference statistics .npz (stores `mu` and `sigma`), e.g. the
# guided-diffusion VIRTUAL_imagenet256_labeled.npz.
# Overridable via the FID_REF_NPZ env var (set by tools/cluster_fid.sh).
REF_STATS_NPZ = os.environ.get(
    "FID_REF_NPZ", "/content/drive/MyDrive/FiT/fid/VIRTUAL_imagenet256_labeled.npz"
)

# VAE checkpoint.
VAE_PATH = gi.VAE_PATH

# Output directory for FID results (and an optional features cache).
# Overridable via the FID_OUTPUT_DIR env var.
OUTPUT_DIR = os.environ.get("FID_OUTPUT_DIR", "/content/drive/MyDrive/FiT/fid_eval")

# Checkpoints to evaluate. On the cluster, set FID_CLUSTER=1 to use the cluster
# checkpoint paths; otherwise reuse the active (Colab) list from generate_images.
if os.environ.get("FID_CLUSTER") == "1":
    CHECKPOINTS = [
        dict(
            name="baseline",
            dir="/visinf/projects_students/mb_mvigel/checkpoints/model_ema.safetensors",
            loss_type="baseline",
        ),
    ]
else:
    CHECKPOINTS = gi.CHECKPOINTS

# InceptionV3 batch size for feature extraction.
INCEPTION_BATCH_SIZE = 64

# ─────────────────────────────────────────────────────────────────────────────


# ──────────────────────────── distributed setup ─────────────────────────────

def _setup_distributed():
    """Initialise (optional) DDP. Works both under torchrun and single-process.

    Returns (rank, world_size, device). When launched without torchrun (no
    RANK/WORLD_SIZE env vars), runs as a single process on `gi.DEVICE`.
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count()))
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        rank, world_size = 0, 1
        device = torch.device(gi.DEVICE)
    # Point the reused generation helpers at this rank's device.
    gi.DEVICE = device
    globals()["DEVICE"] = device
    return rank, world_size, device


def _is_dist():
    return dist.is_available() and dist.is_initialized()


def _gather_features(local_feats: np.ndarray) -> np.ndarray | None:
    """All-gather per-rank (N_i, 2048) feature arrays onto rank 0.

    Ranks may hold different counts, so we pad to the global max, gather, then
    trim. Returns the concatenated features on rank 0, None on other ranks.
    """
    if not _is_dist():
        return local_feats

    device = DEVICE
    t = torch.from_numpy(local_feats).to(device)
    n = torch.tensor([t.shape[0]], device=device)
    counts = [torch.zeros_like(n) for _ in range(dist.get_world_size())]
    dist.all_gather(counts, n)
    counts = [int(c.item()) for c in counts]
    max_n = max(counts)

    feat_dim = t.shape[1]
    padded = torch.zeros(max_n, feat_dim, device=device, dtype=t.dtype)
    padded[: t.shape[0]] = t
    gathered = [torch.zeros_like(padded) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, padded)

    if dist.get_rank() != 0:
        return None
    parts = [g[:c].cpu().numpy() for g, c in zip(gathered, counts)]
    return np.concatenate(parts, axis=0)


def _shard_range(total: int, rank: int, world_size: int) -> tuple[int, int]:
    """Contiguous [start, end) slice of `total` items for this rank."""
    per = (total + world_size - 1) // world_size
    start = rank * per
    end = min(start + per, total)
    return start, max(start, end)


def _load_inception():
    """Load the pytorch-fid InceptionV3 (pool3, 2048-d) feature extractor."""
    from pytorch_fid.inception import InceptionV3
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    model = InceptionV3([block_idx]).to(DEVICE).eval()
    return model


@torch.no_grad()
def _inception_features(model, imgs_uint8: torch.Tensor) -> np.ndarray:
    """Extract 2048-d pool3 features for a batch of uint8 images (B, H, W, 3)."""
    feats = []
    for i in range(0, imgs_uint8.shape[0], INCEPTION_BATCH_SIZE):
        batch = imgs_uint8[i:i + INCEPTION_BATCH_SIZE].to(DEVICE)
        # pytorch-fid expects float in [0, 1], NCHW; it resizes to 299 internally.
        x = batch.permute(0, 3, 1, 2).float() / 255.0
        out = model(x)[0]  # (B, 2048, 1, 1)
        feats.append(out.squeeze(-1).squeeze(-1).cpu().numpy())
    return np.concatenate(feats, axis=0)


def _frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6) -> float:
    """Standard FID between two Gaussians (same maths as pytorch-fid)."""
    from scipy import linalg

    mu1, mu2 = np.atleast_1d(mu1), np.atleast_1d(mu2)
    sigma1, sigma2 = np.atleast_2d(sigma1), np.atleast_2d(sigma2)

    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset) @ (sigma2 + offset))
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError(f"Imaginary component {m} in sqrtm")
        covmean = covmean.real
    return float(diff @ diff + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean))


def _load_ref_stats(npz_path: str):
    """Load (mu, sigma) from an ADM-style reference .npz."""
    if not os.path.isfile(npz_path):
        raise FileNotFoundError(f"Reference statistics not found: {npz_path}")
    data = np.load(npz_path)
    if "mu" in data and "sigma" in data:
        return data["mu"], data["sigma"]
    # Some ADM files store raw activations under "arr_0" instead of mu/sigma.
    if "arr_0" in data:
        acts = data["arr_0"]
        return acts.mean(axis=0), np.cov(acts, rowvar=False)
    raise KeyError(
        f"{npz_path} has neither (mu, sigma) nor arr_0; keys = {list(data.keys())}"
    )


def _decode_to_uint8(vae, x1_sp: torch.Tensor) -> torch.Tensor:
    """VAE-decode a latent (B, C, H, W) into uint8 images (B, H, W, 3)."""
    with torch.no_grad():
        imgs = vae.decode(x1_sp / vae.config.scaling_factor).sample
    imgs = torch.clamp(127.5 * imgs + 128.0, 0, 255)
    return imgs.permute(0, 2, 3, 1).to(torch.uint8).cpu()


def _gen_euler_fr(model, vae, y, noise_fr_sp, H_fr, W_fr, dtype) -> torch.Tensor:
    z = patchify(noise_fr_sp, PATCH_SIZE)
    x1_sp = euler_sample_fr(
        model=model, z=z, y=y, H_fr=H_fr, W_fr=W_fr,
        n_steps=N_STEPS, cfg_scale=CFG_SCALE, device=DEVICE, dtype=dtype,
    )
    return _decode_to_uint8(vae, x1_sp)


def _gen_noise_field(model, vae, y, grid_sizes, dtype, seed) -> torch.Tensor:
    bs = y.shape[0]
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

    torch.manual_seed(seed)
    x1_sp = noise_field_sample(
        model_fn,
        scale_schedule=[g * PATCH_SIZE for g in grid_sizes],
        b=bs, d=C_IN, device=DEVICE, dtype=dtype,
    )
    return _decode_to_uint8(vae, x1_sp)


def _compute_fid_for_sampler(sampler_name, gen_one_batch, inception,
                             ref_mu, ref_sigma, all_labels, all_noise_fr_sp,
                             H_fr, W_fr, dtype, rank, world_size) -> float | None:
    """Generate this rank's shard of N_IMAGES, gather features, return FID.

    Each rank generates a contiguous, non-overlapping slice of the shared
    labels/noise so the union covers all N_IMAGES exactly once. Features are
    all-gathered onto rank 0, which computes and returns the FID (other ranks
    return None).
    """
    shard_start, shard_end = _shard_range(N_IMAGES, rank, world_size)
    feats = []
    generated = shard_start
    while generated < shard_end:
        bs = min(BATCH_SIZE, shard_end - generated)
        y = all_labels[generated:generated + bs]
        noise_fr_sp = all_noise_fr_sp[generated:generated + bs].to(dtype)
        imgs_uint8 = gen_one_batch(y, noise_fr_sp, generated)
        feats.append(_inception_features(inception, imgs_uint8))
        generated += bs
        if rank == 0:
            done = generated - shard_start
            total = shard_end - shard_start
            print(f"    [{sampler_name}] rank0 {done}/{total} (×{world_size} ranks)",
                  end="\r", flush=True)

    local_feats = (np.concatenate(feats, axis=0) if feats
                   else np.zeros((0, 2048), dtype=np.float32))
    acts = _gather_features(local_feats)
    if rank != 0:
        return None

    print(f"    [{sampler_name}] {acts.shape[0]}/{N_IMAGES} generated"
          f"{' ' * 20}")
    mu = acts.mean(axis=0)
    sigma = np.cov(acts, rowvar=False)
    fid = _frechet_distance(mu, sigma, ref_mu, ref_sigma)
    print(f"    [{sampler_name}] FID = {fid:.4f}")
    return fid


def main():
    rank, world_size, device = _setup_distributed()
    is_main = rank == 0

    def log(*a, **k):
        if is_main:
            print(*a, **k)

    if is_main:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
    torch.manual_seed(GLOBAL_SEED)
    log(f"Device: {device}  |  world_size: {world_size}")
    log(f"Computing FID over {N_IMAGES} images per (checkpoint, sampler)")
    log(f"CFG={CFG_SCALE}, euler steps={N_STEPS}")

    H_fr = TARGET_LEN_PIX // (PATCH_SIZE * VAE_SCALE)
    W_fr = TARGET_LEN_PIX // (PATCH_SIZE * VAE_SCALE)
    log(f"Full-res grid: {H_fr}×{W_fr}")

    # Reference stats are only needed on rank 0 (it computes the FID).
    ref_mu = ref_sigma = None
    if is_main:
        log(f"\nLoading reference statistics from {REF_STATS_NPZ} …")
        ref_mu, ref_sigma = _load_ref_stats(REF_STATS_NPZ)
        log(f"  reference: mu {ref_mu.shape}, sigma {ref_sigma.shape}")

    log("Loading InceptionV3 (pytorch-fid) …")
    inception = _load_inception()

    # Fixed labels and full-res noise shared across all samplers/checkpoints.
    # Generated on CPU with a fixed seed so every rank produces the identical
    # global tensors; each rank then operates on its own contiguous shard.
    g = torch.Generator().manual_seed(GLOBAL_SEED)
    all_labels = torch.randint(0, NUM_CLASSES, (N_IMAGES,), generator=g).to(device)
    all_noise_fr_sp = torch.randn(
        N_IMAGES, C_IN, H_fr * PATCH_SIZE, W_fr * PATCH_SIZE, generator=g
    ).to(device)
    log(f"Fixed labels and noise pre-generated (seed={GLOBAL_SEED}).")

    log(f"\nLoading VAE from {VAE_PATH} …")
    vae = AutoencoderKL.from_pretrained(VAE_PATH).to(device).eval()

    results = {}
    for ckpt_cfg in CHECKPOINTS:
        ckpt_name = ckpt_cfg["name"]
        ckpt_dir = ckpt_cfg["dir"]
        loss_type = ckpt_cfg["loss_type"]
        use_ema = ckpt_cfg.get("use_ema", None)

        if os.path.isfile(ckpt_dir):
            ckpt_path = ckpt_dir
        else:
            fname = "model_1.safetensors" if use_ema else "model.safetensors"
            ckpt_path = os.path.join(ckpt_dir, fname)

        log(f"\n{'='*60}")
        log(f"Checkpoint: {ckpt_name}  loss={loss_type}")
        log(f"  {ckpt_path}")
        if not os.path.isfile(ckpt_path):
            log("  [skip] file not found")
            continue

        model = load_model(ckpt_path, model_cfg_for(loss_type), device)
        dtype = next(model.parameters()).dtype
        results[ckpt_name] = {}

        # --- standard full-res Euler integrator ---
        def euler_batch(y, noise_fr_sp, generated):
            return _gen_euler_fr(model, vae, y, noise_fr_sp, H_fr, W_fr, dtype)

        results[ckpt_name]["euler_fr"] = _compute_fid_for_sampler(
            "euler_fr", euler_batch, inception, ref_mu, ref_sigma,
            all_labels, all_noise_fr_sp, H_fr, W_fr, dtype, rank, world_size,
        )

        # --- noise-field samplers for the requested schedules ---
        for sched_idx, grid_sizes in NOISE_FIELD_SCHEDULES.items():
            name = f"noise_field_{sched_idx:03d}"

            def nf_batch(y, noise_fr_sp, generated, _gs=grid_sizes):
                # Seed is keyed to the absolute (global) image index so every
                # image's noise field is identical regardless of which rank
                # produces it.
                return _gen_noise_field(model, vae, y, _gs, dtype,
                                        GLOBAL_SEED + generated)

            results[ckpt_name][name] = _compute_fid_for_sampler(
                name, nf_batch, inception, ref_mu, ref_sigma,
                all_labels, all_noise_fr_sp, H_fr, W_fr, dtype, rank, world_size,
            )

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if is_main:
        out_json = os.path.join(OUTPUT_DIR, "fid_results.json")
        with open(out_json, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\n{'='*60}\nFID results:")
        for ckpt_name, sampler_fids in results.items():
            print(f"  {ckpt_name}")
            for sampler_name, fid in sampler_fids.items():
                print(f"    {sampler_name:18s}  FID = {fid:.4f}")
        print(f"\nSaved {out_json}")

    if _is_dist():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
