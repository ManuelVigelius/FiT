"""
FID measurement script for the noise-field schedules 19, 29, 33, 34 and the
standard full-resolution Euler integrator.

For each (checkpoint, sampler) pair this script:
  1. generates N_IMAGES images with the given sampler,
  2. writes them as PNGs to <OUTPUT_DIR>/<ckpt>/<sampler>/,
  3. computes the Fréchet Inception Distance with the `clean-fid` library
     against a precomputed custom reference built from real ImageNet-256.

The samplers measured are:
  - "euler_fr"          : standard full-res Euler ODE (generate_images.euler_sample_fr)
  - "noise_field_019"   : noise-field sampler, schedule [1,4,7,10,13,16,...,16]
  - "noise_field_029"   : noise-field sampler, schedule [6,8,10,12,14,16,...,16]
  - "noise_field_033"   : noise-field sampler, schedule [11,12,13,14,15,16,...,16]
  - "noise_field_034"   : noise-field sampler, schedule [16,16,...,16]

Reproducibility: both the reference statistics and the generated-image
statistics are produced by clean-fid's single Inception pipeline (identical
weights, preprocessing and resizing), so the FID is comparable across runs and
between samplers. clean-fid ships no ImageNet reference, so we build a custom
one (REF_STATS_NAME) once from real ImageNet-256 images (REF_IMAGE_DIR); it is
cached by clean-fid and reused on subsequent runs.

All configuration lives in the CONFIG block below — no CLI arguments needed.

Requires `clean-fid` (pip install clean-fid).

Results are printed and written to <OUTPUT_DIR>/fid_results.json.
"""

import json
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from diffusers.models import AutoencoderKL
from PIL import Image

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

# clean-fid custom reference. clean-fid ships no ImageNet reference, so we
# build one once (cached by clean-fid under this name) from real ImageNet-256
# images, then compare every sampler against it. Both sides go through the same
# clean-fid Inception pipeline → reproducible, sampler-comparable FID.
#   REF_STATS_NAME — clean-fid custom-stats name (the cache key).
#   REF_IMAGE_DIR  — folder of real images used to build the reference the
#                    first time. May be an ImageFolder tree (clean-fid walks it
#                    recursively). Ignored once the stats are cached.
# Overridable via FID_REF_NAME / FID_REF_DIR (set by tools/cluster_fid.sh).
REF_STATS_NAME = os.environ.get("FID_REF_NAME", "fit_imagenet256")
REF_IMAGE_DIR = os.environ.get(
    "FID_REF_DIR", "/content/drive/MyDrive/FiT/fid/imagenet256_real"
)
# clean-fid mode: "clean" (recommended, bicubic+antialias) for new, internally
# consistent numbers; "legacy_tensorflow" only if you must match old TF-FID.
FID_MODE = os.environ.get("FID_MODE", "clean")

# How to interpret REF_IMAGE_DIR when building the reference:
#   FID_REF_IS_LATENT=1 → it is an IN1kLatentDataset tree of VAE-encoded
#     latents (the data we train on); we decode them through the VAE to images
#     first, so the reference goes through the SAME decode path as the generated
#     samples. This is the cluster default (the fastdata set has no raw images).
#   FID_REF_IS_LATENT=0 → it is a folder of real image files; clean-fid reads
#     them directly (Colab convenience).
REF_IS_LATENT = os.environ.get("FID_REF_IS_LATENT", "1") == "1"

# Where to write decoded reference images while building the stats. They are
# only needed during make_custom_stats (the stats are cached afterwards), so a
# scratch dir is fine. Defaults to a sibling of OUTPUT_DIR.
REF_DECODE_DIR = os.environ.get("FID_REF_DECODE_DIR", "")

# Cap on the number of real images used to build the reference. Decoding the
# full latent set is expensive and 150k reals already give a stable FID
# reference (well above the 10k generated). Set FID_REF_MAX=0 to use all.
REF_MAX = int(os.environ.get("FID_REF_MAX", "150000"))

# VAE checkpoint.
VAE_PATH = gi.VAE_PATH

# Output directory: per-sampler PNG dirs + the results JSON.
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
        dict(
        name="loss_a_8k_ema",
        dir="/content/drive/MyDrive/FiT/inference_weights/checkpoint-8000",
        loss_type="A",
        use_ema=True,
    ),
    ]
else:
    CHECKPOINTS = gi.CHECKPOINTS

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


def _shard_range(total: int, rank: int, world_size: int) -> tuple[int, int]:
    """Contiguous [start, end) slice of `total` items for this rank."""
    per = (total + world_size - 1) // world_size
    start = rank * per
    end = min(start + per, total)
    return start, max(start, end)


# ─────────────────────────────── clean-fid ──────────────────────────────────

def _latent_files(root_dir: str) -> list[str]:
    """All per-image latent .safetensors paths in an IN1kLatentDataset tree.

    Mirrors IN1kLatentDataset's layout (from_16_to_256 / greater_than_256_resize
    / greater_than_256_crop). In practice only greater_than_256_resize/ is
    populated (256x256 crops); the other subdirs are tolerated when present. For
    files that exist in both the resize and crop variants we keep a single path
    (crop), matching the dataset's deduplication, so each image is counted once.
    """
    import os.path as osp

    dir_1 = osp.join(root_dir, f"from_16_to_{TARGET_LEN_PIX}")
    dir_2 = osp.join(root_dir, f"greater_than_{TARGET_LEN_PIX}_resize")
    dir_3 = osp.join(root_dir, f"greater_than_{TARGET_LEN_PIX}_crop")
    files_1 = os.listdir(dir_1) if osp.isdir(dir_1) else []
    files_2 = os.listdir(dir_2) if osp.isdir(dir_2) else []
    files_3 = os.listdir(dir_3) if osp.isdir(dir_3) else []
    files_23 = set(files_2) - set(files_3)
    paths = [osp.join(dir_1, f) for f in files_1]
    paths += [osp.join(dir_2, f) for f in files_23]
    paths += [osp.join(dir_3, f) for f in files_3]  # crop variant when both exist
    return paths


def _decode_reference_latents(vae, dst_dir: str, rank: int, world_size: int):
    """Decode the latent reference set into PNGs under dst_dir (sharded by rank).

    Reads IN1kLatentDataset .safetensors files, takes the unflipped variant,
    unpatchifies the (H_g, W_g, 16) grid to a (4, H, W) spatial latent and
    VAE-decodes it with the SAME path used for generated samples. Each rank
    decodes a contiguous slice; filenames are keyed to the global file index so
    shards never collide. Returns the total number of reference images.
    """
    from safetensors.torch import load_file

    paths = sorted(_latent_files(REF_IMAGE_DIR))
    if not paths:
        raise FileNotFoundError(
            f"No latent .safetensors found under {REF_IMAGE_DIR} (expected "
            f"IN1kLatentDataset subdirs, e.g. greater_than_{TARGET_LEN_PIX}_resize/). "
            f"Set FID_REF_IS_LATENT=0 if this is a folder of real images."
        )
    # Shuffle, then cap. Filenames are class-prefixed, so an alphabetical prefix
    # would be skewed toward the lowest class ids; shuffling makes the capped
    # subset span all classes. Seed sorted paths with a fixed RNG so every rank
    # produces the same permutation (and runs are reproducible).
    g = torch.Generator().manual_seed(GLOBAL_SEED)
    perm = torch.randperm(len(paths), generator=g).tolist()
    paths = [paths[i] for i in perm]
    if REF_MAX and len(paths) > REF_MAX:
        if rank == 0:
            print(f"    reference: shuffled, capped at {REF_MAX} / {len(perm)} latents")
        paths = paths[:REF_MAX]
    dtype = next(vae.parameters()).dtype
    start, end = _shard_range(len(paths), rank, world_size)
    for gi_idx in range(start, end):
        data = load_file(paths[gi_idx])
        feat_hw = data["feature"][0]                 # (H_g, W_g, 16), unflipped
        H_g, W_g = feat_hw.shape[0], feat_hw.shape[1]
        tokens = feat_hw.reshape(1, H_g * W_g, 16)   # (1, N, c*p*p)
        x1_sp = unpatchify(tokens, (H_g * PATCH_SIZE, W_g * PATCH_SIZE), PATCH_SIZE)
        x1_sp = x1_sp.to(DEVICE, dtype)
        imgs_uint8 = _decode_to_uint8(vae, x1_sp)    # (1, H, W, 3) uint8 CPU
        _save_pngs(imgs_uint8, dst_dir, gi_idx)
        if rank == 0 and (gi_idx - start) % 200 == 0:
            print(f"    decoding reference {gi_idx - start}/{end - start} "
                  f"(×{world_size} ranks)", end="\r", flush=True)
    return len(paths)


def _ensure_reference(vae, is_main: bool, rank: int, world_size: int):
    """Make sure the clean-fid custom reference exists (build it once).

    clean-fid ships no ImageNet reference, so the first run computes statistics
    from REF_IMAGE_DIR and caches them under REF_STATS_NAME. Subsequent runs
    (and other machines sharing the cache) reuse them.

    When REF_IS_LATENT, REF_IMAGE_DIR holds VAE-encoded latents (our training
    data): all ranks decode them to PNGs in parallel, then rank 0 builds the
    stats from the decoded folder. Otherwise REF_IMAGE_DIR is a folder of real
    images that clean-fid reads directly (rank 0 only).
    """
    from cleanfid import fid as cleanfid

    if cleanfid.test_stats_exists(REF_STATS_NAME, mode=FID_MODE):
        if is_main:
            print(f"  reference '{REF_STATS_NAME}' ({FID_MODE}) already cached.")
        return

    if not os.path.isdir(REF_IMAGE_DIR):
        if is_main:
            raise FileNotFoundError(
                f"Reference '{REF_STATS_NAME}' is not cached and REF_IMAGE_DIR "
                f"does not exist: {REF_IMAGE_DIR}"
            )
        return

    if REF_IS_LATENT:
        # Decode latents → PNGs (all ranks), barrier, then rank 0 builds stats.
        decode_dir = REF_DECODE_DIR or os.path.join(OUTPUT_DIR, "_reference_decoded")
        if is_main:
            os.makedirs(decode_dir, exist_ok=True)
            print(f"  decoding latent reference from {REF_IMAGE_DIR}\n"
                  f"  → {decode_dir} (VAE-decoded, then scored once) …")
        if _is_dist():
            dist.barrier()
        _decode_reference_latents(vae, decode_dir, rank, world_size)
        if _is_dist():
            dist.barrier()
        if is_main:
            print(f"\n  building reference '{REF_STATS_NAME}' from decoded PNGs …")
            cleanfid.make_custom_stats(REF_STATS_NAME, decode_dir, mode=FID_MODE)
            print(f"  reference '{REF_STATS_NAME}' built and cached.")
        return

    # Plain image folder: rank 0 builds directly.
    if is_main:
        print(f"  building reference '{REF_STATS_NAME}' from {REF_IMAGE_DIR} …")
        cleanfid.make_custom_stats(REF_STATS_NAME, REF_IMAGE_DIR, mode=FID_MODE)
        print(f"  reference '{REF_STATS_NAME}' built and cached.")


def _compute_fid_dir(image_dir: str, num_gen: int) -> float:
    """clean-fid between a folder of generated PNGs and the custom reference."""
    from cleanfid import fid as cleanfid

    return float(
        cleanfid.compute_fid(
            image_dir,
            dataset_name=REF_STATS_NAME,
            dataset_split="custom",
            mode=FID_MODE,
            num_gen=num_gen,
        )
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


def _save_pngs(imgs_uint8: torch.Tensor, out_dir: str, start_idx: int):
    """Write a batch of uint8 images (B, H, W, 3) as zero-padded PNGs.

    Filenames are keyed to the absolute global image index so shards from
    different ranks never collide.
    """
    arr = imgs_uint8.numpy()
    for j in range(arr.shape[0]):
        Image.fromarray(arr[j]).save(
            os.path.join(out_dir, f"{start_idx + j:07d}.png")
        )


def _generate_sampler_pngs(sampler_name, gen_one_batch, out_dir,
                           all_labels, all_noise_fr_sp, dtype,
                           rank, world_size):
    """Generate this rank's contiguous shard of N_IMAGES and write them as PNGs.

    Each rank takes a non-overlapping slice of the shared labels/noise, so the
    union of all shards covers N_IMAGES exactly once. FID is computed separately
    (by rank 0 via clean-fid) once every rank has finished writing.
    """
    shard_start, shard_end = _shard_range(N_IMAGES, rank, world_size)
    generated = shard_start
    while generated < shard_end:
        bs = min(BATCH_SIZE, shard_end - generated)
        y = all_labels[generated:generated + bs]
        noise_fr_sp = all_noise_fr_sp[generated:generated + bs].to(dtype)
        imgs_uint8 = gen_one_batch(y, noise_fr_sp, generated)
        _save_pngs(imgs_uint8, out_dir, generated)
        generated += bs
        if rank == 0:
            done = generated - shard_start
            total = shard_end - shard_start
            print(f"    [{sampler_name}] rank0 {done}/{total} (×{world_size} ranks)",
                  end="\r", flush=True)
    if rank == 0:
        print(f"    [{sampler_name}] PNGs written{' ' * 30}")


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

    log(f"\nLoading VAE from {VAE_PATH} …")
    vae = AutoencoderKL.from_pretrained(VAE_PATH).to(device).eval()

    # Build / verify the clean-fid reference. When the reference is latent it is
    # VAE-decoded here (all ranks), so the VAE must already be loaded. Ranks then
    # sync so nobody starts generating against a half-built cache.
    log(f"\nReference: clean-fid custom stats '{REF_STATS_NAME}' (mode={FID_MODE})")
    _ensure_reference(vae, is_main, rank, world_size)
    if _is_dist():
        dist.barrier()

    # Fixed labels and full-res noise shared across all samplers/checkpoints.
    # Generated on CPU with a fixed seed so every rank produces the identical
    # global tensors; each rank then operates on its own contiguous shard.
    g = torch.Generator().manual_seed(GLOBAL_SEED)
    all_labels = torch.randint(0, NUM_CLASSES, (N_IMAGES,), generator=g).to(device)
    all_noise_fr_sp = torch.randn(
        N_IMAGES, C_IN, H_fr * PATCH_SIZE, W_fr * PATCH_SIZE, generator=g
    ).to(device)
    log(f"Fixed labels and noise pre-generated (seed={GLOBAL_SEED}).")

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

        # Build the list of (sampler_name, per-batch generator) to run.
        def euler_batch(y, noise_fr_sp, generated):
            return _gen_euler_fr(model, vae, y, noise_fr_sp, H_fr, W_fr, dtype)

        samplers = [("euler_fr", euler_batch)]
        for sched_idx, grid_sizes in NOISE_FIELD_SCHEDULES.items():
            name = f"noise_field_{sched_idx:03d}"

            def nf_batch(y, noise_fr_sp, generated, _gs=grid_sizes):
                # Seed keyed to the absolute (global) image index so every
                # image's noise field is identical regardless of producing rank.
                return _gen_noise_field(model, vae, y, _gs, dtype,
                                        GLOBAL_SEED + generated)

            samplers.append((name, nf_batch))

        # --- Phase 1: every rank generates its shard of PNGs for each sampler.
        sampler_dirs = {}
        for name, gen_batch in samplers:
            out_dir = os.path.join(OUTPUT_DIR, ckpt_name, name)
            sampler_dirs[name] = out_dir
            if is_main:
                os.makedirs(out_dir, exist_ok=True)
            if _is_dist():
                dist.barrier()  # ensure the dir exists before any rank writes
            _generate_sampler_pngs(
                name, gen_batch, out_dir,
                all_labels, all_noise_fr_sp, dtype, rank, world_size,
            )

        # --- Phase 2: wait for all PNGs, then rank 0 scores each dir.
        if _is_dist():
            dist.barrier()
        if is_main:
            for name in sampler_dirs:
                fid = _compute_fid_dir(sampler_dirs[name], num_gen=N_IMAGES)
                results[ckpt_name][name] = fid
                log(f"    [{name}] FID = {fid:.4f}")

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
