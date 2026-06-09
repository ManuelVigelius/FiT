#!/bin/bash
# Cluster FID runner for FiT on visinf lab nodes.
# Measures FID for the noise-field schedules 19/29/33/34 and the standard
# full-res Euler integrator, across multiple GPUs via torch.distributed.
#
# Usage: bash tools/cluster_fid.sh
#
# Mirrors tools/cluster_train.sh: same conda env, same LD_LIBRARY_PATH setup,
# same torch.distributed.run launch. Unlike training it needs no dataset copy
# and no checkpoint symlink — measure_fid.py reads checkpoints directly.
#
# Storage layout (see cluster_train.sh for the full picture):
#   /visinf/home/mb_mvigel/FiT                       code (this repo)
#   /visinf/projects_students/mb_mvigel/checkpoints  model weights
#   /visinf/projects_students/mb_mvigel/envs/fit     conda environment
#   /visinf/projects_students/mb_mvigel/fid          FID reference + results

set -e

REPO_DIR="/visinf/home/mb_mvigel/FiT"
CONDA_ENV="/visinf/projects_students/mb_mvigel/envs/fit"
PERSISTENT_OUT="/visinf/projects_students/mb_mvigel"

# FID reference (clean-fid custom stats) and output directory.
# clean-fid ships no ImageNet reference, so measure_fid.py builds one ONCE and
# caches it under FID_REF_NAME (clean-fid's own cache, in the env).
#
# The reference is the SAME data we train on: VAE-encoded ImageNet-256 latents
# (an IN1kLatentDataset tree), already copied to fastdata by cluster_train.sh.
# FID_REF_IS_LATENT=1 makes measure_fid.py VAE-decode those latents to images
# (via the same decode path as the generated samples) before scoring, so the
# real and generated sides go through identical VAE+Inception pipelines.
FASTDATA="/fastdata/mb_mvigel/fit"
export FID_REF_NAME="${FID_REF_NAME:-fit_imagenet256}"
export FID_REF_DIR="${FID_REF_DIR:-$FASTDATA/datasets}"
export FID_REF_IS_LATENT="${FID_REF_IS_LATENT:-1}"
export FID_MODE="${FID_MODE:-clean}"
export FID_OUTPUT_DIR="${FID_OUTPUT_DIR:-$PERSISTENT_OUT/fid/eval}"
export FID_CLUSTER=1   # make measure_fid.py use the cluster checkpoint paths

GPUS_PER_NODE=4
MASTER_PORT=29501   # differ from train (29500) so both can run on one node

# Pin to a fixed set of GPUs so the run can NEVER touch devices owned by other
# jobs. Without this, rank 0 ends up opening a small CUDA context on EVERY
# visible device (the first torch.cuda.device_count() / NCCL init probes them
# all), spilling ~1 GB onto GPUs 6/7 where other people's jobs run. Masking
# visibility here means device_count()==4 and NCCL only ever sees 0-3.
# Override on the command line if 0-3 are taken, e.g.:
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash tools/cluster_fid.sh
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

echo "=== FiT Cluster FID Setup ==="
echo "Repo dir   : $REPO_DIR"
echo "GPUs       : $GPUS_PER_NODE"
echo "Reference  : clean-fid '$FID_REF_NAME' (mode=$FID_MODE, latent=$FID_REF_IS_LATENT)"
echo "Ref latents: $FID_REF_DIR"
echo "Output dir : $FID_OUTPUT_DIR"
echo ""

# The reference latents must be present on fastdata. cluster_train.sh copies
# them there; if you run FID standalone, copy them first (or point FID_REF_DIR
# at the persistent dataset at $PERSISTENT_OUT/datasets). In practice the latent
# tree only has the greater_than_256_resize/ split (256x256 crops).
if [ "$FID_REF_IS_LATENT" = "1" ] && [ ! -d "$FID_REF_DIR/greater_than_256_resize" ]; then
  echo "WARNING: $FID_REF_DIR does not look like an IN1kLatentDataset tree"
  echo "         (no greater_than_256_resize/ subdir). Falling back to persistent dataset."
  export FID_REF_DIR="$PERSISTENT_OUT/datasets"
fi

# Cap the number of real images used to build the reference (safety margin over
# the 10k generated). Set to 0 to use all available latents.
export FID_REF_MAX="${FID_REF_MAX:-150000}"

# --- Activate conda environment (identical to cluster_train.sh) ---
echo "[1/2] Activating conda environment..."
export PATH="$CONDA_ENV/bin:$PATH"
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH="$(find "$CONDA_ENV/lib/python3.11/site-packages/nvidia" -name "lib" -type d | tr '\n' ':')$CONDA_ENV/lib/python3.11/site-packages/triton/backends/nvidia/lib/cupti:$LD_LIBRARY_PATH"

# --- clean-fid provides the standardized FID pipeline (Inception + stats) ---
# Install with --no-deps so pip cannot upgrade/replace pinned training deps
# (a plain install here previously corrupted `regex` and broke diffusers).
# clean-fid's runtime deps (torch, torchvision, scipy, pillow, numpy, requests)
# are already present in the training env.
if ! python -c "import cleanfid" 2>/dev/null; then
  echo "[1/2] Installing clean-fid into the conda env (--no-deps)..."
  python -m pip install --no-input --no-deps clean-fid
fi

mkdir -p "$FID_OUTPUT_DIR"

# --- Launch distributed FID measurement ---
echo "[2/2] Launching FID measurement with $GPUS_PER_NODE GPUs..."
cd "$REPO_DIR"

python -m torch.distributed.run \
  --nnodes 1 \
  --nproc_per_node "$GPUS_PER_NODE" \
  --rdzv_id $RANDOM \
  --rdzv_backend c10d \
  --rdzv_endpoint "localhost:$MASTER_PORT" \
  measure_fid.py

echo ""
echo "Done. Results in $FID_OUTPUT_DIR/fid_results.json"
