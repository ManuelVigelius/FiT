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

# FID reference statistics (.npz) and output directory.
# Download the reference once with:
#   mkdir -p "$PERSISTENT_OUT/fid"
#   wget -c -O "$PERSISTENT_OUT/fid/VIRTUAL_imagenet256_labeled.npz" \
#     https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/256/VIRTUAL_imagenet256_labeled.npz
export FID_REF_NPZ="${FID_REF_NPZ:-$PERSISTENT_OUT/fid/VIRTUAL_imagenet256_labeled.npz}"
export FID_OUTPUT_DIR="${FID_OUTPUT_DIR:-$PERSISTENT_OUT/fid/eval}"
export FID_CLUSTER=1   # make measure_fid.py use the cluster checkpoint paths

GPUS_PER_NODE=4
MASTER_PORT=29501   # differ from train (29500) so both can run on one node

echo "=== FiT Cluster FID Setup ==="
echo "Repo dir   : $REPO_DIR"
echo "GPUs       : $GPUS_PER_NODE"
echo "Reference  : $FID_REF_NPZ"
echo "Output dir : $FID_OUTPUT_DIR"
echo ""

# --- Activate conda environment (identical to cluster_train.sh) ---
echo "[1/2] Activating conda environment..."
export PATH="$CONDA_ENV/bin:$PATH"
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH="$(find "$CONDA_ENV/lib/python3.11/site-packages/nvidia" -name "lib" -type d | tr '\n' ':')$CONDA_ENV/lib/python3.11/site-packages/triton/backends/nvidia/lib/cupti:$LD_LIBRARY_PATH"

# --- pytorch-fid is required for the FID-standard InceptionV3 weights ---
if ! python -c "import pytorch_fid" 2>/dev/null; then
  echo "[1/2] Installing pytorch-fid into the conda env..."
  python -m pip install --no-input pytorch-fid
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
