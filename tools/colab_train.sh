#!/bin/bash
# Each session: copy assets from Drive to local filesystem, then launch training.
# Usage: bash tools/colab_train.sh [A|B|C] [DRIVE_BASE]
# DRIVE_BASE defaults to /content/drive/MyDrive/FiT

set -e

LOSS="${1:?Usage: colab_train.sh [A|B|C] [DRIVE_BASE]}"
DRIVE_BASE="${2:-/content/drive/MyDrive/FiT}"
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"

case "${LOSS^^}" in
  A) CONFIG="configs/fitv2/config_fitv2_xl_colab_a.yaml" ;;
  B) CONFIG="configs/fitv2/config_fitv2_xl_colab_b.yaml" ;;
  C) CONFIG="configs/fitv2/config_fitv2_xl_colab_c.yaml" ;;
  *) echo "Unknown loss variant '${LOSS}'. Must be A, B, or C."; exit 1 ;;
esac

PROJECT="fitv2_xl_colab_${LOSS,,}"
WORKDIR="workdir/${PROJECT}"

# --- Copy tarball locally, then extract ---
DATASET_DEST="$REPO_DIR/datasets/imagenet1k_latents_256_sd_vae_ft_ema"
mkdir -p "$DATASET_DEST"
cp "$DRIVE_BASE/datasets/imagenet1k_latents_256_sd_vae_ft_ema/greater_than_256_crop.tar.gz" \
   "$DATASET_DEST/"
tar -xzf "$DATASET_DEST/greater_than_256_crop.tar.gz" -C "$DATASET_DEST"
rm "$DATASET_DEST/greater_than_256_crop.tar.gz"

# --- Copy checkpoint ---
mkdir -p "$REPO_DIR/checkpoints"
cp "$DRIVE_BASE/checkpoints/fitv2_xl.safetensors" "$REPO_DIR/checkpoints/"

# --- Extract previous workdir for resuming (optional) ---
if [ -f "$DRIVE_BASE/workdir_${PROJECT}.tar.gz" ]; then
  tar -xzf "$DRIVE_BASE/workdir_${PROJECT}.tar.gz" -C "$REPO_DIR"
fi

# --- Launch training ---
cd "$REPO_DIR"
torchrun --nnodes 1 --nproc_per_node 1 \
  train_fitv2.py \
    --project_name "$PROJECT" \
    --workdir "$WORKDIR" \
    --cfgdir "$CONFIG" \
    --seed 0 --scale_lr --allow_tf32 \
    --resume_from_checkpoint latest \
    --use_ema

# --- Save workdir back to Drive as compressed archive ---
tar -czf "$DRIVE_BASE/workdir_${PROJECT}.tar.gz" -C "$REPO_DIR" "$WORKDIR"
