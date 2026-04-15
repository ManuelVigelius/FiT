#!/bin/bash
# Each session: copy assets from Drive to local filesystem, then launch training.
# Usage: bash tools/colab_train.sh [A|B|C] [DRIVE_BASE]
# DRIVE_BASE defaults to /content/drive/MyDrive/FiT

set -e

LOSS="${1:?Usage: colab_train.sh [A|B|C] [DRIVE_BASE]}"
DRIVE_BASE="${2:-/content/drive/MyDrive/FiT}"
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "=== FiT Colab Training Setup ==="
echo "Loss variant : ${LOSS^^}"
echo "Drive base   : $DRIVE_BASE"
echo "Repo dir     : $REPO_DIR"

case "${LOSS^^}" in
  A) CONFIG="configs/fitv2/config_fitv2_xl_colab_a.yaml" ;;
  B) CONFIG="configs/fitv2/config_fitv2_xl_colab_b.yaml" ;;
  C) CONFIG="configs/fitv2/config_fitv2_xl_colab_c.yaml" ;;
  *) echo "Unknown loss variant '${LOSS}'. Must be A, B, or C."; exit 1 ;;
esac

PROJECT="fitv2_xl_colab_${LOSS,,}"
WORKDIR="workdir/${PROJECT}"

echo "Config       : $CONFIG"
echo "Project      : $PROJECT"
echo ""

# --- Stream tarball directly from Drive and extract (no intermediate copy) ---
DATASET_DEST="$REPO_DIR/datasets/imagenet1k_latents_256_sd_vae_ft_ema"
if [ ! -d "$DATASET_DEST" ] || [ -z "$(ls -A "$DATASET_DEST" 2>/dev/null)" ]; then
  echo "[1/4] Installing pigz..."
  apt-get install -y pigz
  echo "[1/4] Streaming and extracting dataset from Drive..."
  mkdir -p "$DATASET_DEST"
  tar -xf "$DRIVE_BASE/datasets/imagenet1k_latents_256_sd_vae_ft_ema/greater_than_256_crop.tar.gz" \
      -C "$DATASET_DEST" --use-compress-program=pigz
  echo "[1/4] Dataset ready."
else
  echo "[1/4] Dataset already present, skipping extraction."
fi

# --- Copy checkpoint ---
CHECKPOINT="$REPO_DIR/checkpoints/fitv2_xl.safetensors"
if [ ! -f "$CHECKPOINT" ]; then
  echo "[2/4] Copying base checkpoint from Drive..."
  mkdir -p "$REPO_DIR/checkpoints"
  cp "$DRIVE_BASE/checkpoints/fitv2_xl.safetensors" "$REPO_DIR/checkpoints/"
  echo "[2/4] Checkpoint ready."
else
  echo "[2/4] Checkpoint already present, skipping copy."
fi

# --- Extract previous workdir for resuming (optional) ---
if [ -f "$DRIVE_BASE/workdir_${PROJECT}.tar.gz" ]; then
  echo "[3/4] Restoring previous workdir from Drive..."
  tar -xzf "$DRIVE_BASE/workdir_${PROJECT}.tar.gz" -C "$REPO_DIR"
  echo "[3/4] Workdir restored."
else
  echo "[3/4] No previous workdir found, starting fresh."
fi

# --- Launch training ---
echo ""
echo "[4/4] Launching training..."
cd "$REPO_DIR"
# Persist the torch.compile / Inductor kernel cache across sessions to avoid
# re-paying the max-autotune warmup cost each time.
export TORCHINDUCTOR_CACHE_DIR="$DRIVE_BASE/inductor_cache"
torchrun --nnodes 1 --nproc_per_node 1 \
  train_fitv2.py \
    --project_name "$PROJECT" \
    --workdir "$WORKDIR" \
    --cfgdir "$CONFIG" \
    --seed 0 --scale_lr --allow_tf32 \
    --resume_from_checkpoint latest \
    --use_ema

# --- Save workdir back to Drive as compressed archive ---
echo ""
echo "Training complete. Saving workdir to Drive..."
tar -czf "$DRIVE_BASE/workdir_${PROJECT}.tar.gz" -C "$REPO_DIR" "$WORKDIR"
echo "Workdir saved to $DRIVE_BASE/workdir_${PROJECT}.tar.gz"
