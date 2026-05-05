#!/bin/bash
# Each session: copy assets from Drive to local filesystem, then launch training.
# Usage: bash tools/colab_train.sh [A|B|C|WA|WC] [DRIVE_BASE] [--save_to_drive]
#
#   A / B / C   — full training for the respective loss variant
#   WA          — warmup for Loss A (freeze everything except size_embedder)
#   WC          — warmup for Loss C (freeze everything except size_embedder + upsampler)
#
#   WA and WC write into the same workdir as their full-training counterpart (A or C).
#   After warmup, launch full training with the same variant letter — it will detect the
#   checkpoint and add --reset_optimizer automatically to start with a fresh optimizer.
#
#   --save_to_drive   tar the workdir back to Drive after training completes
#
# DRIVE_BASE defaults to /content/drive/MyDrive/FiT

set -e

LOSS="${1:?Usage: colab_train.sh [A|B|C|WA|WC] [DRIVE_BASE] [--save_to_drive]}"
DRIVE_BASE="${2:-/content/drive/MyDrive/FiT}"
SAVE_TO_DRIVE=false

for arg in "$@"; do
  if [ "$arg" = "--save_to_drive" ]; then
    SAVE_TO_DRIVE=true
  fi
done

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "=== FiT Colab Training Setup ==="
echo "Loss variant : ${LOSS^^}"
echo "Drive base   : $DRIVE_BASE"
echo "Repo dir     : $REPO_DIR"
echo "Save to Drive: $SAVE_TO_DRIVE"

IS_WARMUP=false
RESET_OPTIMIZER=false
case "${LOSS^^}" in
  A)  CONFIG="configs/fitv2/config_fitv2_xl_colab_a.yaml";       BASE_LOSS="a" ;;
  B)  CONFIG="configs/fitv2/config_fitv2_xl_colab_b.yaml";       BASE_LOSS="b" ;;
  C)  CONFIG="configs/fitv2/config_fitv2_xl_colab_c.yaml";       BASE_LOSS="c" ;;
  WA) CONFIG="configs/fitv2/config_fitv2_xl_colab_warmup_a.yaml"; BASE_LOSS="a"; IS_WARMUP=true ;;
  WC) CONFIG="configs/fitv2/config_fitv2_xl_colab_warmup.yaml";   BASE_LOSS="c"; IS_WARMUP=true ;;
  *)  echo "Unknown loss variant '${LOSS}'. Must be A, B, C, WA, or WC."; exit 1 ;;
esac

# Warmup and full training share the same project/workdir so checkpoints carry over.
PROJECT="fitv2_xl_colab_${BASE_LOSS}"
WORKDIR="workdir/${PROJECT}"

# When launching full training (A/C) and a warmup checkpoint exists, reset the optimizer
# so it is freshly initialised over all parameters.
if [ "$IS_WARMUP" = false ] && [ -d "$REPO_DIR/$WORKDIR/checkpoints" ]; then
  CKPTS=$(ls "$REPO_DIR/$WORKDIR/checkpoints" 2>/dev/null | grep "^checkpoint" | wc -l)
  if [ "$CKPTS" -gt 0 ]; then
    RESET_OPTIMIZER=true
    echo "Warmup checkpoint detected — will reset optimizer for full training."
  fi
fi

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

# --- Restore previous workdir from Drive (optional) ---
DRIVE_ARCHIVE="$DRIVE_BASE/workdir_${PROJECT}.tar.gz"
if [ -f "$DRIVE_ARCHIVE" ]; then
  echo "[3/4] Restoring previous workdir from Drive..."
  tar -xzf "$DRIVE_ARCHIVE" -C "$REPO_DIR"
  echo "[3/4] Workdir restored."
else
  echo "[3/4] No previous workdir found, starting fresh."
fi

# --- Build extra flags ---
EXTRA_FLAGS=""
case "${LOSS^^}" in
  WA) EXTRA_FLAGS="--freeze_new_layers size_embedder" ;;
  WC) EXTRA_FLAGS="--freeze_new_layers size_embedder,upsampler" ;;
  *)  EXTRA_FLAGS="--use_ema" ;;
esac
if [ "$RESET_OPTIMIZER" = true ]; then
  EXTRA_FLAGS="$EXTRA_FLAGS --reset_optimizer"
fi

# --- Launch training ---
echo ""
echo "[4/4] Launching training..."
cd "$REPO_DIR"
export TORCHINDUCTOR_CACHE_DIR="$DRIVE_BASE/inductor_cache"

torchrun --nnodes 1 --nproc_per_node 1 \
  train_fitv2.py \
    --project_name "$PROJECT" \
    --workdir "$WORKDIR" \
    --cfgdir "$CONFIG" \
    --seed 0 --scale_lr --allow_tf32 \
    --resume_from_checkpoint latest \
    $EXTRA_FLAGS

# --- Optionally save workdir back to Drive ---
echo ""
if [ "$SAVE_TO_DRIVE" = true ]; then
  echo "Saving workdir to Drive..."
  tar -czf "$DRIVE_BASE/workdir_${PROJECT}.tar.gz" -C "$REPO_DIR" "$WORKDIR"
  echo "Workdir saved to $DRIVE_BASE/workdir_${PROJECT}.tar.gz"
else
  echo "Workdir NOT saved to Drive (pass --save_to_drive to persist it)."
  echo "Local workdir: $REPO_DIR/$WORKDIR"
fi

# --- After warmup: hint at next step ---
if [ "$IS_WARMUP" = true ]; then
  echo ""
  echo "Warmup complete. To continue with full training, run:"
  echo "  bash tools/colab_train.sh ${BASE_LOSS^^} $DRIVE_BASE [--save_to_drive]"
  echo "(The optimizer will be reset automatically; model and EMA weights carry over.)"
fi
