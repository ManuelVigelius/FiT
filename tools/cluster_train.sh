#!/bin/bash
# Cluster training runner for FiT on visinf lab nodes.
# Usage: bash tools/cluster_train.sh [A|B|C|WA|WC]
#
#   A / B / C   — full training for the respective loss variant
#   WA          — warmup for Loss A (freeze everything except size_embedder)
#   WC          — warmup for Loss C (freeze everything except size_embedder + upsampler)
#
#   WA/WC share a workdir with their full counterpart (A/C). After warmup, run the
#   full variant — it detects the warmup checkpoint and resets the optimizer automatically.
#
# Storage layout:
#   /visinf/home/mb_mvigel/FiT                          code (this repo)
#   /visinf/projects_students/mb_mvigel/datasets/...    dataset source (permanent)
#   /visinf/projects_students/mb_mvigel/checkpoints/    base checkpoint + final exports
#   /visinf/projects_students/mb_mvigel/envs/fit        conda environment
#   /fastdata/mb_mvigel/fit/                            dataset copy + workdir during training
#                                                        (dataset deleted after training)

set -e

LOSS="${1:?Usage: cluster_train.sh [A|B|C|WA|WC]}"

REPO_DIR="/visinf/home/mb_mvigel/FiT"
DATASET_SRC="/visinf/projects_students/mb_mvigel/datasets/greater_than_256_crop"
CHECKPOINT_SRC="/visinf/projects_students/mb_mvigel/checkpoints/model_ema.safetensors"
CONDA_ENV="/visinf/projects_students/mb_mvigel/envs/fit"

FASTDATA="/fastdata/mb_mvigel/fit"
DATASET_FAST="$FASTDATA/datasets/imagenet1k_latents_256_sd_vae_ft_ema"
WORKDIR_FAST="$FASTDATA/workdir"

PERSISTENT_OUT="/visinf/projects_students/mb_mvigel"

GPUS_PER_NODE=4
MASTER_PORT=29500

echo "=== FiT Cluster Training Setup ==="
echo "Loss variant : ${LOSS^^}"
echo "Repo dir     : $REPO_DIR"
echo "Fastdata     : $FASTDATA"
echo "GPUs         : $GPUS_PER_NODE"

IS_WARMUP=false
RESET_OPTIMIZER=false
case "${LOSS^^}" in
  A)  CONFIG="$REPO_DIR/configs/fitv2/config_fitv2_xl_colab_a.yaml";       BASE_LOSS="a" ;;
  B)  CONFIG="$REPO_DIR/configs/fitv2/config_fitv2_xl_colab_b.yaml";       BASE_LOSS="b" ;;
  C)  CONFIG="$REPO_DIR/configs/fitv2/config_fitv2_xl_colab_c.yaml";       BASE_LOSS="c" ;;
  WA) CONFIG="$REPO_DIR/configs/fitv2/config_fitv2_xl_colab_warmup_a.yaml"; BASE_LOSS="a"; IS_WARMUP=true ;;
  WC) CONFIG="$REPO_DIR/configs/fitv2/config_fitv2_xl_colab_warmup.yaml";   BASE_LOSS="c"; IS_WARMUP=true ;;
  *)  echo "Unknown loss variant '${LOSS}'. Must be A, B, C, WA, or WC."; exit 1 ;;
esac

PROJECT="fitv2_xl_cluster_${BASE_LOSS}"
WORKDIR="$WORKDIR_FAST/$PROJECT"

if [ "$IS_WARMUP" = false ] && [ -d "$WORKDIR/checkpoints" ]; then
  CKPTS=$(ls "$WORKDIR/checkpoints" 2>/dev/null | grep "^checkpoint" | wc -l)
  if [ "$CKPTS" -gt 0 ]; then
    RESET_OPTIMIZER=true
    echo "Warmup checkpoint detected — will reset optimizer for full training."
  fi
fi

echo "Config       : $CONFIG"
echo "Project      : $PROJECT"
echo "Workdir      : $WORKDIR"
echo ""

# --- Activate conda environment ---
echo "[1/3] Activating conda environment..."
# Derive the conda base from the env path (two levels up from the env dir)
CONDA_BASE="$(dirname "$(dirname "$CONDA_ENV")")"
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

# --- Copy dataset to fastdata ---
if [ ! -d "$DATASET_FAST" ] || [ -z "$(ls -A "$DATASET_FAST" 2>/dev/null)" ]; then
  echo "[2/3] Copying dataset to fastdata (this may take a while)..."
  mkdir -p "$DATASET_FAST"
  cp -r "$DATASET_SRC/." "$DATASET_FAST/"
  echo "[2/3] Dataset ready at $DATASET_FAST"
else
  echo "[2/3] Dataset already present at $DATASET_FAST, skipping copy."
fi

# --- Symlink checkpoint into repo (train script resolves relative to repo root) ---
CHECKPOINT_LINK="$REPO_DIR/checkpoints/fitv2_xl.safetensors"
if [ ! -e "$CHECKPOINT_LINK" ]; then
  echo "[3/3] Creating checkpoint symlink..."
  mkdir -p "$REPO_DIR/checkpoints"
  ln -s "$CHECKPOINT_SRC" "$CHECKPOINT_LINK"
  echo "[3/3] Checkpoint symlink ready."
else
  echo "[3/3] Checkpoint symlink already present, skipping."
fi

# --- Build extra flags ---
EXTRA_FLAGS=""
case "${LOSS^^}" in
  WA) EXTRA_FLAGS="--freeze_new_layers size_embedder" ;;
  WC) EXTRA_FLAGS="--freeze_new_layers size_embedder,upsampler" ;;
  *)  EXTRA_FLAGS="--use_ema --ema_decay 0.9995" ;;
esac
if [ "$RESET_OPTIMIZER" = true ]; then
  EXTRA_FLAGS="$EXTRA_FLAGS --reset_optimizer"
fi

# --- Write a one-line config override to point data_path at fastdata ---
OVERRIDE_CFG="$FASTDATA/data_path_override.yaml"
mkdir -p "$FASTDATA"
cat > "$OVERRIDE_CFG" <<YAML
data:
  params:
    train:
      data_path: ${DATASET_FAST}
YAML

# --- Launch training ---
echo ""
echo "Launching training with $GPUS_PER_NODE GPUs..."
mkdir -p "$WORKDIR"
cd "$REPO_DIR"

torchrun \
  --nnodes 1 \
  --nproc_per_node "$GPUS_PER_NODE" \
  --rdzv_id $RANDOM \
  --rdzv_backend c10d \
  --rdzv_endpoint "localhost:$MASTER_PORT" \
  train_fitv2.py \
    --project_name "$PROJECT" \
    --workdir "$WORKDIR_FAST/workdir" \
    --cfgdir "$CONFIG" "$OVERRIDE_CFG" \
    --seed 0 --scale_lr --allow_tf32 \
    --resume_from_checkpoint latest \
    $EXTRA_FLAGS

# --- After warmup: automatically continue with full training ---
if [ "$IS_WARMUP" = true ]; then
  echo ""
  echo "Warmup complete. Starting full training for loss ${BASE_LOSS^^}..."
  exec bash "$0" "${BASE_LOSS^^}"
fi

# --- Move workdir to persistent storage ---
echo ""
WORKDIR_OUT="$PERSISTENT_OUT/workdir/${PROJECT}"
echo "Moving workdir to persistent storage: $WORKDIR_OUT ..."
mkdir -p "$PERSISTENT_OUT/workdir"
mv "$WORKDIR" "$WORKDIR_OUT"
echo "Workdir moved."

# --- Export final model weights ---
# The accelerate checkpoints bundle optimizer + scheduler state. Extract just the
# model (and EMA) weights as plain safetensors for inference / further fine-tuning.
LATEST_CKPT=$(ls -d "$WORKDIR_OUT/checkpoints/checkpoint-"* 2>/dev/null \
  | sort -t- -k2 -n | tail -1)
if [ -n "$LATEST_CKPT" ]; then
  CKPT_OUT="$PERSISTENT_OUT/checkpoints/${PROJECT}"
  mkdir -p "$CKPT_OUT"
  echo "Exporting final model weights to $CKPT_OUT ..."
  python - <<EOF
import os, glob, shutil

ckpt = "$LATEST_CKPT"
out  = "$CKPT_OUT"

for name, pattern in [("model", "model.safetensors"), ("ema", "ema_model.safetensors")]:
    src = os.path.join(ckpt, pattern)
    if not os.path.exists(src):
        candidates = glob.glob(os.path.join(ckpt, f"**/{pattern}"), recursive=True)
        if not candidates:
            print(f"  {name}: not found in checkpoint, skipping")
            continue
        src = candidates[0]
    dst = os.path.join(out, pattern)
    shutil.copy2(src, dst)
    print(f"  {name}: {dst}")
EOF
  echo "Export done."
else
  echo "No checkpoint found to export."
fi

echo "Done."
