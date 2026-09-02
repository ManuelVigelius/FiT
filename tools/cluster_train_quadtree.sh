#!/bin/bash
# Cluster training runner for the QUADTREE FiT variant on visinf lab nodes.
# Sibling of cluster_train.sh, which covers the base fitv2 model.
#
# Usage: bash tools/cluster_train_quadtree.sh [LEARNED|BASELINE|GT|WLEARNED|WGT|PRIOR] [--reset-optimizer]
#
#   LEARNED   — full training, residual pyramid encoder + decoder, new head
#   BASELINE  — full training, no learned compression (mean-pooled leaves,
#               pretrained head). The reference point for the ablation.
#   GT        — full training, oracle quadtree from the TRUE variance of the
#               clean latent. Diagnostic only: it reads x0, which does not exist
#               at sampling time. Needs no variance predictor.
#   WLEARNED  — warmup for LEARNED (freeze the pretrained trunk + the seeded
#               base_proj; train size_embedder, final_layer and the pyramid),
#               then automatically continues into full LEARNED training
#   WGT       — same warmup, for the GT config
#   PRIOR     — the small autoregressive prior over quadtree STRUCTURES
#               (quadtree_prior.train). Separate model, single GPU, no warmup.
#   --reset-optimizer — reset optimizer/scheduler state and the step counter on
#               start (used automatically when transitioning warmup -> full)
#
# BASELINE has no warmup variant on purpose: it builds no pyramid, so the only
# new tensors are size_embedder and final_layer, and the pretrained head is
# loaded rather than fresh. There is nothing substantial to warm up.
#
# Storage layout (identical to cluster_train.sh):
#   /visinf/home/mb_mvigel/FiT                          code (this repo)
#   /visinf/projects_students/mb_mvigel/datasets/...    dataset source (permanent)
#   /visinf/projects_students/mb_mvigel/checkpoints/    base checkpoint + final exports
#   /visinf/projects_students/mb_mvigel/envs/fit        conda environment
#   /fastdata/mb_mvigel/fit/                            dataset copy + workdir during training

set -e

VARIANT="${1:?Usage: cluster_train_quadtree.sh [LEARNED|BASELINE|GT|WLEARNED|WGT|PRIOR] [--reset-optimizer]}"
RESET_OPTIMIZER=false
if [ "${2}" = "--reset-optimizer" ]; then
  RESET_OPTIMIZER=true
  echo "Reset optimizer flag set."
fi

REPO_DIR="/visinf/home/mb_mvigel/FiT"
DATASET_SRC="/visinf/projects_students/mb_mvigel/datasets"
CHECKPOINT_SRC="/visinf/projects_students/mb_mvigel/checkpoints/model_ema.safetensors"
# The frozen VariancePredictor that decides the quadtree for the LEARNED and
# BASELINE configs. Produced by running
#   python -m fit.quadtree_compression.variance_prediction ... --save <path>
# The GT config does not use it at all.
VP_CKPT_SRC="/visinf/projects_students/mb_mvigel/checkpoints/variance_predictor.pt"
CONDA_ENV="/visinf/projects_students/mb_mvigel/envs/fit"

FASTDATA="/fastdata/mb_mvigel/fit"
DATASET_FAST="$FASTDATA/datasets"
WORKDIR_FAST="$FASTDATA/workdir"

PERSISTENT_OUT="/visinf/projects_students/mb_mvigel"

GPUS_PER_NODE=4
MASTER_PORT=29501     # differs from cluster_train.sh so both can run at once

CFG_DIR="$REPO_DIR/configs/quadtree"
WARMUP_OVERLAY="$CFG_DIR/warmup_overlay.yaml"

IS_WARMUP=false
IS_PRIOR=false
NEEDS_VP=true
case "${VARIANT^^}" in
  LEARNED)  CONFIG="$CFG_DIR/config_quadtree_xl_learned.yaml";      BASE_VARIANT="learned" ;;
  BASELINE) CONFIG="$CFG_DIR/config_quadtree_xl_baseline.yaml";     BASE_VARIANT="baseline" ;;
  GT)       CONFIG="$CFG_DIR/config_quadtree_xl_gt_variance.yaml";  BASE_VARIANT="gt"; NEEDS_VP=false ;;
  WLEARNED) CONFIG="$CFG_DIR/config_quadtree_xl_learned.yaml";      BASE_VARIANT="learned"; IS_WARMUP=true ;;
  WGT)      CONFIG="$CFG_DIR/config_quadtree_xl_gt_variance.yaml";  BASE_VARIANT="gt"; IS_WARMUP=true; NEEDS_VP=false ;;
  PRIOR)    CONFIG="$REPO_DIR/configs/quadtree_prior/config_prior_s.yaml"
            BASE_VARIANT="prior"; IS_PRIOR=true; NEEDS_VP=false ;;
  *) echo "Unknown variant '${VARIANT}'. Must be LEARNED, BASELINE, GT, WLEARNED, WGT, or PRIOR."; exit 1 ;;
esac

# The warmup writes into the SAME project dir as the full run that follows it, so
# the full stage can pick the warmup's checkpoint up with --resume_from_checkpoint
# latest + --reset_optimizer. That is the whole handoff.
PROJECT="quadtree_xl_cluster_${BASE_VARIANT}"
[ "$IS_PRIOR" = true ] && PROJECT="quadtree_prior_s_cluster"

# The trainers write to os.path.join(--workdir, project_name), so the actual
# output dir is "$WORKDIR_FAST/$PROJECT". Single source of truth.
WORKDIR="$WORKDIR_FAST/$PROJECT"
WORKDIR_OUT="$PERSISTENT_OUT/workdir/${PROJECT}"

echo "=== Quadtree FiT Cluster Training Setup ==="
echo "Variant      : ${VARIANT^^}"
echo "Config       : $CONFIG"
echo "Project      : $PROJECT"
echo "Workdir      : $WORKDIR"
echo "GPUs         : $GPUS_PER_NODE"
echo ""

# Fail fast: if the persistent destination already exists, the post-training mv
# would nest the new workdir inside it. Refuse to start rather than burn GPU
# hours and fail at the move. Skipped for the warmup, which does not move
# anything — it hands off to the full stage in place.
if [ "$IS_WARMUP" = false ] && [ -e "$WORKDIR_OUT" ]; then
  echo "ERROR: persistent destination already exists: $WORKDIR_OUT" >&2
  echo "       Delete or rename it before training, e.g.:" >&2
  echo "         rm -r \"$WORKDIR_OUT\"" >&2
  exit 1
fi

# --- Activate conda environment ---
echo "[1/3] Activating conda environment..."
export PATH="$CONDA_ENV/bin:$PATH"
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH="$(find "$CONDA_ENV/lib/python3.11/site-packages/nvidia" -name "lib" -type d | tr '\n' ':')$CONDA_ENV/lib/python3.11/site-packages/triton/backends/nvidia/lib/cupti:$LD_LIBRARY_PATH"

# --- Copy dataset to fastdata ---
if [ ! -d "$DATASET_FAST" ] || [ -z "$(ls -A "$DATASET_FAST" 2>/dev/null)" ]; then
  echo "[2/3] Copying dataset to fastdata (this may take a while)..."
  mkdir -p "$DATASET_FAST"
  cp -r "$DATASET_SRC/." "$DATASET_FAST/"
  echo "[2/3] Dataset ready at $DATASET_FAST"
else
  echo "[2/3] Dataset already present at $DATASET_FAST, skipping copy."
fi

# --- Symlink checkpoints into repo (configs resolve them relative to repo root) ---
echo "[3/3] Checking checkpoint symlinks..."
mkdir -p "$REPO_DIR/checkpoints"

CHECKPOINT_LINK="$REPO_DIR/checkpoints/fitv2_xl.safetensors"
if [ "$IS_PRIOR" = false ] && [ ! -e "$CHECKPOINT_LINK" ]; then
  ln -s "$CHECKPOINT_SRC" "$CHECKPOINT_LINK"
  echo "        base checkpoint symlink created."
fi

# The LEARNED and BASELINE configs name checkpoints/variance_predictor.pt. It is
# NOT produced by any script in this repo's training flow, so check for it up
# front rather than failing several minutes in, after the dataset copy.
VP_LINK="$REPO_DIR/checkpoints/variance_predictor.pt"
if [ "$NEEDS_VP" = true ]; then
  if [ ! -e "$VP_LINK" ]; then
    if [ -e "$VP_CKPT_SRC" ]; then
      ln -s "$VP_CKPT_SRC" "$VP_LINK"
      echo "        variance predictor symlink created."
    else
      echo "ERROR: ${VARIANT^^} needs a trained VariancePredictor, but neither" >&2
      echo "         $VP_LINK" >&2
      echo "       nor  $VP_CKPT_SRC" >&2
      echo "       exists. Train one first:" >&2
      echo "         python -m fit.quadtree_compression.variance_prediction --save <path>" >&2
      echo "       (The GT variant needs no predictor and can run right now.)" >&2
      exit 1
    fi
  else
    echo "        variance predictor symlink already present."
  fi
fi
echo "[3/3] Checkpoints ready."

# --- Write a one-line config override to point data_path at fastdata ---
# Same key path as the base model, so this file is interchangeable with the one
# cluster_train.sh writes.
OVERRIDE_CFG="$FASTDATA/data_path_override.yaml"
mkdir -p "$FASTDATA"
cat > "$OVERRIDE_CFG" <<YAML
data:
  params:
    train:
      data_path: ${DATASET_FAST}
YAML

# --- Build the config list and extra flags ---
# Config precedence is left-to-right, so the warmup overlay lands on top of the
# base config and the data_path override on top of everything.
CONFIGS="$CONFIG"
EXTRA_FLAGS="--use_ema --ema_decay 0.9995"

if [ "$IS_WARMUP" = true ]; then
  CONFIGS="$CONFIG $WARMUP_OVERLAY"
  # 'default' = freeze what the checkpoint supplied, train what is new. The
  # trainer derives the actual list from the config's own wiring.
  EXTRA_FLAGS="$EXTRA_FLAGS --freeze_pretrained default"
fi
CONFIGS="$CONFIGS $OVERRIDE_CFG"

if [ "$RESET_OPTIMIZER" = true ]; then
  EXTRA_FLAGS="$EXTRA_FLAGS --reset_optimizer"
fi

# --- Launch training ---
echo ""
mkdir -p "$WORKDIR"
cd "$REPO_DIR"

export WANDB_MODE=offline

# Restrict to specific physical GPUs (override with CUDA_VISIBLE_DEVICES=... in
# the environment). torch sees the selected devices as cuda:0..N-1, so
# nproc_per_node must match the count listed here.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2,3,4}"

if [ "$IS_PRIOR" = true ]; then
  # The prior is a ~10M-param model on batches of 256; one GPU is plenty and it
  # has its own entry point and default project name.
  echo "Launching quadtree PRIOR training on 1 GPU..."
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES%%,*}"
  python -m quadtree_prior.train \
      --project_name "$PROJECT" \
      --workdir "$WORKDIR_FAST" \
      --cfgdir $CONFIGS \
      --seed 0 --scale_lr --allow_tf32 \
      --resume_from_checkpoint latest
else
  echo "Launching training with $GPUS_PER_NODE GPUs..."
  python -m torch.distributed.run \
    --nnodes 1 \
    --nproc_per_node "$GPUS_PER_NODE" \
    --rdzv_id $RANDOM \
    --rdzv_backend c10d \
    --rdzv_endpoint "localhost:$MASTER_PORT" \
    train_quadtree.py \
      --project_name "$PROJECT" \
      --workdir "$WORKDIR_FAST" \
      --cfgdir $CONFIGS \
      --seed 0 --scale_lr --allow_tf32 \
      --resume_from_checkpoint latest \
      $EXTRA_FLAGS
fi

# --- After warmup: automatically continue with full training ---
# --reset-optimizer is what makes the full stage load the warmup's weights while
# discarding its optimizer/scheduler state, so the fresh optimizer covers every
# parameter (including the ones that were frozen a moment ago).
if [ "$IS_WARMUP" = true ]; then
  echo ""
  echo "Warmup complete. Launching full ${BASE_VARIANT^^} training (optimizer reset)..."
  exec bash "$0" "${BASE_VARIANT^^}" --reset-optimizer
fi

# --- Move workdir to persistent storage ---
echo ""
echo "Moving workdir to persistent storage: $WORKDIR_OUT ..."
mkdir -p "$PERSISTENT_OUT/workdir"
mv "$WORKDIR" "$WORKDIR_OUT"
echo "Workdir moved."

# --- Export final model weights ---
# The accelerate checkpoints bundle optimizer + scheduler state. Extract just the
# model weights as plain safetensors for inference / further fine-tuning.
#
# NOTE: unlike the base model, a quadtree run saves THREE prepared modules per
# checkpoint — model.safetensors, then (with EMA) model_1 = EMA and model_2 =
# compressor. Sampling needs the compressor as well as the transformer, so all of
# them are exported under explicit names.
LATEST_CKPT=$(ls -d "$WORKDIR_OUT/checkpoints/"*checkpoint-* 2>/dev/null \
  | sort -t- -k2 -n | tail -1)
if [ -n "$LATEST_CKPT" ]; then
  CKPT_OUT="$PERSISTENT_OUT/checkpoints/${PROJECT}"
  mkdir -p "$CKPT_OUT"
  echo "Exporting final weights to $CKPT_OUT ..."
  # The prior trains a single model (no compressor); the quadtree runs prepare a
  # compressor as a third module. Both trainers default --use_ema to True.
  HAS_COMPRESSOR=1
  [ "$IS_PRIOR" = true ] && HAS_COMPRESSOR=0
  python - <<EOF
import os, shutil

ckpt = "$LATEST_CKPT"
out  = "$CKPT_OUT"
has_compressor = bool(int("$HAS_COMPRESSOR"))

# prepare_model() order in train_quadtree.py: model, ema_model, compressor.
# quadtree_prior/train.py prepares only model, ema_model. accelerate numbers the
# saved files in exactly that order.
names = ["model", "ema"] + (["compressor"] if has_compressor else [])

for idx, name in enumerate(names):
    stem = "model" if idx == 0 else f"model_{idx}"
    for ext, prefix in ((".safetensors", ""), (".bin", "pytorch_")):
        src = os.path.join(ckpt, f"{prefix}{stem}{ext}")
        if os.path.exists(src):
            dst = os.path.join(out, f"{name}{ext}")
            shutil.copy2(src, dst)
            print(f"  {name}: {dst}")
            break
    else:
        print(f"  {name}: {stem} not found in checkpoint, skipping")
EOF
  echo "Export done."
else
  echo "No checkpoint found to export."
fi

echo "Done."
