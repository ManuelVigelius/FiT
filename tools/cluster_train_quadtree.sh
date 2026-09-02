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
# Rendezvous port. Base differs from cluster_train.sh (29500) so a quadtree run
# and a base-model run can share a node; the per-variant offset below then lets
# TWO quadtree variants run at once — e.g. the pyramid-decoder ablation pair on
# GPUs 0-3 and 4-7. Override with MASTER_PORT=... for anything else.
MASTER_PORT_BASE="${MASTER_PORT:-29501}"

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

# Port offset per VARIANT FAMILY, so two different variants can train side by
# side on one node. Keyed on BASE_VARIANT rather than the raw argument so a
# warmup and the full run it exec's into share a port — they never overlap in
# time, and reusing it avoids stranding a second port per experiment.
case "$BASE_VARIANT" in
  learned)  PORT_OFFSET=0 ;;
  baseline) PORT_OFFSET=1 ;;
  gt)       PORT_OFFSET=2 ;;
  prior)    PORT_OFFSET=3 ;;
esac
MASTER_PORT=$((MASTER_PORT_BASE + PORT_OFFSET))

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
echo "GPUs         : $GPUS_PER_NODE (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1,2,3,4})"
echo "Rendezvous   : localhost:$MASTER_PORT"
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

# `ln -sfn` via a temp name + mv is atomic and idempotent, which matters when two
# variants are launched simultaneously on one node (the ablation pair on GPUs 0-3
# and 4-7). A plain [ ! -e ] guard is a race: both see it missing, the loser's
# `ln` fails, and `set -e` kills that run before it starts.
# -f replaces an existing link rather than failing, and -n stops it descending
# into one that points at a directory. Both matter because two variants may be
# launched at once (the ablation pair on GPUs 0-3 and 4-7): a bare `[ ! -e ] &&
# ln -s` guard is a race where the loser's `ln` fails and `set -e` kills that run
# before it starts. Every caller writes the SAME target here, so last-writer-wins
# is the correct outcome and no temp-file dance is needed.
# Not fatal on failure: `ln -f` is unlink-then-symlink (two syscalls) on BSD, so
# simultaneous callers can collide with "File exists" even though every one of
# them is writing the SAME target. Verify the end state instead of the exit code
# — if the link resolves to the intended file, the job is done, whoever won.
link_atomic() {  # link_atomic <target> <linkname>
  local i
  for i in 1 2 3; do
    ln -sfn "$1" "$2" 2>/dev/null || true
    # -e follows the link, so this is true only once it resolves to a real file.
    # A rival's unlink-then-symlink can leave a brief window where it does not;
    # retrying rides that out.
    [ -e "$2" ] && return 0
  done
  echo "ERROR: failed to create symlink $2 -> $1" >&2
  return 1
}

CHECKPOINT_LINK="$REPO_DIR/checkpoints/fitv2_xl.safetensors"
if [ "$IS_PRIOR" = false ]; then
  link_atomic "$CHECKPOINT_SRC" "$CHECKPOINT_LINK"
  echo "        base checkpoint symlink ready."
fi

# The LEARNED and BASELINE configs name checkpoints/variance_predictor.pt. It is
# NOT produced by any script in this repo's training flow, so check for it up
# front rather than failing several minutes in, after the dataset copy.
VP_LINK="$REPO_DIR/checkpoints/variance_predictor.pt"
if [ "$NEEDS_VP" = true ]; then
  # -e follows symlinks, so an existing-but-dangling link counts as missing here
  # and gets repointed rather than silently used.
  if [ ! -e "$VP_LINK" ]; then
    if [ -e "$VP_CKPT_SRC" ]; then
      link_atomic "$VP_CKPT_SRC" "$VP_LINK"
      echo "        variance predictor symlink ready."
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
# Per-variant filename: two runs starting together would otherwise truncate and
# rewrite one shared file while the other is reading it. The contents are
# identical, so this costs nothing.
OVERRIDE_CFG="$FASTDATA/data_path_override_${BASE_VARIANT}.yaml"
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

# torch.compile is OFF by default on these nodes. The quadtree path feeds the
# model a different token count nearly every step, so dynamo keeps recompiling
# for new shapes, and on this hardware some of those backward kernels want more
# shared memory than the GPU has (99KB), which is a hard compile failure:
#
#   RuntimeError: No valid triton configs. OutOfResources: out of resource:
#   shared memory, Required: 102144, Hardware limit: 101376
#
# Eager is slower but it runs. Set FIT_COMPILE=1 to try compiling again (e.g. on
# A100/H100 nodes, which have 164KB+ and do not hit this).
export FIT_COMPILE="${FIT_COMPILE:-0}"

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
