#!/bin/bash
# Cluster FID runner for the QUADTREE FiT variants on visinf lab nodes.
# Generates N_IMAGES samples per condition with generate_quadtree_images.py,
# split across the selected GPUs, then verifies the runs and scores them with
# clean-fid against the same reference as tools/cluster_fid.sh.
#
# Usage: bash tools/cluster_quadtree_fid.sh [CONDITION ...]
#   CONDITION  baseline | learned | gt | prior   (default: baseline learned gt)
#
# Examples:
#   bash tools/cluster_quadtree_fid.sh
#   CUDA_VISIBLE_DEVICES=4,5 bash tools/cluster_quadtree_fid.sh learned
#   N_IMAGES=64 OUT_DIR=/tmp/qt_test bash tools/cluster_quadtree_fid.sh learned
#
# GPUs: one process per GPU listed in CUDA_VISIBLE_DEVICES (default 0,1,2,3),
# each pinned to its own device. Plain processes, no torch.distributed: the
# shards share nothing but the output folder.
#
# Reproducibility: images depend only on the settings recorded in
# <OUT_DIR>/<condition>/run.json (seed, image count, batch size, steps, CFG,
# checkpoint step, EMA), NOT on how many GPUs produced them. A rerun with the
# same settings resumes an interrupted run; a rerun with different settings into
# the same OUT_DIR is refused. Labels and initial noise match measure_fid.py for
# the same SEED and N_IMAGES.
#
# Outputs:
#   <OUT_DIR>/<condition>/NNNNNN.png      samples
#   <OUT_DIR>/<condition>/run.json        settings the images depend on
#   <OUT_DIR>/fid_results.json            FID per condition (+ its run.json)
#   <OUT_DIR>/logs/<condition>_shard<i>.log

set -euo pipefail

REPO_DIR="/visinf/home/mb_mvigel/FiT"
CONDA_ENV="/visinf/projects_students/mb_mvigel/envs/fit"
PERSISTENT_OUT="/visinf/projects_students/mb_mvigel"

# --- Run settings (override via env) ---
N_IMAGES="${N_IMAGES:-10000}"
BATCH_SIZE="${BATCH_SIZE:-64}"   # part of the run record; keep it fixed per OUT_DIR
NUM_STEPS="${NUM_STEPS:-16}"
CFG_SCALE="${CFG_SCALE:-4.0}"
STEP="${STEP:-8000}"             # diffusion checkpoint step
SEED="${SEED:-42}"
OUT_DIR="${OUT_DIR:-$PERSISTENT_OUT/fid/quadtree_eval}"
EXTRA_ARGS="${EXTRA_ARGS:-}"     # e.g. "--no-ema"

# Same clean-fid reference as cluster_fid.sh. It must already be built there:
# this script only scores against it.
export FID_REF_NAME="${FID_REF_NAME:-fit_imagenet256}"
export FID_MODE="${FID_MODE:-clean}"

if [ "$#" -gt 0 ]; then
  CONDITIONS=("$@")
else
  CONDITIONS=(baseline learned gt)
fi

# Never touch GPUs outside this list (other people's jobs may run there).
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
IFS=',' read -ra GPUS <<< "$CUDA_VISIBLE_DEVICES"
NUM_SHARDS=${#GPUS[@]}

LOG_DIR="$OUT_DIR/logs"
mkdir -p "$LOG_DIR"

echo "=== FiT Quadtree FID ==="
echo "Conditions : ${CONDITIONS[*]}"
echo "GPUs       : ${GPUS[*]} ($NUM_SHARDS shards)"
echo "Images     : $N_IMAGES  (batch $BATCH_SIZE, steps $NUM_STEPS, CFG $CFG_SCALE, ckpt step $STEP, seed $SEED)"
echo "Reference  : clean-fid '$FID_REF_NAME' (mode=$FID_MODE)"
echo "Output dir : $OUT_DIR"
echo ""

# --- Activate conda environment (identical to cluster_fid.sh) ---
export PATH="$CONDA_ENV/bin:$PATH"
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH="$(find "$CONDA_ENV/lib/python3.11/site-packages/nvidia" -name "lib" -type d | tr '\n' ':')$CONDA_ENV/lib/python3.11/site-packages/triton/backends/nvidia/lib/cupti:${LD_LIBRARY_PATH:-}"

# --no-deps so pip cannot replace pinned training deps (see cluster_fid.sh).
if ! python -c "import cleanfid" 2>/dev/null; then
  echo "Installing clean-fid into the conda env (--no-deps)..."
  python -m pip install --no-input --no-deps clean-fid
fi

cd "$REPO_DIR"

COMMON_ARGS=(
  --n-images "$N_IMAGES" --batch-size "$BATCH_SIZE" --num-steps "$NUM_STEPS"
  --cfg-scale "$CFG_SCALE" --step "$STEP" --seed "$SEED" --out "$OUT_DIR"
)
# shellcheck disable=SC2206  # word-splitting EXTRA_ARGS is intended
COMMON_ARGS+=($EXTRA_ARGS)

# Ctrl-C / kill: take the shard processes down too, instead of leaving them
# running on the GPUs.
trap 'echo "Interrupted, stopping shards..."; kill $(jobs -p) 2>/dev/null; exit 130' INT TERM

# --- Generate: conditions one after another, each spread over all GPUs ---
for cond in "${CONDITIONS[@]}"; do
  echo "[$cond] generating on $NUM_SHARDS GPUs (logs: $LOG_DIR/${cond}_shard*.log)"
  pids=()
  for i in "${!GPUS[@]}"; do
    CUDA_VISIBLE_DEVICES="${GPUS[$i]}" python generate_quadtree_images.py \
      --condition "$cond" "${COMMON_ARGS[@]}" \
      --num-shards "$NUM_SHARDS" --shard-id "$i" \
      > "$LOG_DIR/${cond}_shard${i}.log" 2>&1 &
    pids+=($!)
  done
  failed=()
  for i in "${!pids[@]}"; do
    wait "${pids[$i]}" || failed+=("$i")
  done
  if [ "${#failed[@]}" -gt 0 ]; then
    echo "[$cond] shard(s) ${failed[*]} failed. Last log lines:"
    for i in "${failed[@]}"; do
      echo "--- $LOG_DIR/${cond}_shard${i}.log"
      tail -n 20 "$LOG_DIR/${cond}_shard${i}.log"
    done
    echo "Fix the cause and rerun the same command: finished batches are kept."
    exit 1
  fi
  echo "[$cond] done."
done

# --- Verify completeness and score (one GPU is enough for Inception) ---
FID_FLAG="--fid"
if [ "$N_IMAGES" -lt 2048 ]; then
  echo "N_IMAGES < 2048: skipping FID (covariance would be singular)."
  FID_FLAG=""
fi
CUDA_VISIBLE_DEVICES="${GPUS[0]}" python generate_quadtree_images.py \
  --merge $FID_FLAG --condition "${CONDITIONS[@]}" --out "$OUT_DIR"

echo ""
if [ -n "$FID_FLAG" ]; then
  echo "Done. Results in $OUT_DIR/fid_results.json"
else
  echo "Done. Samples under $OUT_DIR"
fi
