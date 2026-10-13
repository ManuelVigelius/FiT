#!/bin/bash
# Wait until N GPUs have been free for a while, then run a command on them.
#
# Usage: bash tools/wait_for_gpus.sh N [command ...]
#
#   With a command: runs it with CUDA_VISIBLE_DEVICES set to the N GPUs.
#   Without one:    prints the GPUs and exits.
#
# "Free" means no compute process on the GPU at all; memory usage is not a
# reliable signal (an idle process can hold memory at 0% utilisation). A GPU
# only counts once it has been free at every check across STABLE_INTERVALS
# intervals of INTERVAL seconds (default: 3 x 5 min, i.e. 4 checks over 15 min),
# so a short gap between someone else's jobs is not mistaken for a free GPU.
# Busy at any check resets that GPU's streak.
#
# Example — start training as soon as 4 GPUs have been free for 15 minutes:
#   nohup bash tools/wait_for_gpus.sh 4 \
#       bash tools/cluster_train_quadtree.sh WGT_STRUCT > wgt_struct.log 2>&1 &

set -euo pipefail

N="${1:?Usage: wait_for_gpus.sh N [command ...]}"
shift
INTERVAL="${INTERVAL:-300}"
STABLE_INTERVALS="${STABLE_INTERVALS:-3}"
NEEDED_CHECKS=$((STABLE_INTERVALS + 1))

declare -A streak   # GPU index -> consecutive checks it was free

while true; do
  busy=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | sort -u)
  ready=()
  while IFS=', ' read -r idx uuid; do
    if grep -qF "$uuid" <<< "$busy"; then
      streak[$idx]=0
    else
      streak[$idx]=$(( ${streak[$idx]:-0} + 1 ))
    fi
    [ "${streak[$idx]}" -ge "$NEEDED_CHECKS" ] && ready+=("$idx")
  done < <(nvidia-smi --query-gpu=index,uuid --format=csv,noheader)

  status=""
  for idx in $(printf '%s\n' "${!streak[@]}" | sort -n); do
    status+=" $idx:${streak[$idx]}"
  done
  echo "$(date '+%F %T') free-check streaks (need $NEEDED_CHECKS):$status"

  if [ "${#ready[@]}" -ge "$N" ]; then
    gpus=$(printf '%s\n' "${ready[@]}" | head -n "$N" | paste -sd,)
    echo "$(date '+%F %T') GPUs $gpus free for $STABLE_INTERVALS x ${INTERVAL}s"
    if [ "$#" -eq 0 ]; then
      echo "$gpus"
      exit 0
    fi
    CUDA_VISIBLE_DEVICES="$gpus" exec "$@"
  fi
  sleep "$INTERVAL"
done
