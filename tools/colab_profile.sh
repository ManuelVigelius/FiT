#!/bin/bash
# Profile a short training window with Nsight Systems.
# Usage: bash tools/colab_profile.sh [A|B|C] [DRIVE_BASE]
# DRIVE_BASE defaults to /content/drive/MyDrive/FiT
#
# The script installs nsys if needed, then runs train_fitv2.py under nsys with
# cudaProfilerApi capture range so only the annotated 30-step window is recorded.
# Kill the process after seeing "[profiler] cudaProfilerStop" in the log — the
# .nsys-rep is written to DRIVE_BASE on exit.

set -e

LOSS="${1:?Usage: colab_profile.sh [A|B|C] [DRIVE_BASE]}"
DRIVE_BASE="${2:-/content/drive/MyDrive/FiT}"
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "=== FiT Colab Profiling ==="
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

# --- Install nsys if not already present ---
NSYS="$(command -v nsys 2>/dev/null \
  || ls /usr/local/cuda-*/bin/nsys 2>/dev/null | head -1 \
  || ls /opt/nvidia/nsight-systems-*/bin/nsys 2>/dev/null | head -1 \
  || true)"

if [ -z "$NSYS" ]; then
  echo "[1/2] Installing Nsight Systems..."
  CUDA_VER=$(nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+' | head -1)
  CUDA_MAJOR=$(echo "$CUDA_VER" | cut -d. -f1)
  CUDA_MINOR=$(echo "$CUDA_VER" | cut -d. -f2)
  PKG="cuda-nsight-systems-${CUDA_MAJOR}-${CUDA_MINOR}"
  echo "  Installing $PKG for CUDA $CUDA_VER..."
  apt-get install -y "$PKG"
  NSYS="$(command -v nsys 2>/dev/null \
    || ls /usr/local/cuda-*/bin/nsys 2>/dev/null | head -1 \
    || ls /opt/nvidia/nsight-systems-*/bin/nsys 2>/dev/null | head -1)"
  echo "  nsys found at: $NSYS"
else
  echo "[1/2] nsys already available at: $NSYS"
fi

# --- Launch profiling ---
echo ""
echo "[2/2] Launching profiler..."
echo "  Kill after seeing: [profiler] cudaProfilerStop"
echo "  Output: $DRIVE_BASE/profile_${PROJECT}.nsys-rep"
echo ""

cd "$REPO_DIR"
export TORCHINDUCTOR_CACHE_DIR="$DRIVE_BASE/inductor_cache"

"$NSYS" profile \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop \
  --trace=cuda,nvtx,osrt \
  --output="$DRIVE_BASE/profile_${PROJECT}" \
  torchrun --nnodes 1 --nproc_per_node 1 \
    train_fitv2.py \
      --project_name "$PROJECT" \
      --workdir "$WORKDIR" \
      --cfgdir "$CONFIG" \
      --seed 0 --scale_lr --allow_tf32 \
      --resume_from_checkpoint latest \
      --use_ema
