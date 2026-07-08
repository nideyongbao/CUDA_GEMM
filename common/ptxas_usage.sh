#!/usr/bin/env bash
# ptxas resource tracker — recompile ONE kernel source with -Xptxas=-v and print
# registers/thread, spills, smem, cmem. Static, no-GPU, deterministic. Many rungs
# (warp-tiling, double-buffering, multi-MMA-tile FA) are fundamentally a register-
# pressure story; this is the zero-GPU health metric for that. Spills = the wall.
#   usage: ptxas_usage.sh <src.cu> [extra nvcc flags e.g. -I foo -DBAR]
set -u
SRC="${1:?usage: ptxas_usage.sh <src.cu> [nvcc flags...]}"; shift || true
NVCC="${NVCC:-nvcc}"
ARCH="${ARCH:--arch=sm_80}"
echo "== ptxas resource usage: $SRC =="
"$NVCC" $ARCH -O3 -std=c++17 -Xptxas=-v -Xptxas=-warn-spills -Xptxas=-warn-lmem-usage \
  -c "$SRC" -o /dev/null "$@" 2>&1 \
  | grep -E 'Function properties|registers|spill|smem|cmem|bytes stack frame|Compiling entry' \
  | sed 's/^/  /'
