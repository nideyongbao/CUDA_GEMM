#!/usr/bin/env bash
# ncu profiler — collects `ncu --set full` for one representative rung per operator
# and dumps the committed text report into each operator's baselines/. Needs sudo
# (GPU perf counters). Binary .ncu-rep stays local (gitignored); .details.txt is kept.
#   usage: run_ncu.sh <gpu> <outdir>
set -u
GPU="${1:-0}"; OUT="${2:-/tmp/ncu_out}"; mkdir -p "$OUT"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
NCU="/usr/local/cuda-12.8/bin/ncu"
export CUDA_VISIBLE_DEVICES=$GPU
prof(){ # name outfile-dir  cmd...
  local name="$1"; local dir="$2"; shift 2
  mkdir -p "$dir"
  echo "  ncu: $name"
  sudo -E "$NCU" --set full -c 1 -s 0 -f -o "$OUT/$name" "$@" >/dev/null 2>&1
  sudo -E "$NCU" --set full -c 1 -s 0 --page details "$@" > "$dir/$name.details.txt" 2>/dev/null
}
# one telling rung per engine (extend as needed)
prof gemm_cc_10_doublebuffer  "$ROOT/gemm/baselines/cuda_core"    "$ROOT/gemm/build/cuda_core/bench" 10 4096 4096 4096
prof gemm_tc_06_mma_pipe      "$ROOT/gemm/baselines/tensor_core"  "$ROOT/gemm/build/tensor_core/bench" 6 4096 4096 4096
prof softmax_sc_02_block      "$ROOT/softmax/baselines/cuda_core" "$ROOT/softmax/build/cuda_core/bench" 2 8192 8192
prof softmax_sc_05_online     "$ROOT/softmax/baselines/cuda_core" "$ROOT/softmax/build/cuda_core/bench" 5 8192 8192
prof fa_tc_hopper_wgmma       "$ROOT/flash_attn/baselines/tensor_core" "$ROOT/flash_attn/build/tensor_core/bench" bf16 2 32 4096 128 0
prof fa_cc_02_tiled           "$ROOT/flash_attn/baselines/cuda_core"   "$ROOT/flash_attn/build/cuda_core/bench" 2 2 16 2048 64 0
echo "ncu done -> reports under */baselines/, binaries in $OUT"
