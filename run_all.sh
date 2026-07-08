#!/usr/bin/env bash
# cuda-ops-a800 one-shot: pick a free A800, lock clocks, build + verify + bench
# (+ optional ncu) for all three operators, snapshot to result/<timestamp>/.
#
#   bash run_all.sh              # auto-pick freest GPU, build+verify+bench
#   bash run_all.sh --gpu 3      # force GPU 3
#   bash run_all.sh --ncu        # also collect ncu profiles (needs sudo)
#   bash run_all.sh --no-lock    # skip clock lock (no sudo)
set -u
ROOT="$(cd "$(dirname "$0")" && pwd)"
GPU=""; LOCK=1; NCU=0; SM_CLK=1410
while [ $# -gt 0 ]; do case "$1" in
  --gpu) GPU="$2"; shift 2;; --ncu) NCU=1; shift;; --no-lock) LOCK=0; shift;;
  *) echo "unknown arg $1"; exit 1;; esac; done

# ---- pick freest GPU by memory.used if not forced ----
if [ -z "$GPU" ]; then
  GPU=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
        | sort -t, -k2 -n | head -1 | cut -d, -f1 | tr -d ' ')
fi
USED=$(nvidia-smi -i "$GPU" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
echo "== using GPU $GPU (mem used ${USED} MiB) =="
[ "$USED" -gt 4000 ] && echo "WARNING: GPU $GPU is busy (${USED} MiB used) — timing will be contended/unreliable."
export CUDA_VISIBLE_DEVICES=$GPU

TS=$(date +%Y%m%d_%H%M%S)
OUT="$ROOT/result/$TS"; mkdir -p "$OUT"
ln -sfn "$TS" "$ROOT/result/latest"

if [ "$LOCK" = 1 ]; then
  sudo nvidia-smi -i "$GPU" -pm 1 >/dev/null 2>&1
  sudo nvidia-smi -i "$GPU" -lgc ${SM_CLK},${SM_CLK} >/dev/null 2>&1 && echo "locked SM clock ${SM_CLK} MHz" || echo "clock lock failed (no sudo?)"
fi

echo "== build ==" | tee "$OUT/00_build.log"
( cd "$ROOT" && make all ) >> "$OUT/00_build.log" 2>&1 && echo "build OK" || { echo "BUILD FAILED, see $OUT/00_build.log"; exit 1; }

VERIFY="$OUT/01_verify.log"; BENCH="$OUT/02_bench.log"; : > "$VERIFY"; : > "$BENCH"

echo "== verify (correctness) ==" | tee -a "$VERIFY"
{
  echo "### GEMM cuda_core (vs cuBLAS) @2048"; for id in 1 2 4 5 10 12; do "$ROOT"/gemm/build/cuda_core/verify $id 2048 2048 2048; done
  echo "### GEMM tensor_core (vs cuBLAS) @2048"; for id in 1 2 3 6; do "$ROOT"/gemm/build/tensor_core/verify $id 2048 2048 2048; done
  echo "### softmax (vs CPU double)"; for id in 1 2 3 4 5 6; do "$ROOT"/softmax/build/cuda_core/verify $id 1024 2048; done
  echo "### FA tensor_core TinyFA (vs CPU attn)"; for c in "fp16 2 8 512 128 0" "bf16 2 8 512 128 0" "fp16 2 8 512 64 1"; do "$ROOT"/flash_attn/build/tensor_core/verify $c; done
  echo "### FA tensor_core raw ladder fa_tc_01..04 (vs CPU attn)"; for id in 1 2 3 4; do "$ROOT"/flash_attn/build/tensor_core/fa_tc_verify $id 2 8 512 0; "$ROOT"/flash_attn/build/tensor_core/fa_tc_verify $id 2 8 512 1; done
  echo "### FA cuda_core scaffold (vs CPU attn)"; for id in 1 2; do "$ROOT"/flash_attn/build/cuda_core/verify $id 2 4 256 64 0; done
} 2>&1 | tee -a "$VERIFY"

echo "== bench (performance @ locked clock) ==" | tee -a "$BENCH"
{
  echo "### GEMM FP32 cuda_core @4096"; for id in 1 2 3 4 5 9 10 11 12; do "$ROOT"/gemm/build/cuda_core/bench $id 4096 4096 4096; done
  echo "### GEMM tensor_core @4096"; for id in 1 2 3 6; do "$ROOT"/gemm/build/tensor_core/bench $id 4096 4096 4096; done
  echo "### softmax @8192x8192"; for id in 1 2 3 4 5 6; do "$ROOT"/softmax/build/cuda_core/bench $id 8192 8192; done
  echo "### FA tensor_core TinyFA (CuTe endpoint) B2 H32 S4096 D128"; for c in "fp16 2 32 4096 128 0" "bf16 2 32 4096 128 0" "fp16 2 32 4096 128 1"; do "$ROOT"/flash_attn/build/tensor_core/bench $c; done
  echo "### FA tensor_core raw ladder fa_tc_01..04 B2 H32 S4096 D128 fp16"; for id in 1 2 3 4; do "$ROOT"/flash_attn/build/tensor_core/fa_tc_bench $id 2 32 4096 0; done
  echo "### FA cuda_core scaffold B2 H16 S2048 D64"; for id in 1 2; do "$ROOT"/flash_attn/build/cuda_core/bench $id 2 16 2048 64 0; done
} 2>&1 | tee -a "$BENCH"

if [ "$NCU" = 1 ]; then
  echo "== ncu profiling (subset) =="
  bash "$ROOT/common/run_ncu.sh" "$GPU" "$OUT" 2>&1 | tee "$OUT/03_ncu.log"
fi

echo "== done. snapshot: $OUT =="
[ "$LOCK" = 1 ] && sudo nvidia-smi -i "$GPU" -rgc >/dev/null 2>&1
