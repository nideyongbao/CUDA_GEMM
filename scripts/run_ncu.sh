#!/usr/bin/env bash
# ============================================================================
# 统一 ncu 剖析器（架构自适应）：对各 kernel 做 `ncu --set full` 逐核剖析，
# 输出 <name>.ncu-rep + <name>.details.txt 到指定目录。取代旧的
# profiling/{cuda_core,tensor_core}/run_ncu.sh 与 profiling/a800/run_ncu_a800.sh。
#
# 用法: scripts/run_ncu.sh <OUTDIR> [SZ] [BUILD_DIR] [quick|full]
#   OUTDIR     输出目录（如 result/<ts>/profiling，或 baselines/<机型> 生成参考基线）
#   SZ         剖析尺寸（默认 2048，与各 SUMMARY 同口径）
#   BUILD_DIR  可执行文件所在（默认 build；需先 make ... BUILD=<该目录>）
#   quick|full quick=代表性 6 核（默认）；full=全量(FP32×13 + 张量核 tc_01..)
#
# 原理：ncu 用 kernel replay 把同一 launch 重放多遍(--set full≈39 pass)采计数器；
#   -s 1 -c 1 跳过第 1 次、只剖析第 2 次(warmup) launch。采计数器需 root（sudo）。
# 架构自适应：Ampere/Ada 只剖析 WMMA(tc_01-03)；Hopper 额外 tc_04(WGMMA)/tc_05(FP8)。
# ============================================================================
set -uo pipefail
OUTDIR=${1:?用法: run_ncu.sh <OUTDIR> [SZ] [BUILD_DIR] [quick|full]}
SZ=${2:-2048}; BUILD=${3:-build}; MODE=${4:-quick}
mkdir -p "$OUTDIR"

NCU=$(command -v ncu 2>/dev/null || echo /usr/local/cuda/bin/ncu)
[ -x "$NCU" ] || { echo "找不到 ncu"; exit 1; }
SUDO=""; sudo -n true 2>/dev/null && SUDO="sudo -n" || echo "警告: 无免密 sudo，ncu 采计数器可能失败"
CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')
HOP=0; [ "${CC%.*}" = "9" ] && HOP=1
CB="$BUILD/cuda_core/bench"; TB="$BUILD/tensor_core/bench"
[ -x "$CB" ] || { echo "找不到 $CB —— 先 make ARCH=... BUILD=$BUILD 与 make tc ..."; exit 1; }

prof(){  # prof <name> <kernel_regex> <exe> <id>
  echo "  ncu $1 (~/$2/) @ ${SZ}^3"
  $SUDO $NCU --set full -k "regex:$2" -s 1 -c 1 -f -o "$OUTDIR/$1" "$3" "$4" "$SZ" "$SZ" "$SZ" > "$OUTDIR/$1.run.log" 2>&1
  "$NCU" -i "$OUTDIR/$1.ncu-rep" --page details > "$OUTDIR/$1.details.txt" 2>/dev/null || true
}

echo "== ncu 剖析 (CC $CC, mode=$MODE, out=$OUTDIR) =="
if [ "$MODE" = "full" ]; then
  prof cc_00_cublas        gemm                 "$CB" 0
  prof cc_01_naive         naive_kernel         "$CB" 1
  prof cc_02_smem          smem_kernel          "$CB" 2
  prof cc_03_blocktiling   blocktiling_kernel   "$CB" 3
  prof cc_04_2Dblocktiling Dblocktiling_kernel  "$CB" 4
  prof cc_05_vectorized    vectorized_kernel    "$CB" 5
  prof cc_07_warptile      warptile_kernel      "$CB" 9
  prof cc_08_warptile_vec  warptile_vec_kernel  "$CB" 10
  prof cc_09_bankconflict  bank_conflict_kernel "$CB" 11
  prof cc_10_doublebuffer  double_buffer_kernel "$CB" 12
else
  prof cc_01_naive         naive_kernel         "$CB" 1
  prof cc_02_smem          smem_kernel          "$CB" 2
  prof cc_10_doublebuffer  double_buffer_kernel "$CB" 12
fi
prof tc_01_wmma_naive wmma_naive_kernel "$TB" 1
prof tc_02_wmma_smem  wmma_smem_kernel  "$TB" 2
prof tc_03_wmma_pipe  wmma_pipe_kernel  "$TB" 3
if [ "$HOP" = "1" ]; then
  prof tc_04_wgmma_tma_ws wgmma_kernel     "$TB" 4
  prof tc_05_wgmma_fp8    wgmma_fp8_kernel "$TB" 5
fi
$SUDO chown -R "$(id -u):$(id -g)" "$OUTDIR" 2>/dev/null || true
echo "ALL_NCU_DONE → $OUTDIR"
