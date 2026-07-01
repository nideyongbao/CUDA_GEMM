#!/usr/bin/env bash
# ============================================================================
# CUDA_GEMM 一键全量测试入口（编译 + 正确性 + 性能 + 遥测 + 汇总报告）
#
# 目标：在任意新机器上（云租赁、已装基础 CUDA + torch 的镜像）一条命令跑通，
#       自动识别 GPU 架构、编译、跑全量 GEMM 测试、按机型给出完整结果与对比。
#
# 用法：
#   bash run_all.sh                 # 默认：自动检测架构，开箱默认时钟，全量测试
#   bash run_all.sh --lock          # 额外锁频到额定 boost（需 sudo，测可复现的满频上限）
#   bash run_all.sh --both          # 一次跑【默认+锁频】两轮并出对比（归档基线标准；锁频步需 sudo）
#   bash run_all.sh --gpu 1         # 指定用哪张 GPU（默认 0）
#   bash run_all.sh --quick         # 快速版（跳过尺寸缩放/autotune）
#   bash run_all.sh --ncu           # 额外跑 Nsight Compute 剖析（需 root，很慢）
#   bash run_all.sh --sizes "2048 4096"   # 自定义 headline bench 尺寸
#
# 产物（瞬时快照，均落在 result/<时间戳>/，软链 result/latest）：
#   00_summary.md/.txt   汇总报告          00_timings.tsv   每步耗时
#   00_clock_policy.txt  时钟策略          telemetry_dmon.txt 全程遥测(时钟/功耗/温度)
#   01_fingerprint.txt   平台指纹          02_build.log     编译输出
#   verify/<engine>/<id>_<name>.log        每个 kernel 单独的正确性日志
#   bench/<engine>/<id>_<name>_<size>.log  每个 kernel 单独的性能日志
#   scaling.log  autotune.log  profiling/(--ncu 时)
#
# 设计约定：代码目录(kernels/include/scripts/Makefile/run_all.sh)只放代码；
#   逐次执行日志与 profiling 等【瞬时产物】统一进 result/<时间戳>/ 做对比快照。
#   baselines/{cuda_core,tensor_core,a800}/ 是各机型策展好的【参考 ncu 基线】(文档引用)。
#
# 架构自适应：CUDA core 全机型；Tensor Core Ampere/Ada 仅 WMMA(tc_01-03)，
#   Hopper 额外 WGMMA+TMA(tc_04)/FP8(tc_05)。
# ============================================================================
set -uo pipefail   # 故意不加 -e：单个 kernel 失败不应中断整轮

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ---------------------------------------------------------------------------
# 参数
# ---------------------------------------------------------------------------
GPU=0; DO_LOCK=0; DO_NCU=0; DO_SCALING=1; DO_AUTOTUNE=1; DO_BOTH=0
BENCH_SIZE=4096; VERIFY_SIZE=2048; SCALING_SIZES="1024 2048 4096 8192"
BENCH_SIZE_SET=""
PASS_FLAGS=()   # --both 透传给两次子调用的标志
while [ $# -gt 0 ]; do
  case "$1" in
    --gpu)   GPU="$2"; shift 2 ;;
    --lock)  DO_LOCK=1; shift ;;
    --both)  DO_BOTH=1; shift ;;
    --ncu)   DO_NCU=1; PASS_FLAGS+=(--ncu); shift ;;
    --quick) DO_SCALING=0; DO_AUTOTUNE=0; PASS_FLAGS+=(--quick); shift ;;
    --sizes) BENCH_SIZE_SET="$2"; PASS_FLAGS+=(--sizes "$2"); shift 2 ;;
    -h|--help) grep -E '^#( |$)' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "未知参数: $1（用 -h 看帮助）"; exit 1 ;;
  esac
done

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULT_DIR="${SCRIPT_DIR}/result"

# --both：一次跑【默认 + 锁频】两轮(各自完整)，同落 <ts>_both/{default,locked}/ 并自动出对比。
# 用途：生成归档基线 / 直接看时钟策略对每个 kernel 的影响。子调用透传 --quick/--ncu/--sizes。
if [ "$DO_BOTH" = 1 ]; then
  PARENT="${RESULT_DIR}/${TIMESTAMP}_both"; mkdir -p "$PARENT"
  echo "== --both：默认 + 锁频 两轮 → $PARENT =="
  RUN_DIR_OVERRIDE="$PARENT/default" bash "$0" --gpu "$GPU" ${PASS_FLAGS[@]+"${PASS_FLAGS[@]}"}
  RUN_DIR_OVERRIDE="$PARENT/locked"  bash "$0" --gpu "$GPU" --lock ${PASS_FLAGS[@]+"${PASS_FLAGS[@]}"}
  python3 "${SCRIPT_DIR}/scripts/gemm_compare.py" "$PARENT/default" "$PARENT/locked" > "$PARENT/00_compare.md" 2>&1
  ln -sfn "$PARENT" "${RESULT_DIR}/latest"
  echo ""; echo "== --both 完成：对比报告 $PARENT/00_compare.md =="
  cat "$PARENT/00_compare.md"
  exit 0
fi

RUN_DIR="${RUN_DIR_OVERRIDE:-${RESULT_DIR}/${TIMESTAMP}}"
mkdir -p "$RUN_DIR"
TIMINGS_FILE="${RUN_DIR}/00_timings.tsv"; printf 'test\tseconds\tstatus\n' > "$TIMINGS_FILE"

log() { echo "$@"; }
# 单命令步骤（build/fingerprint/summary）：整步一个 .log
run_test() {
  local name="$1"; shift; local cmd="$*"
  log ">>> [$(date +%H:%M:%S)] ${name}"
  local t0=$(date +%s) status
  if eval "$cmd" > "${RUN_DIR}/${name}.log" 2>&1; then status="ok"; else status="FAIL"; fi
  printf '%s\t%s\t%s\n' "$name" "$(( $(date +%s) - t0 ))" "$status" >> "$TIMINGS_FILE"
  log "<<< ${name} ${status} ($(( $(date +%s) - t0 ))s)"
}
# 分组步骤（verify/bench）：整组计时，但组内每个 kernel 各写一份日志
timed() {
  local name="$1"; shift
  log ">>> [$(date +%H:%M:%S)] ${name}"
  local t0=$(date +%s) status=ok
  "$@" || status="FAIL"
  printf '%s\t%s\t%s\n' "$name" "$(( $(date +%s) - t0 ))" "$status" >> "$TIMINGS_FILE"
  log "<<< ${name} ${status} ($(( $(date +%s) - t0 ))s)"
}
# 跑一个 kernel，单独存日志： per_ex <subdir> <exe> <id> <name> <args...>
per_ex() {
  local sub="$1" exe="$2" id="$3" name="$4"; shift 4
  local dir="${RUN_DIR}/${sub}"; mkdir -p "$dir"
  "$exe" "$id" "$@" > "${dir}/$(printf '%02d' "$id")_${name}.log" 2>&1 || true
}

# ---------------------------------------------------------------------------
# 0. 预检
# ---------------------------------------------------------------------------
log "=============================================="
log " CUDA_GEMM 全量测试   结果目录: ${RUN_DIR}"
log " 开始: $(date -Iseconds)"
log "=============================================="
command -v nvidia-smi >/dev/null 2>&1 || { echo "错误: 找不到 nvidia-smi（需 NVIDIA 驱动）"; exit 1; }
command -v nvcc >/dev/null 2>&1 || { for d in /usr/local/cuda/bin /usr/local/cuda-*/bin; do [ -x "$d/nvcc" ] && export PATH="$d:$PATH" && break; done; }
command -v nvcc >/dev/null 2>&1 || { echo "错误: 找不到 nvcc（把 /usr/local/cuda/bin 加进 PATH）"; exit 1; }
command -v make >/dev/null 2>&1 || { echo "错误: 找不到 make"; exit 1; }
JOBS=$(nproc 2>/dev/null || echo 4)

# ---------------------------------------------------------------------------
# 1. 检测架构
# ---------------------------------------------------------------------------
CC=$(nvidia-smi -i "$GPU" --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | tr -d ' ')
GPU_NAME=$(nvidia-smi -i "$GPU" --query-gpu=name --format=csv,noheader 2>/dev/null)
MAXCLK=$(nvidia-smi -i "$GPU" --query-gpu=clocks.max.sm --format=csv,noheader 2>/dev/null | grep -oE '[0-9]+' | head -1)
[ -z "$CC" ] && { echo "错误: 无法读取 GPU $GPU 的 compute_cap"; exit 1; }
MAJOR=${CC%.*}; MINOR=${CC#*.}
case "$MAJOR" in
  9)          SM_ARCH="sm_90a"; TC_HOPPER=1 ;;
  10|11|12)   SM_ARCH="sm_${MAJOR}${MINOR}"; TC_HOPPER=0 ;;
  *)          SM_ARCH="sm_${MAJOR}${MINOR}"; TC_HOPPER=0 ;;
esac
[ "$TC_HOPPER" = 1 ] && TC_MAX=5 || TC_MAX=3
export CUDA_VISIBLE_DEVICES="$GPU"
log ""
log " GPU: $GPU_NAME | CC $CC → ARCH=-arch=$SM_ARCH TC_HOPPER=$TC_HOPPER (tensor 1-$TC_MAX) | max ${MAXCLK:-?}MHz"

# kernel 名（用于逐示例日志文件名；与各 bench/verify 驱动的注册表一致）
FP32_NAMES=(cublas_ref naive smem blocktiling 2Dblocktiling vectorized autotune_64x64x8_8x4 autotune_64x64x16_8x4 autotune_64x64x8_8x8 warptile warptile_vec bank_conflict double_buffer)
BF16_NAMES=(cublas_bf16 naive smem blocktiling 2Dblocktiling vectorized autotune_64x64x8_8x4 autotune_64x64x16_8x4 autotune_64x64x8_8x8 warptile warptile_vec bank_conflict double_buffer)
TC_NAMES=(_ tc01_wmma_naive tc02_wmma_smem tc03_wmma_pipe tc04_wgmma_tma_ws tc05_wgmma_fp8)  # 下标从 1 起

# ---------------------------------------------------------------------------
# 2. 可选锁频（trap 退出解锁）
# ---------------------------------------------------------------------------
CLOCK_POLICY="default(开箱自适应boost)"
if [ "$DO_LOCK" = 1 ]; then
  if sudo -n true 2>/dev/null && [ -n "$MAXCLK" ] && sudo -n nvidia-smi -i "$GPU" -lgc "$MAXCLK" >/dev/null 2>&1; then
    CLOCK_POLICY="locked@${MAXCLK}MHz(额定boost)"; log " 已锁频 GPU$GPU → ${MAXCLK}MHz（退出自动解锁）"
  else log " 警告: --lock 需免密 sudo + 可读 max 时钟；改用默认时钟"; fi
fi
log " 时钟策略: $CLOCK_POLICY"; echo "$CLOCK_POLICY" > "${RUN_DIR}/00_clock_policy.txt"

# ---------------------------------------------------------------------------
# 3. 后台遥测 + 退出清理
# ---------------------------------------------------------------------------
nvidia-smi dmon -i "$GPU" -s pct -d 1 -o T > "${RUN_DIR}/telemetry_dmon.txt" 2>&1 &
MON_PID=$!
cleanup(){ kill "$MON_PID" 2>/dev/null; [ "$DO_LOCK" = 1 ] && sudo -n nvidia-smi -i "$GPU" -rgc >/dev/null 2>&1; }
trap cleanup EXIT

# ---------------------------------------------------------------------------
# 4. 平台指纹
# ---------------------------------------------------------------------------
{
  echo "=== Platform Fingerprint ==="; date -Iseconds
  echo "--- GPU ---"; nvidia-smi -i "$GPU" --query-gpu=name,compute_cap,memory.total,clocks.max.sm,clocks.max.mem,power.limit --format=csv 2>&1
  nvidia-smi -i "$GPU" 2>&1 | head -15
  echo "--- CUDA ---"; nvcc --version 2>&1 | tail -4
  echo "--- CPU/MEM ---"; lscpu 2>/dev/null | grep -E "Model name|^CPU\(s\)"; free -h 2>/dev/null | head -2
  echo "--- torch ---"; python3 -c "import torch;print('torch',torch.__version__,'cuda',torch.version.cuda);p=torch.cuda.get_device_properties(0);print(p.name,f'CC{p.major}.{p.minor}',p.multi_processor_count,'SM',f'{p.total_memory/1e9:.0f}GB')" 2>&1
} > "${RUN_DIR}/01_fingerprint.txt" 2>&1
run_test "01_fingerprint" "cat ${RUN_DIR}/01_fingerprint.txt"

# ---------------------------------------------------------------------------
# 5. 编译
# ---------------------------------------------------------------------------
# out-of-source 构建：产物进 result/<ts>/build/，代码目录 kernels/ 保持干净
BDIR="${RUN_DIR}/build"
run_test "02_build" "make clean BUILD='$BDIR'; make ARCH=-arch=$SM_ARCH BUILD='$BDIR' -j$JOBS && make tc ARCH=-arch=$SM_ARCH TC_HOPPER=$TC_HOPPER BUILD='$BDIR' -j$JOBS"
[ -x "$BDIR/cuda_core/bench" ] && [ -x "$BDIR/tensor_core/bench" ] || { echo "错误: 编译失败，见 ${RUN_DIR}/02_build.log"; tail -20 "${RUN_DIR}/02_build.log"; exit 1; }
CC_B="$BDIR/cuda_core/bench"; CC_V="$BDIR/cuda_core/verify"
BF_B="$BDIR/cuda_core/bench_bf16"; BF_V="$BDIR/cuda_core/verify_bf16"
TC_B="$BDIR/tensor_core/bench"; TC_V="$BDIR/tensor_core/verify"

# ---------------------------------------------------------------------------
# 6. 正确性（逐 kernel 各存一份日志）
# ---------------------------------------------------------------------------
V=$VERIFY_SIZE
verify_fp32(){ for id in $(seq 0 12); do per_ex verify/cuda_core_fp32 "$CC_V" "$id" "${FP32_NAMES[$id]}" "$V" "$V" "$V"; done; }
verify_bf16(){ for id in $(seq 0 12); do per_ex verify/cuda_core_bf16 "$BF_V" "$id" "${BF16_NAMES[$id]}" "$V" "$V" "$V"; done; }
verify_tc(){   for id in $(seq 1 "$TC_MAX"); do per_ex verify/tensor_core "$TC_V" "$id" "${TC_NAMES[$id]}" "$V" "$V" "$V"; done; }
timed "03_verify_fp32" verify_fp32
timed "04_verify_bf16" verify_bf16
timed "05_verify_tc"   verify_tc

# ---------------------------------------------------------------------------
# 7. Headline 性能（逐 kernel 各存一份日志，文件名含尺寸）
# ---------------------------------------------------------------------------
SIZES_TO_RUN="${BENCH_SIZE_SET:-$BENCH_SIZE}"
bench_fp32(){ local S=$1; for id in $(seq 0 12); do per_ex "bench/cuda_core_fp32" "$CC_B" "$id" "${FP32_NAMES[$id]}_${S}" "$S" "$S" "$S"; done; }
bench_bf16(){ local S=$1; for id in $(seq 0 12); do per_ex "bench/cuda_core_bf16" "$BF_B" "$id" "${BF16_NAMES[$id]}_${S}" "$S" "$S" "$S"; done; }
bench_tc(){   local S=$1; for id in $(seq 1 "$TC_MAX"); do per_ex "bench/tensor_core" "$TC_B" "$id" "${TC_NAMES[$id]}_${S}" "$S" "$S" "$S"; done; }
for SZ in $SIZES_TO_RUN; do
  timed "06_bench_fp32_${SZ}" bench_fp32 "$SZ"
  timed "07_bench_bf16_${SZ}" bench_bf16 "$SZ"
  timed "08_bench_tc_${SZ}"   bench_tc   "$SZ"
done

# ---------------------------------------------------------------------------
# 8. 尺寸缩放（跨尺寸整体，一个 log）
# ---------------------------------------------------------------------------
if [ "$DO_SCALING" = 1 ]; then
  run_test "09_scaling" "for sz in $SCALING_SIZES; do echo \"==== size=\$sz ====\"; \
    echo -n '[FP32 cublas]     '; $CC_B 0 \$sz \$sz \$sz  | grep -oE 'GFLOPS=[0-9.]+'; \
    echo -n '[FP32 double_buf] '; $CC_B 12 \$sz \$sz \$sz | grep -oE 'GFLOPS=[0-9.]+'; \
    echo -n '[BF16 cublas]     '; $BF_B 0 \$sz \$sz \$sz  | grep -oE 'GFLOPS=[0-9.]+'; \
    echo -n '[WMMA tc_03_pipe] '; $TC_B 3 \$sz \$sz \$sz  | grep -oE 'GFLOPS=[0-9.]+'; \
    done"
fi

# ---------------------------------------------------------------------------
# 9. autotune
# ---------------------------------------------------------------------------
[ "$DO_AUTOTUNE" = 1 ] && run_test "10_autotune" "$CC_B autotune $BENCH_SIZE $BENCH_SIZE $BENCH_SIZE"

# ---------------------------------------------------------------------------
# 10. 可选 Nsight Compute（瞬时产物 → result/<ts>/baselines/，不进代码/参考树）
# ---------------------------------------------------------------------------
if [ "$DO_NCU" = 1 ]; then
  if command -v ncu >/dev/null 2>&1 || [ -x /usr/local/cuda/bin/ncu ]; then
    # 瞬时 profiling → result/<ts>/profiling/（统一走 scripts/run_ncu.sh，架构自适应）
    timed "11_ncu" bash "${SCRIPT_DIR}/scripts/run_ncu.sh" "${RUN_DIR}/profiling" 2048 "$BDIR" quick
  else log " 跳过 --ncu：找不到 ncu"; fi
fi

# ---------------------------------------------------------------------------
# 11. 汇总
# ---------------------------------------------------------------------------
kill "$MON_PID" 2>/dev/null
run_test "99_summary" "python3 ${SCRIPT_DIR}/scripts/gemm_summary.py ${RUN_DIR} '$GPU_NAME' '$CC' '$CLOCK_POLICY' '${MAXCLK:-0}'"
printf 'TOTAL\t%s\tok\n' "$(awk -F'\t' 'NR>1{s+=$2}END{print s}' "$TIMINGS_FILE")" >> "$TIMINGS_FILE"

ln -sfn "$RUN_DIR" "${RESULT_DIR}/latest"
log ""
log "=============================================="
log " 完成！结果目录: ${RUN_DIR}"
log " 逐示例日志: ${RUN_DIR}/{verify,bench}/<engine>/<id>_<name>.log"
log " 汇总报告:   ${RUN_DIR}/00_summary.md   (软链 result/latest)"
log "=============================================="
[ -f "${RUN_DIR}/00_summary.txt" ] && cat "${RUN_DIR}/00_summary.txt"
