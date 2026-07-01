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
#   bash run_all.sh --gpu 1         # 指定用哪张 GPU（默认 0）
#   bash run_all.sh --quick         # 快速版（跳过尺寸缩放/autotune）
#   bash run_all.sh --ncu           # 额外跑 Nsight Compute 剖析（需 root，很慢）
#   bash run_all.sh --sizes "2048 4096"   # 自定义 headline bench 尺寸
#
# 产物：result/<时间戳>/  （各步 .log + 遥测 csv + 00_summary.md / .txt），并软链到 result/latest
#
# 架构自适应：
#   - CUDA core (FP32/BF16) 全机型可跑
#   - Tensor Core：Ampere/Ada(sm_80/86/89) 只跑 WMMA(tc_01-03)；
#     Hopper(sm_90) 额外跑 WGMMA+TMA(tc_04) 与 FP8(tc_05)
# ============================================================================
set -uo pipefail   # 故意不加 -e：单个 kernel 失败不应中断整轮，由 run_test 记录状态

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ---------------------------------------------------------------------------
# 参数
# ---------------------------------------------------------------------------
GPU=0
DO_LOCK=0
DO_NCU=0
DO_SCALING=1
DO_AUTOTUNE=1
BENCH_SIZE=4096
VERIFY_SIZE=2048
SCALING_SIZES="1024 2048 4096 8192"
while [ $# -gt 0 ]; do
  case "$1" in
    --gpu)      GPU="$2"; shift 2 ;;
    --lock)     DO_LOCK=1; shift ;;
    --ncu)      DO_NCU=1; shift ;;
    --quick)    DO_SCALING=0; DO_AUTOTUNE=0; shift ;;
    --sizes)    BENCH_SIZE_SET="$2"; shift 2 ;;   # 多尺寸 headline（空格分隔）
    -h|--help)  grep -E '^#( |$)' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "未知参数: $1（用 -h 看帮助）"; exit 1 ;;
  esac
done

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULT_DIR="${SCRIPT_DIR}/result"
RUN_DIR="${RESULT_DIR}/${TIMESTAMP}"
mkdir -p "$RUN_DIR"
TIMINGS_FILE="${RUN_DIR}/00_timings.tsv"
printf 'test\tseconds\tstatus\n' > "$TIMINGS_FILE"

log() { echo "$@"; }
run_test() {  # run_test <name> <cmd...>
  local name="$1"; shift
  local cmd="$*"
  log ">>> [$(date +%H:%M:%S)] ${name}"
  local t0=$(date +%s) status
  if eval "$cmd" > "${RUN_DIR}/${name}.log" 2>&1; then status="ok"; else status="FAIL"; fi
  local dt=$(( $(date +%s) - t0 ))
  printf '%s\t%s\t%s\n' "$name" "$dt" "$status" >> "$TIMINGS_FILE"
  log "<<< ${name} ${status} (${dt}s)"
}

# ---------------------------------------------------------------------------
# 0. 预检：nvidia-smi / nvcc / make
# ---------------------------------------------------------------------------
log "=============================================="
log " CUDA_GEMM 全量测试   结果目录: ${RUN_DIR}"
log " 开始: $(date -Iseconds)"
log "=============================================="

command -v nvidia-smi >/dev/null 2>&1 || { echo "错误: 找不到 nvidia-smi（需 NVIDIA 驱动）"; exit 1; }
if ! command -v nvcc >/dev/null 2>&1; then
  for d in /usr/local/cuda/bin /usr/local/cuda-*/bin; do [ -x "$d/nvcc" ] && export PATH="$d:$PATH" && break; done
fi
command -v nvcc >/dev/null 2>&1 || { echo "错误: 找不到 nvcc（需 CUDA Toolkit；把 /usr/local/cuda/bin 加进 PATH）"; exit 1; }
command -v make >/dev/null 2>&1 || { echo "错误: 找不到 make"; exit 1; }
JOBS=$(nproc 2>/dev/null || echo 4)

# ---------------------------------------------------------------------------
# 1. 检测 GPU 架构 → 选 ARCH / TC_HOPPER
# ---------------------------------------------------------------------------
CC=$(nvidia-smi -i "$GPU" --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | tr -d ' ')
GPU_NAME=$(nvidia-smi -i "$GPU" --query-gpu=name --format=csv,noheader 2>/dev/null)
MAXCLK=$(nvidia-smi -i "$GPU" --query-gpu=clocks.max.sm --format=csv,noheader 2>/dev/null | grep -oE '[0-9]+' | head -1)
[ -z "$CC" ] && { echo "错误: 无法读取 GPU $GPU 的 compute_cap"; exit 1; }
MAJOR=${CC%.*}; MINOR=${CC#*.}

case "$MAJOR" in
  9)        SM_ARCH="sm_90a"; TC_HOPPER=1 ;;                  # Hopper: WGMMA/TMA/FP8
  10|11|12) SM_ARCH="sm_${MAJOR}${MINOR}"; TC_HOPPER=0 ;;     # Blackwell+: 仅 WMMA（本仓库 WGMMA 是 sm_90a 专写）
  *)        SM_ARCH="sm_${MAJOR}${MINOR}"; TC_HOPPER=0 ;;     # Ampere/Ada/Volta/Turing: 仅 WMMA
esac
[ "$TC_HOPPER" = 1 ] && TC_MAX=5 || TC_MAX=3

export CUDA_VISIBLE_DEVICES="$GPU"   # 后续 bench/verify 里 device 0 = 物理 GPU $GPU

log ""
log " GPU:        $GPU_NAME"
log " ComputeCap: $CC  →  ARCH=-arch=$SM_ARCH  TC_HOPPER=$TC_HOPPER (tensor 用例 1-$TC_MAX)"
log " 额定 max SM 时钟: ${MAXCLK:-未知} MHz"
log ""

# ---------------------------------------------------------------------------
# 2. 可选锁频（需 sudo；trap 保证退出时解锁，绝不残留改动全局状态）
# ---------------------------------------------------------------------------
CLOCK_POLICY="default(开箱自适应boost)"
if [ "$DO_LOCK" = 1 ]; then
  if sudo -n true 2>/dev/null && [ -n "$MAXCLK" ]; then
    sudo -n nvidia-smi -i "$GPU" -lgc "$MAXCLK" >/dev/null 2>&1 \
      && { CLOCK_POLICY="locked@${MAXCLK}MHz(额定boost)"; trap "sudo -n nvidia-smi -i $GPU -rgc >/dev/null 2>&1" EXIT; log " 已锁频 $GPU → ${MAXCLK} MHz（退出自动解锁）"; } \
      || log " 警告: 锁频失败，改用默认时钟"
  else
    log " 警告: --lock 需要免密 sudo 且能读到 max 时钟；当前不满足，改用默认时钟"
  fi
fi
log " 时钟策略: $CLOCK_POLICY"
echo "$CLOCK_POLICY" > "${RUN_DIR}/00_clock_policy.txt"

# ---------------------------------------------------------------------------
# 3. 后台遥测（只读 nvidia-smi dmon：时钟/功耗/温度/throttle，安全无需 root）
# ---------------------------------------------------------------------------
MON_CSV="${RUN_DIR}/telemetry_dmon.txt"
nvidia-smi dmon -i "$GPU" -s pct -d 1 -o T > "$MON_CSV" 2>&1 &
MON_PID=$!
trap "kill $MON_PID 2>/dev/null; { [ \"$DO_LOCK\" = 1 ] && sudo -n nvidia-smi -i $GPU -rgc >/dev/null 2>&1; } " EXIT

# ---------------------------------------------------------------------------
# 4. 平台指纹
# ---------------------------------------------------------------------------
{
  echo "=== Platform Fingerprint ==="; date -Iseconds
  echo "--- GPU ---"; nvidia-smi -i "$GPU" --query-gpu=name,compute_cap,memory.total,clocks.max.sm,clocks.max.mem,power.limit --format=csv 2>&1
  echo "--- GPU 详情 ---"; nvidia-smi -i "$GPU" 2>&1 | head -15
  echo "--- CUDA ---"; nvcc --version 2>&1 | tail -4
  echo "--- CPU/MEM ---"; lscpu 2>/dev/null | grep -E "Model name|^CPU\(s\)"; free -h 2>/dev/null | head -2
  echo "--- torch ---"; python3 -c "import torch;print('torch',torch.__version__,'cuda',torch.version.cuda);p=torch.cuda.get_device_properties(0);print(p.name,f'CC{p.major}.{p.minor}',p.multi_processor_count,'SM', f'{p.total_memory/1e9:.0f}GB')" 2>&1
} > "${RUN_DIR}/01_fingerprint.txt" 2>&1
run_test "01_fingerprint" "cat ${RUN_DIR}/01_fingerprint.txt"

# ---------------------------------------------------------------------------
# 5. 编译（自动用检测到的 ARCH / TC_HOPPER）
# ---------------------------------------------------------------------------
run_test "02_build" "make clean; make ARCH=-arch=$SM_ARCH -j$JOBS && make tc ARCH=-arch=$SM_ARCH TC_HOPPER=$TC_HOPPER -j$JOBS"
if [ ! -x kernels/cuda_core/bench ] || [ ! -x kernels/tensor_core/bench ]; then
  echo "错误: 编译失败，见 ${RUN_DIR}/02_build.log"; tail -20 "${RUN_DIR}/02_build.log"; exit 1
fi

CC_B=kernels/cuda_core/bench
CC_V=kernels/cuda_core/verify
BF_B=kernels/cuda_core/bench_bf16
BF_V=kernels/cuda_core/verify_bf16
TC_B=kernels/tensor_core/bench
TC_V=kernels/tensor_core/verify

# ---------------------------------------------------------------------------
# 6. 正确性（对拍 cuBLAS，全部 PASS 才算通过）
# ---------------------------------------------------------------------------
run_test "03_verify_fp32" "for id in \$(seq 0 12); do echo -n \"id=\$id \"; $CC_V \$id $VERIFY_SIZE $VERIFY_SIZE $VERIFY_SIZE 2>&1 | grep -E 'PASS|FAIL' | tail -1; done"
run_test "04_verify_bf16" "for id in \$(seq 0 12); do echo -n \"id=\$id \"; $BF_V \$id $VERIFY_SIZE $VERIFY_SIZE $VERIFY_SIZE 2>&1 | grep -E 'PASS|FAIL' | tail -1; done"
run_test "05_verify_tc"   "for id in \$(seq 1 $TC_MAX); do $TC_V \$id $VERIFY_SIZE $VERIFY_SIZE $VERIFY_SIZE 2>&1; done"

# ---------------------------------------------------------------------------
# 7. Headline 性能 @ BENCH_SIZE（可多尺寸）
# ---------------------------------------------------------------------------
SIZES_TO_RUN="${BENCH_SIZE_SET:-$BENCH_SIZE}"
for SZ in $SIZES_TO_RUN; do
  run_test "06_bench_fp32_${SZ}" "for id in \$(seq 0 12); do $CC_B \$id $SZ $SZ $SZ; done"
  run_test "07_bench_bf16_${SZ}" "for id in \$(seq 0 12); do $BF_B \$id $SZ $SZ $SZ; done"
  run_test "08_bench_tc_${SZ}"   "for id in \$(seq 1 $TC_MAX); do $TC_B \$id $SZ $SZ $SZ; done"
done

# ---------------------------------------------------------------------------
# 8. 尺寸缩放（cuBLAS + 代表性手写 kernel）
# ---------------------------------------------------------------------------
if [ "$DO_SCALING" = 1 ]; then
  run_test "09_scaling" "for sz in $SCALING_SIZES; do echo \"==== size=\$sz ====\"; \
    echo -n '[FP32 cublas]      '; $CC_B 0 \$sz \$sz \$sz  | grep -oE 'GFLOPS=[0-9.]+'; \
    echo -n '[FP32 double_buf]  '; $CC_B 12 \$sz \$sz \$sz | grep -oE 'GFLOPS=[0-9.]+'; \
    echo -n '[BF16 cublas]      '; $BF_B 0 \$sz \$sz \$sz  | grep -oE 'GFLOPS=[0-9.]+'; \
    echo -n '[WMMA tc_03_pipe]  '; $TC_B 3 \$sz \$sz \$sz  | grep -oE 'GFLOPS=[0-9.]+'; \
    done"
fi

# ---------------------------------------------------------------------------
# 9. autotune 配置扫描
# ---------------------------------------------------------------------------
[ "$DO_AUTOTUNE" = 1 ] && run_test "10_autotune" "$CC_B autotune $BENCH_SIZE $BENCH_SIZE $BENCH_SIZE"

# ---------------------------------------------------------------------------
# 10. 可选：Nsight Compute 剖析（需 root，很慢）
# ---------------------------------------------------------------------------
if [ "$DO_NCU" = 1 ]; then
  if sudo -n true 2>/dev/null && command -v ncu >/dev/null 2>&1; then
    run_test "11_ncu" "sudo -n ncu --set full -k regex:'naive_kernel|smem_kernel|double_buffer_kernel|wmma_' -s 1 -c 1 -f -o ${RUN_DIR}/ncu_profile $CC_B 12 2048 2048 2048; ncu -i ${RUN_DIR}/ncu_profile.ncu-rep --page details 2>/dev/null | head -200"
  else
    log " 跳过 --ncu：需要免密 sudo 且 PATH 里有 ncu"
  fi
fi

# ---------------------------------------------------------------------------
# 11. 汇总报告（解析日志 + 设备算力 → GFLOPS / %峰值 / %cuBLAS / MFU）
# ---------------------------------------------------------------------------
kill $MON_PID 2>/dev/null
run_test "99_summary" "python3 ${SCRIPT_DIR}/scripts/gemm_summary.py ${RUN_DIR} '$GPU_NAME' '$CC' '$CLOCK_POLICY' '${MAXCLK:-0}'"

printf 'TOTAL\t%s\tok\n' "$(cat "$TIMINGS_FILE" | awk -F'\t' 'NR>1{s+=$2}END{print s}')" >> "$TIMINGS_FILE"

ln -sfn "$RUN_DIR" "${RESULT_DIR}/latest"
log ""
log "=============================================="
log " 完成！结果目录: ${RUN_DIR}"
log " 汇总报告: ${RUN_DIR}/00_summary.md"
log " 快捷软链: ${RESULT_DIR}/latest"
log "=============================================="
[ -f "${RUN_DIR}/00_summary.txt" ] && cat "${RUN_DIR}/00_summary.txt"
