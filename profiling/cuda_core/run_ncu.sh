#!/bin/bash
# =============================================================================
# CUDA core 侧 ncu 剖析（与 tensor_core 的 profiling/tensor_core/run_ncu.sh 对称的另一半）。
# 对每一个 CUDA core 用例各做一次 --set full 剖析：
#   · FP32   : kernels/cuda_core/bench      <id 0-12>   13 个（含 cublas 基线 + 3 autotune 配置）
#   · BF16   : kernels/cuda_core/bench_bf16 <id>         9 个手写 kernel（bf16 输入 + FP32 累加）
#
# 原理：ncu 用 kernel replay 把同一次 launch 重放多遍(--set full≈39 passes)，每遍采一部分
#   硬件计数器 → 只剖析 1 次有代表性的 launch：-s 1 -c 1 = 跳过第 1 次、剖析第 2 次(warmup)。
#   ./bench <id> 内部 2 warmup + 10 timed = 12 次同核 launch，正好取第 2 次。
#   -k regex:<核名> 只抓目标 __global__（cublas 内部核名不固定，本机 FP32=sm80_xmma_gemm_*,
#   BF16=nvjet_*）。采计数器需 root → sudo -n /usr/local/cuda/bin/ncu（本机免密）。
#
# 用法: profiling/cuda_core/run_ncu.sh [SZ]      # SZ 默认 2048（与 tensor_core 表一致）
# 产物: profiling/cuda_core/<name>.{ncu-rep,details.txt,run.log}（cc_* / cc_bf16_* 前缀）
# =============================================================================
cd /shard_data/brooksli/workspace/0625/CUDA_MATMUAL || exit 1
SZ=${1:-2048}
NCU="sudo -n /usr/local/cuda/bin/ncu"   # 采计数器要 root
IMP="/usr/local/cuda/bin/ncu"           # 导文本不需要 root
OUT=profiling/cuda_core
mkdir -p "$OUT"

prof() {  # prof <name> <kernel_regex> <exe> <id>
  local name=$1 rgx=$2 exe=$3 id=$4
  echo "==== ncu $name (kernel ~/$rgx/, id $id) @ ${SZ}^3 ===="
  $NCU --set full -k "regex:$rgx" -s 1 -c 1 -f \
      -o "$OUT/$name" "$exe" "$id" "$SZ" "$SZ" "$SZ" > "$OUT/$name.run.log" 2>&1
  echo "  rc=$?"
  $IMP -i "$OUT/$name.ncu-rep" --page details > "$OUT/$name.details.txt" 2>&1 || true
}

# ===================== FP32 (kernels/cuda_core/bench) ========================
B=kernels/cuda_core/bench
echo "######## CUDA core · FP32 ########"
prof cc_00_cublas                 xmma_gemm            $B 0
prof cc_01_naive                  naive_kernel         $B 1
prof cc_02_smem                   smem_kernel          $B 2
prof cc_03_blocktiling            blocktiling_kernel   $B 3
prof cc_04_2Dblocktiling          Dblocktiling_kernel  $B 4
prof cc_05_vectorized             vectorized_kernel    $B 5
prof cc_06a_autotune_64x64x8_8x4  autotuning_kernel    $B 6
prof cc_06b_autotune_64x64x16_8x4 autotuning_kernel    $B 7
prof cc_06c_autotune_64x64x8_8x8  autotuning_kernel    $B 8
prof cc_07_warptile               warptile_kernel      $B 9
prof cc_08_warptile_vec           warptile_vec_kernel  $B 10
prof cc_09_bankconflict           bank_conflict_kernel $B 11
prof cc_10_doublebuffer           double_buffer_kernel $B 12

# ===================== BF16 (kernels/cuda_core/bench_bf16) ===================
# 仅 9 个手写 kernel。如需 bf16 的 cublas/autotune 基线，追加：
#   prof cc_bf16_00_cublas   nvjet                 kernels/cuda_core/bench_bf16 0
#   prof cc_bf16_06_autotune autotuning_kernel_bf16 kernels/cuda_core/bench_bf16 6
BB=kernels/cuda_core/bench_bf16
echo "######## CUDA core · BF16 ########"
prof cc_bf16_01_naive         naive_bf16_kernel         $BB 1
prof cc_bf16_02_smem          smem_bf16_kernel          $BB 2
prof cc_bf16_03_blocktiling   blocktiling_bf16_kernel   $BB 3
prof cc_bf16_04_2Dblocktiling Dblocktiling_bf16_kernel  $BB 4
prof cc_bf16_05_vectorized    vectorized_bf16_kernel    $BB 5
prof cc_bf16_07_warptile      warptile_bf16_kernel      $BB 9
prof cc_bf16_08_warptile_vec  warptile_vec_bf16_kernel  $BB 10
prof cc_bf16_09_bankconflict  bank_conflict_bf16_kernel $BB 11
prof cc_bf16_10_doublebuffer  double_buffer_bf16_kernel $BB 12

sudo -n chown $(id -u):$(id -g) $OUT/cc_*.ncu-rep 2>/dev/null || true
echo "ALL_NCU_DONE"
