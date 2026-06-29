#!/bin/bash
# 给 tensor_core 5 个用例做 ncu --set full 剖析。统一驱动 ./kernels/tensor_core/bench <id>
# 是纯计时（无对拍），kernel 从 warmup 开始 launch；-s 1 -c 1 跳过第 1 次、剖析第 2 次
# （一次有代表性的 warmup launch）。需要 GPU 性能计数器权限 -> 用 sudo 全路径 ncu。
# 产物落在 profiling/tensor_core/（与 cuda_core 那半对称）。
cd /shard_data/brooksli/workspace/0625/CUDA_MATMUAL
SZ=2048
BENCH=./kernels/tensor_core/bench
OUT=profiling/tensor_core
NCU="sudo -n /usr/local/cuda/bin/ncu"
IMP="/usr/local/cuda/bin/ncu"
mkdir -p "$OUT"
rm -f "$OUT"/_probe.ncu-rep
declare -A K=(
  [tc_01_wmma_naive]=wmma_naive_kernel
  [tc_02_wmma_smem]=wmma_smem_kernel
  [tc_03_wmma_pipe]=wmma_pipe_kernel
  [tc_04_wgmma_tma_ws]=wgmma_kernel
  [tc_05_wgmma_fp8]=wgmma_fp8_kernel
)
declare -A ID=(
  [tc_01_wmma_naive]=1
  [tc_02_wmma_smem]=2
  [tc_03_wmma_pipe]=3
  [tc_04_wgmma_tma_ws]=4
  [tc_05_wgmma_fp8]=5
)
for exe in tc_01_wmma_naive tc_02_wmma_smem tc_03_wmma_pipe tc_04_wgmma_tma_ws tc_05_wgmma_fp8; do
  echo "==== ncu $exe (kernel ${K[$exe]}, id ${ID[$exe]}) @ ${SZ}^3 ===="
  $NCU --set full -k "regex:${K[$exe]}" -s 1 -c 1 -f \
      -o "$OUT/$exe" $BENCH ${ID[$exe]} $SZ $SZ $SZ > "$OUT/${exe}.run.log" 2>&1
  echo "  rc=$?"
  $IMP -i "$OUT/$exe.ncu-rep" --page details > "$OUT/$exe.details.txt" 2>&1 || true
done
sudo -n chown $(id -u):$(id -g) "$OUT"/*.ncu-rep 2>/dev/null || true
echo "ALL_NCU_DONE"
