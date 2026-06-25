#!/bin/bash
# 给 tensor_core 5 个用例做 ncu --set full 剖析（profile bench-size 的一次 launch）。
# 需要 GPU 性能计数器权限 -> 用 sudo 全路径 ncu。-s 1 -c 1：跳过第 1 次(verify)，
# 剖析第 2 次(warmup，bench 尺寸)。
cd /shard_data/brooksli/workspace/0625/CUDA_MATMUAL
SZ=2048
NCU="sudo -n /usr/local/cuda/bin/ncu"
IMP="/usr/local/cuda/bin/ncu"
rm -f profiling/_probe.ncu-rep
declare -A K=(
  [tc_01_wmma_naive]=wmma_naive_kernel
  [tc_02_wmma_smem]=wmma_smem_kernel
  [tc_03_wmma_pipe]=wmma_pipe_kernel
  [tc_04_wgmma_tma_ws]=wgmma_kernel
  [tc_05_wgmma_fp8]=wgmma_fp8_kernel
)
for exe in tc_01_wmma_naive tc_02_wmma_smem tc_03_wmma_pipe tc_04_wgmma_tma_ws tc_05_wgmma_fp8; do
  echo "==== ncu $exe (kernel ${K[$exe]}) @ ${SZ}^3 ===="
  $NCU --set full -k "regex:${K[$exe]}" -s 1 -c 1 -f \
      -o profiling/$exe ./$exe $SZ $SZ $SZ > profiling/${exe}.run.log 2>&1
  echo "  rc=$?"
  $IMP -i profiling/$exe.ncu-rep --page details > profiling/$exe.details.txt 2>&1 || true
done
sudo -n chown $(id -u):$(id -g) profiling/*.ncu-rep 2>/dev/null || true
echo "ALL_NCU_DONE"
