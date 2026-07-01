#!/bin/bash
# A800 (sm_80) ncu 剖析：与 H20 的 profiling/{cuda_core,tensor_core}/SUMMARY.md 同口径(2048^3, --set full,
# -s 1 -c 1 取第2次 warmup launch)，用于 A800 vs H20 的逐核对比。采计数器需 root(本机 sudo 免密)。
# 前置：先构建到 build/ —— make ARCH=-arch=sm_80 BUILD=build && make tc ARCH=-arch=sm_80 TC_HOPPER=0 BUILD=build
# （生成本参考基线用；日常 ncu 建议直接 `bash run_all.sh --ncu`，输出进 result/<时间戳>/profiling/）
cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 1
SZ=${1:-2048}
NCU="sudo -n /usr/local/cuda-12.8/bin/ncu"
IMP="/usr/local/cuda-12.8/bin/ncu"
OUT=profiling/a800
prof(){ local name=$1 rgx=$2 exe=$3 id=$4
  echo "==== ncu $name (~/$rgx/, id $id) @ ${SZ}^3 ===="
  $NCU --set full -k "regex:$rgx" -s 1 -c 1 -f -o "$OUT/$name" "$exe" "$id" "$SZ" "$SZ" "$SZ" > "$OUT/$name.run.log" 2>&1
  echo "  rc=$?"
  $IMP -i "$OUT/$name.ncu-rep" --page details > "$OUT/$name.details.txt" 2>&1 || true
}
B=build/cuda_core/bench
BB=build/cuda_core/bench_bf16
T=build/tensor_core/bench
echo "######## A800 CUDA core FP32 ########"
prof a800_cc_00_cublas        gemm                 $B 0
prof a800_cc_01_naive         naive_kernel         $B 1
prof a800_cc_02_smem          smem_kernel          $B 2
prof a800_cc_04_2Dblocktiling Dblocktiling_kernel  $B 4
prof a800_cc_05_vectorized    vectorized_kernel    $B 5
prof a800_cc_10_doublebuffer  double_buffer_kernel $B 12
echo "######## A800 CUDA core BF16 ########"
prof a800_ccbf16_10_doublebuffer double_buffer_bf16_kernel $BB 12
echo "######## A800 Tensor core WMMA ########"
prof a800_tc_01_wmma_naive    wmma_naive_kernel    $T 1
prof a800_tc_02_wmma_smem     wmma_smem_kernel     $T 2
prof a800_tc_03_wmma_pipe     wmma_pipe_kernel     $T 3
sudo -n chown $(id -u):$(id -g) $OUT/*.ncu-rep 2>/dev/null || true
echo "ALL_NCU_DONE"
