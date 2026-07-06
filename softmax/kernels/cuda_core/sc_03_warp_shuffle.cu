// sc_03 warp-shuffle — one warp per row, reductions via __shfl_xor (no shared mem,
// no __syncthreads). Why this rung: replace sc_02's shared-memory tree + barriers
// with register-only warp shuffles. Removes the smem round-trips and syncs; when a
// row fits a warp's striding this is the leanest reduction. Still 3 DRAM reads of x.
#include "softmax.h"

static constexpr int kWarpsPerBlock = 8;   // 8 warps -> 256 threads/block

__device__ __forceinline__ float warpMax(float v) {
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, o));
  return v;
}
__device__ __forceinline__ float warpSum(float v) {
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
  return v;
}

__global__ void sc03_warp_shuffle_kernel(const float* __restrict__ x,
                                         float* __restrict__ y, int M, int N) {
  int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  int lane = threadIdx.x & 31;
  if (warp >= M) return;
  const float* xr = x + (size_t)warp * N;
  float* yr = y + (size_t)warp * N;

  float m = -INFINITY;
  for (int j = lane; j < N; j += 32) m = fmaxf(m, xr[j]);
  m = warpMax(m);

  float s = 0.f;
  for (int j = lane; j < N; j += 32) s += __expf(xr[j] - m);
  s = warpSum(s);
  float inv = 1.f / s;

  for (int j = lane; j < N; j += 32) yr[j] = __expf(xr[j] - m) * inv;
}

void softmax_warp_shuffle(const float* x, float* y, int M, int N, cudaStream_t st) {
  int block = kWarpsPerBlock * 32;
  int grid = (M + kWarpsPerBlock - 1) / kWarpsPerBlock;
  sc03_warp_shuffle_kernel<<<grid, block, 0, st>>>(x, y, M, N);
}
