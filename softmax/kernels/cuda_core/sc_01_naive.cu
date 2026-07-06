// sc_01 naive — one thread per row, three sequential passes (max, sum, normalize).
// Why this rung: establishes correctness with the safe-softmax formulation and a
// baseline. Bottleneck: one thread walks a whole row -> adjacent threads touch
// rows N apart, so every global access is fully uncoalesced (stride-N). Pure
// latency-bound, near-zero DRAM efficiency. Everything after this fixes memory.
#include "softmax.h"

__global__ void sc01_naive_kernel(const float* __restrict__ x,
                                  float* __restrict__ y, int M, int N) {
  int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= M) return;
  const float* xr = x + (size_t)r * N;
  float* yr = y + (size_t)r * N;

  float m = -INFINITY;
  for (int j = 0; j < N; ++j) m = fmaxf(m, xr[j]);        // pass 1: row max
  float s = 0.f;
  for (int j = 0; j < N; ++j) s += __expf(xr[j] - m);     // pass 2: exp-sum
  float inv = 1.f / s;
  for (int j = 0; j < N; ++j) yr[j] = __expf(xr[j] - m) * inv;  // pass 3: write
}

void softmax_naive(const float* x, float* y, int M, int N, cudaStream_t st) {
  int block = 256;
  int grid = (M + block - 1) / block;
  sc01_naive_kernel<<<grid, block, 0, st>>>(x, y, M, N);
}
