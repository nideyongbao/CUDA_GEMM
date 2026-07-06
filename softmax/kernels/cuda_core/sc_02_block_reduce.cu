// sc_02 block-reduce — one block per row, threads cooperate via shared memory.
// Why this rung: fix sc_01's uncoalesced access. A whole block strides across the
// row, so consecutive threads read consecutive elements -> coalesced loads. Row
// max and exp-sum are computed with a shared-memory tree reduction.
// Bottleneck now: x is read from DRAM three times (max, sum, write) -> memory
// traffic bound; that's what sc_03/04/05 chip away at.
#include "softmax.h"

static constexpr int kBlock = 256;

__global__ void sc02_block_reduce_kernel(const float* __restrict__ x,
                                         float* __restrict__ y, int N) {
  int r = blockIdx.x;
  const float* xr = x + (size_t)r * N;
  float* yr = y + (size_t)r * N;
  int t = threadIdx.x, nt = blockDim.x;
  __shared__ float red[kBlock];

  // pass 1: row max
  float m = -INFINITY;
  for (int j = t; j < N; j += nt) m = fmaxf(m, xr[j]);
  red[t] = m; __syncthreads();
  for (int s = nt >> 1; s > 0; s >>= 1) {
    if (t < s) red[t] = fmaxf(red[t], red[t + s]);
    __syncthreads();
  }
  m = red[0]; __syncthreads();

  // pass 2: exp-sum
  float sum = 0.f;
  for (int j = t; j < N; j += nt) sum += __expf(xr[j] - m);
  red[t] = sum; __syncthreads();
  for (int s = nt >> 1; s > 0; s >>= 1) {
    if (t < s) red[t] += red[t + s];
    __syncthreads();
  }
  float inv = 1.f / red[0]; __syncthreads();

  // pass 3: normalize + write
  for (int j = t; j < N; j += nt) yr[j] = __expf(xr[j] - m) * inv;
}

void softmax_block_reduce(const float* x, float* y, int M, int N, cudaStream_t st) {
  sc02_block_reduce_kernel<<<M, kBlock, 0, st>>>(x, y, N);
}
