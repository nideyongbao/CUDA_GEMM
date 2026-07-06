// sc_04 vectorized — block per row + 128-bit (float4) loads/stores. Why this rung:
// softmax is DRAM-bandwidth bound, so widen each memory transaction to 16 B. Four
// elements per instruction cuts issue overhead and maximizes achieved bandwidth per
// request. Requires N % 4 == 0. Reduction stays a shared-memory tree (as sc_02).
#include "softmax.h"

static constexpr int kBlock = 256;

__global__ void sc04_vectorized_kernel(const float* __restrict__ x,
                                       float* __restrict__ y, int N) {
  int r = blockIdx.x;
  const float4* xr = reinterpret_cast<const float4*>(x + (size_t)r * N);
  float4* yr = reinterpret_cast<float4*>(y + (size_t)r * N);
  int N4 = N >> 2;
  int t = threadIdx.x, nt = blockDim.x;
  __shared__ float red[kBlock];

  // pass 1: row max over float4
  float m = -INFINITY;
  for (int j = t; j < N4; j += nt) {
    float4 v = xr[j];
    m = fmaxf(m, fmaxf(fmaxf(v.x, v.y), fmaxf(v.z, v.w)));
  }
  red[t] = m; __syncthreads();
  for (int s = nt >> 1; s > 0; s >>= 1) { if (t < s) red[t] = fmaxf(red[t], red[t + s]); __syncthreads(); }
  m = red[0]; __syncthreads();

  // pass 2: exp-sum over float4
  float sum = 0.f;
  for (int j = t; j < N4; j += nt) {
    float4 v = xr[j];
    sum += __expf(v.x - m) + __expf(v.y - m) + __expf(v.z - m) + __expf(v.w - m);
  }
  red[t] = sum; __syncthreads();
  for (int s = nt >> 1; s > 0; s >>= 1) { if (t < s) red[t] += red[t + s]; __syncthreads(); }
  float inv = 1.f / red[0]; __syncthreads();

  // pass 3: normalize + write float4
  for (int j = t; j < N4; j += nt) {
    float4 v = xr[j], o;
    o.x = __expf(v.x - m) * inv; o.y = __expf(v.y - m) * inv;
    o.z = __expf(v.z - m) * inv; o.w = __expf(v.w - m) * inv;
    yr[j] = o;
  }
}

void softmax_vectorized(const float* x, float* y, int M, int N, cudaStream_t st) {
  if (N % 4 != 0) { softmax_block_reduce(x, y, M, N, st); return; }  // fallback
  sc04_vectorized_kernel<<<M, kBlock, 0, st>>>(x, y, N);
}
