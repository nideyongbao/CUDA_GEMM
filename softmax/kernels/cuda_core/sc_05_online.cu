// sc_05 online — THE FlashAttention bridge. One block per row, float4 loads.
// Why this rung: sc_02/03/04 read x THREE times (max pass, sum pass, write pass).
// Online softmax fuses the max pass and the sum pass into ONE streaming pass by
// carrying a running (max m, sum l) and rescaling l when a bigger max appears:
//     m_new = max(m, x);   l = l * exp(m - m_new) + exp(x - m_new);   m = m_new
// (this is exactly the Online-Safe-Softmax recurrence in FA 细节.md, Step 3/4).
// Two DRAM reads of x (fused stats + write) instead of three -> ~25% less traffic.
// The SAME (m,l)-merge is what FlashAttention runs on the score fragment; here it
// is isolated on a plain row so the trick is learned before fusing it into attention.
#include "softmax.h"

static constexpr int kBlock = 256;

// merge two online partials (m1,l1) and (m2,l2) into one
__device__ __forceinline__ void mergeMl(float& m, float& l, float m2, float l2) {
  float mn = fmaxf(m, m2);
  l = l * __expf(m - mn) + l2 * __expf(m2 - mn);
  m = mn;
}

__global__ void sc05_online_kernel(const float* __restrict__ x,
                                   float* __restrict__ y, int N) {
  int r = blockIdx.x;
  const float4* xr = reinterpret_cast<const float4*>(x + (size_t)r * N);
  float4* yr = reinterpret_cast<float4*>(y + (size_t)r * N);
  int N4 = N >> 2;
  int t = threadIdx.x, nt = blockDim.x;

  // ---- pass 1 (fused): single streaming scan computes BOTH max and sum ----
  float m = -INFINITY, l = 0.f;
  for (int j = t; j < N4; j += nt) {
    float4 v = xr[j];
    // fold the 4 lane values into this thread's running (m,l)
    mergeMl(m, l, v.x, 1.f);
    mergeMl(m, l, v.y, 1.f);
    mergeMl(m, l, v.z, 1.f);
    mergeMl(m, l, v.w, 1.f);
  }
  // block-combine the per-thread (m,l) partials with the same online merge
  __shared__ float sm[kBlock], sl[kBlock];
  sm[t] = m; sl[t] = l; __syncthreads();
  for (int s = nt >> 1; s > 0; s >>= 1) {
    if (t < s) mergeMl(sm[t], sl[t], sm[t + s], sl[t + s]);
    __syncthreads();
  }
  float M = sm[0], inv = 1.f / sl[0]; __syncthreads();

  // ---- pass 2: normalize + write ----
  for (int j = t; j < N4; j += nt) {
    float4 v = xr[j], o;
    o.x = __expf(v.x - M) * inv; o.y = __expf(v.y - M) * inv;
    o.z = __expf(v.z - M) * inv; o.w = __expf(v.w - M) * inv;
    yr[j] = o;
  }
}

void softmax_online(const float* x, float* y, int M, int N, cudaStream_t st) {
  if (N % 4 != 0) { softmax_block_reduce(x, y, M, N, st); return; }  // fallback
  sc05_online_kernel<<<M, kBlock, 0, st>>>(x, y, N);
}
