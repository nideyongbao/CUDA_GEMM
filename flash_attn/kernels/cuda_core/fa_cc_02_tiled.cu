// fa_cc_02 tiled — the FA2 blocked form on CUDA cores (FA 细节.md "FA2 完整分块算法").
// Outer parallelism = Q tiles (one block per [b,h,Q-tile]); inner loop streams K/V
// tiles through shared memory. Each thread owns one query row and keeps its (m,l,acc)
// resident for the whole KV sweep (FA2's "Q outer, K/V inner" -> O written once).
// Same online-softmax recurrence as fa_cc_01, now with coalesced smem-staged K/V
// reuse. Still fp32 FMA (no tensor core) -> the algorithmic end-state the tensor_core
// path reproduces on mma.sync.
#include "fa_cc.h"

static constexpr int kBr = 32, kBc = 32, kMaxD = 128;

__global__ void fa_cc02_kernel(const float* __restrict__ Q, const float* __restrict__ K,
                               const float* __restrict__ V, float* __restrict__ O,
                               int B, int S, int H, int D, bool causal) {
  int qtile = blockIdx.x, h = blockIdx.y, b = blockIdx.z;
  int t = threadIdx.x;              // query row within tile
  int i = qtile * kBr + t;
  bool active = i < S;
  float scale = rsqrtf((float)D);

  __shared__ float sK[kBc * kMaxD];
  __shared__ float sV[kBc * kMaxD];

  float qreg[kMaxD], acc[kMaxD];
  for (int d = 0; d < D; ++d) { qreg[d] = active ? Q[idx4(b, i, h, d, S, H, D)] : 0.f; acc[d] = 0.f; }
  float m = -INFINITY, l = 0.f;

  int i0 = qtile * kBr;
  int loopEnd = causal ? min(i0 + kBr - 1, S - 1) : S - 1;   // last key any row here needs

  for (int j0 = 0; j0 <= loopEnd; j0 += kBc) {
    // cooperatively stage K/V tile [kBc, D] into shared memory (coalesced)
    for (int idx = t; idx < kBc * D; idx += blockDim.x) {
      int jj = idx / D, dd = idx % D, j = j0 + jj;
      float kk = (j < S) ? K[idx4(b, j, h, dd, S, H, D)] : 0.f;
      float vv = (j < S) ? V[idx4(b, j, h, dd, S, H, D)] : 0.f;
      sK[idx] = kk; sV[idx] = vv;
    }
    __syncthreads();

    if (active) {
      int jmax = causal ? i : S - 1;
      for (int jj = 0; jj < kBc; ++jj) {
        int j = j0 + jj;
        if (j > jmax) break;
        float s = 0.f;
        for (int d = 0; d < D; ++d) s += qreg[d] * sK[jj * D + d];
        s *= scale;
        float mnew = fmaxf(m, s);
        float c = __expf(m - mnew), p = __expf(s - mnew);
        l = l * c + p;
        for (int d = 0; d < D; ++d) acc[d] = acc[d] * c + p * sV[jj * D + d];
        m = mnew;
      }
    }
    __syncthreads();
  }

  if (active) {
    float inv = 1.f / l;
    for (int d = 0; d < D; ++d) O[idx4(b, i, h, d, S, H, D)] = acc[d] * inv;
  }
}

void fa_cc_tiled(const float* Q, const float* K, const float* V, float* O,
                 int B, int S, int H, int D, bool causal, cudaStream_t st) {
  dim3 grid((S + kBr - 1) / kBr, H, B);
  fa_cc02_kernel<<<grid, kBr, 0, st>>>(Q, K, V, O, B, S, H, D, causal);
}
