// fa_cc_01 stream — one thread per query row, single ONLINE pass over K/V.
// This is FA 细节.md "Step 4: FlashAttention single-query iteration": carry a
// running (m,l) and an unnormalized accumulator acc[D]; rescale on a new max;
// divide once at the end (delayed normalization). No S=QK^T matrix is ever
// materialized -> O(S*D) memory instead of O(S^2). CUDA-core FMA only, no tensor
// core, fully uncoalesced (a thread walks a whole row) -> the correctness/algorithm
// baseline the tiled + tensor-core rungs build on.
#include "fa_cc.h"

static constexpr int kMaxD = 128;

__global__ void fa_cc01_kernel(const float* __restrict__ Q, const float* __restrict__ K,
                               const float* __restrict__ V, float* __restrict__ O,
                               int B, int S, int H, int D, bool causal) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;   // one thread per (b,h,i)
  if (gid >= B * H * S) return;
  int i = gid % S; int t = gid / S; int h = t % H; int b = t / H;
  float scale = rsqrtf((float)D);

  const float* q = Q + idx4(b, i, h, 0, S, H, D);
  float acc[kMaxD];
  #pragma unroll 1
  for (int d = 0; d < D; ++d) acc[d] = 0.f;

  float m = -INFINITY, l = 0.f;
  int jmax = causal ? i : S - 1;
  for (int j = 0; j <= jmax; ++j) {
    const float* k = K + idx4(b, j, h, 0, S, H, D);
    float s = 0.f;
    for (int d = 0; d < D; ++d) s += q[d] * k[d];
    s *= scale;
    float mnew = fmaxf(m, s);
    float c = __expf(m - mnew);      // rescale factor for history
    float p = __expf(s - mnew);      // this key's weight
    l = l * c + p;
    const float* v = V + idx4(b, j, h, 0, S, H, D);
    for (int d = 0; d < D; ++d) acc[d] = acc[d] * c + p * v[d];
    m = mnew;
  }
  float inv = 1.f / l;
  float* o = O + idx4(b, i, h, 0, S, H, D);
  for (int d = 0; d < D; ++d) o[d] = acc[d] * inv;
}

void fa_cc_stream(const float* Q, const float* K, const float* V, float* O,
                  int B, int S, int H, int D, bool causal, cudaStream_t st) {
  int total = B * H * S, block = 128;
  fa_cc01_kernel<<<(total + block - 1) / block, block, 0, st>>>(Q, K, V, O, B, S, H, D, causal);
}
