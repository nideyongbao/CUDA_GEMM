// flash_attn operator — shared harness: FLOP model (matches TinyFA benchmark.py),
// CPU reference attention (fp32, MHA/GQA), dtype helpers, cudaEvent median timing.
// Tensor layout follows TinyFA:  Q/K/V/O = [batch, seq, heads, headDim] row-major.
#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>

#define CHECK_CUDA(call)                                                        \
  do { cudaError_t _e=(call); if(_e!=cudaSuccess){                             \
    fprintf(stderr,"CUDA %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(_e));\
    exit(1);} } while(0)

// FLOPs (identical to TinyFA benchmarks/benchmark.py flops_fwd)
inline double fa_flops(int B, int H, int S, int D, bool causal) {
  double f = 4.0 * B * H * (double)S * S * D;
  return causal ? f * 0.5 : f;
}

// index into [B,S,H,D] row-major
__host__ __device__ inline size_t idx4(int b, int s, int h, int d, int S, int H, int D) {
  return (((size_t)b * S + s) * H + h) * D + d;
}

// CPU reference MHA/GQA attention in fp32. scale = 1/sqrt(D). causal: key j<=i.
// numHeadsKV may be < numHeadsQ (GQA): query head h maps to kv head h/(Hq/Hkv).
inline void fa_cpu_ref(const float* Q, const float* K, const float* V, float* O,
                       int B, int S, int Hq, int Hkv, int D, bool causal) {
  float scale = 1.0f / sqrtf((float)D);
  int group = Hq / Hkv;
  std::vector<double> s(S);
  for (int b = 0; b < B; ++b)
    for (int hq = 0; hq < Hq; ++hq) {
      int hk = hq / group;
      for (int i = 0; i < S; ++i) {
        int jmax = causal ? i : S - 1;
        double m = -INFINITY;
        for (int j = 0; j <= jmax; ++j) {
          double dot = 0;
          for (int d = 0; d < D; ++d)
            dot += (double)Q[idx4(b,i,hq,d,S,Hq,D)] * (double)K[idx4(b,j,hk,d,S,Hkv,D)];
          s[j] = dot * scale;
          if (s[j] > m) m = s[j];
        }
        double sum = 0;
        for (int j = 0; j <= jmax; ++j) { s[j] = exp(s[j] - m); sum += s[j]; }
        double inv = 1.0 / sum;
        for (int d = 0; d < D; ++d) {
          double acc = 0;
          for (int j = 0; j <= jmax; ++j) acc += s[j] * (double)V[idx4(b,j,hk,d,S,Hkv,D)];
          O[idx4(b,i,hq,d,S,Hq,D)] = (float)(acc * inv);
        }
      }
    }
}

// ---- dtype tag ----
enum FADtype { FA_FP16, FA_BF16 };
inline FADtype fa_parse_dtype(const char* s) {
  return (s && (s[0]=='b'||s[0]=='B')) ? FA_BF16 : FA_FP16;
}
inline const char* fa_dtype_name(FADtype d){ return d==FA_BF16 ? "bf16" : "fp16"; }
