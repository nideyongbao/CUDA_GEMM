// softmax operator — shared declarations, CPU reference, harness helpers.
// Row-softmax over the last dim: X[M, N] (row-major, fp32) -> Y[M, N],
// y[i,:] = softmax(x[i,:]) with the numerically-stable "safe" formulation.
//
// This mirrors the CUDA_GEMM harness: each rung is a separate kernel file
// exposing one launch function; bench.cu / verify.cu dispatch by integer id.
#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>

#define CHECK_CUDA(call)                                                       \
  do {                                                                         \
    cudaError_t _e = (call);                                                   \
    if (_e != cudaSuccess) {                                                   \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,            \
              cudaGetErrorString(_e));                                         \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

// ---- kernel registry: id -> (name, launcher) ----
// Each rung implements this signature (one block-set per launch, fp32).
using SoftmaxFn = void (*)(const float* x, float* y, int M, int N, cudaStream_t);

void softmax_naive       (const float* x, float* y, int M, int N, cudaStream_t); // sc_01
void softmax_block_reduce(const float* x, float* y, int M, int N, cudaStream_t); // sc_02
void softmax_warp_shuffle(const float* x, float* y, int M, int N, cudaStream_t); // sc_03
void softmax_vectorized  (const float* x, float* y, int M, int N, cudaStream_t); // sc_04
void softmax_online      (const float* x, float* y, int M, int N, cudaStream_t); // sc_05

struct SoftmaxCase { int id; const char* name; SoftmaxFn fn; };

inline const std::vector<SoftmaxCase>& softmax_registry() {
  static const std::vector<SoftmaxCase> reg = {
      {1, "sc_01_naive",        softmax_naive},
      {2, "sc_02_block_reduce", softmax_block_reduce},
      {3, "sc_03_warp_shuffle", softmax_warp_shuffle},
      {4, "sc_04_vectorized",   softmax_vectorized},
      {5, "sc_05_online",       softmax_online},
  };
  return reg;
}

inline const SoftmaxCase* softmax_find(int id) {
  for (const auto& c : softmax_registry())
    if (c.id == id) return &c;
  return nullptr;
}

inline void softmax_print_registry() {
  printf("softmax kernel ids:\n");
  for (const auto& c : softmax_registry()) printf("  %2d  %s\n", c.id, c.name);
}

// ---- CPU reference (double-precision safe softmax) ----
inline void softmax_cpu_ref(const float* x, float* y, int M, int N) {
  for (int i = 0; i < M; ++i) {
    const float* xr = x + (size_t)i * N;
    float* yr = y + (size_t)i * N;
    double m = -INFINITY;
    for (int j = 0; j < N; ++j) m = xr[j] > m ? xr[j] : m;
    double s = 0.0;
    for (int j = 0; j < N; ++j) s += exp((double)xr[j] - m);
    double inv = 1.0 / s;
    for (int j = 0; j < N; ++j) yr[j] = (float)(exp((double)xr[j] - m) * inv);
  }
}
