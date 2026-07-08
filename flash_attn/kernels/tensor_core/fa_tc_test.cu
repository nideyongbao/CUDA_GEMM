// Correctness driver for the from-scratch fa_tc kernel (base config), vs fp32 CPU ref.
//   usage: fa_tc_test [B H S causal]   (D fixed=128)
#include "fa_common.h"
#include "fa_tc.cuh"

template <typename T> __host__ T f2t(float x);
template <> __host__ __half f2t<__half>(float x){ return __float2half(x); }
template <> __host__ __nv_bfloat16 f2t<__nv_bfloat16>(float x){ return __float2bfloat16(x); }
template <typename T> __host__ float t2f(T x);
template <> __host__ float t2f<__half>(__half x){ return __half2float(x); }
template <> __host__ float t2f<__nv_bfloat16>(__nv_bfloat16 x){ return __bfloat162float(x); }

int main(int argc, char** argv) {
  using T = __half;
  int B = argc > 1 ? atoi(argv[1]) : 1;
  int H = argc > 2 ? atoi(argv[2]) : 1;
  int S = argc > 3 ? atoi(argv[3]) : 128;
  bool causal = argc > 4 ? atoi(argv[4]) != 0 : false;
  const int D = 128;
  size_t n = (size_t)B * S * H * D;
  std::vector<float> hf(n), ref(n);
  for (size_t i = 0; i < n; ++i) hf[i] = (float)((i * 40503u + 7u) % 1000) / 500.f - 1.f;
  fa_cpu_ref(hf.data(), hf.data(), hf.data(), ref.data(), B, S, H, H, D, causal);
  std::vector<T> ht(n); for (size_t i = 0; i < n; ++i) ht[i] = f2t<T>(hf[i]);
  T *dQ, *dK, *dV, *dO;
  CHECK_CUDA(cudaMalloc(&dQ, n * sizeof(T))); CHECK_CUDA(cudaMalloc(&dK, n * sizeof(T)));
  CHECK_CUDA(cudaMalloc(&dV, n * sizeof(T))); CHECK_CUDA(cudaMalloc(&dO, n * sizeof(T)));
  CHECK_CUDA(cudaMemcpy(dQ, ht.data(), n*sizeof(T), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dK, ht.data(), n*sizeof(T), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dV, ht.data(), n*sizeof(T), cudaMemcpyHostToDevice));

  
  size_t smem = (fatc::BR * fatc::DH + 2 * fatc::BC * fatc::DH) * sizeof(T);
  auto kern0 = fatc::fa_tc_kernel<T, false, false, false, false>;
  auto kern1 = fatc::fa_tc_kernel<T, false, false, false, true>;
  cudaFuncSetAttribute(kern0, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  cudaFuncSetAttribute(kern1, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  dim3 grid(S / fatc::BR, H, B), block(fatc::WARPS * 32);
  if (causal) kern1<<<grid, block, smem>>>(dQ, dK, dV, dO, B, S, H);
  else        kern0<<<grid, block, smem>>>(dQ, dK, dV, dO, B, S, H);
  CHECK_CUDA(cudaGetLastError()); CHECK_CUDA(cudaDeviceSynchronize());
  std::vector<T> ho(n); CHECK_CUDA(cudaMemcpy(ho.data(), dO, n*sizeof(T), cudaMemcpyDeviceToHost));

  double maxa = 0, maxr = 0; size_t bad = 0;
  for (size_t i = 0; i < n; ++i) {
    double a = t2f<T>(ho[i]), bb = ref[i], d = fabs(a - bb);
    maxa = d > maxa ? d : maxa; double rl = d / (fabs(bb) + 1e-30); maxr = rl > maxr ? rl : maxr;
    if (d > 2e-2 + 2e-2 * fabs(bb)) ++bad;
  }
  printf("fa_tc base B=%d H=%d S=%d D=%d causal=%d  max_abs=%.3e max_rel=%.3e bad=%zu/%zu  %s\n",
         B, H, S, D, causal, maxa, maxr, bad, n, bad == 0 ? "PASS" : "FAIL");
  // dump first row for debugging
  if (bad) { printf("  got[0..4]: "); for (int i=0;i<5;i++) printf("%.4f ", t2f<T>(ho[i]));
             printf("\n  ref[0..4]: "); for (int i=0;i<5;i++) printf("%.4f ", ref[i]); printf("\n"); }
  return bad == 0 ? 0 : 1;
}
