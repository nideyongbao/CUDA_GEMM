// fa_tc ladder verify — each rung vs fp32 CPU reference attention (D=128, fp16).
//   usage: fa_tc_verify <id> [B H S causal]
#include "fa_common.h"
#include "fa_tc_ladder.cuh"

template <typename T> __host__ T f2t(float x);
template <> __host__ __half f2t<__half>(float x){ return __float2half(x); }
template <typename T> __host__ float t2f(T x);
template <> __host__ float t2f<__half>(__half x){ return __half2float(x); }

int main(int argc, char** argv) {
  if (argc < 2) { fatc::fa_tc_print(); return 0; }
  using T = __half;
  int id = atoi(argv[1]);
  int B = argc > 2 ? atoi(argv[2]) : 2, H = argc > 3 ? atoi(argv[3]) : 8;
  int S = argc > 4 ? atoi(argv[4]) : 512;
  bool causal = argc > 5 ? atoi(argv[5]) != 0 : false;
  const int D = 128;
  if (!fatc::fa_tc_name(id)) { printf("bad id\n"); fatc::fa_tc_print(); return 1; }
  size_t n = (size_t)B * S * H * D;
  std::vector<float> hf(n), ref(n);
  for (size_t i = 0; i < n; ++i) hf[i] = (float)((i * 40503u + 7u) % 1000) / 500.f - 1.f;
  fa_cpu_ref(hf.data(), hf.data(), hf.data(), ref.data(), B, S, H, H, D, causal);
  std::vector<T> ht(n); for (size_t i = 0; i < n; ++i) ht[i] = f2t<T>(hf[i]);
  T *dQ,*dK,*dV,*dO;
  CHECK_CUDA(cudaMalloc(&dQ,n*2)); CHECK_CUDA(cudaMalloc(&dK,n*2));
  CHECK_CUDA(cudaMalloc(&dV,n*2)); CHECK_CUDA(cudaMalloc(&dO,n*2));
  CHECK_CUDA(cudaMemcpy(dQ,ht.data(),n*2,cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dK,ht.data(),n*2,cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dV,ht.data(),n*2,cudaMemcpyHostToDevice));
  fatc::launch_id<T>(id, dQ, dK, dV, dO, B, S, H, causal, 0);
  CHECK_CUDA(cudaGetLastError()); CHECK_CUDA(cudaDeviceSynchronize());
  std::vector<T> ho(n); CHECK_CUDA(cudaMemcpy(ho.data(),dO,n*2,cudaMemcpyDeviceToHost));
  double maxa=0,maxr=0; size_t bad=0;
  for (size_t i=0;i<n;++i){ double a=t2f<T>(ho[i]),bb=ref[i],d=fabs(a-bb);
    maxa=d>maxa?d:maxa; double rl=d/(fabs(bb)+1e-30); maxr=rl>maxr?rl:maxr;
    if(d>2e-2+2e-2*fabs(bb))++bad; }
  printf("%-18s B=%d H=%d S=%d D=%d causal=%d  max_abs=%.3e max_rel=%.3e bad=%zu/%zu  %s\n",
         fatc::fa_tc_name(id),B,H,S,D,causal,maxa,maxr,bad,n,bad==0?"PASS":"FAIL");
  cudaFree(dQ);cudaFree(dK);cudaFree(dV);cudaFree(dO);
  return bad==0?0:1;
}
