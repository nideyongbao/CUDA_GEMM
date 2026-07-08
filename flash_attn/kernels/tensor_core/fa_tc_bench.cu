// fa_tc ladder bench — TFLOPS per rung (same FLOP model + median timing as the
// TinyFA bench, so the raw ladder is directly comparable to TinyFA / FA2 / SDPA).
//   usage: fa_tc_bench <id> [B H S causal]   (D=128, fp16)
#include "fa_common.h"
#include "fa_tc_ladder.cuh"
#include <algorithm>

template <typename T> __host__ T f2t(float x);
template <> __host__ __half f2t<__half>(float x){ return __float2half(x); }

int main(int argc, char** argv) {
  if (argc < 2) { fatc::fa_tc_print(); return 0; }
  using T = __half;
  int id = atoi(argv[1]);
  int B = argc > 2 ? atoi(argv[2]) : 2, H = argc > 3 ? atoi(argv[3]) : 32;
  int S = argc > 4 ? atoi(argv[4]) : 4096;
  bool causal = argc > 5 ? atoi(argv[5]) != 0 : false;
  const int D = 128;
  if (!fatc::fa_tc_name(id)) { printf("bad id\n"); fatc::fa_tc_print(); return 1; }
  size_t n = (size_t)B * S * H * D;
  std::vector<float> hf(n);
  for (size_t i=0;i<n;++i) hf[i]=(float)((i*2654435761u+11u)%1000)/500.f-1.f;
  std::vector<T> ht(n); for (size_t i=0;i<n;++i) ht[i]=f2t<T>(hf[i]);
  T *dQ,*dK,*dV,*dO;
  CHECK_CUDA(cudaMalloc(&dQ,n*2)); CHECK_CUDA(cudaMalloc(&dK,n*2));
  CHECK_CUDA(cudaMalloc(&dV,n*2)); CHECK_CUDA(cudaMalloc(&dO,n*2));
  CHECK_CUDA(cudaMemcpy(dQ,ht.data(),n*2,cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dK,ht.data(),n*2,cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dV,ht.data(),n*2,cudaMemcpyHostToDevice));
  auto call=[&](){ fatc::launch_id<T>(id,dQ,dK,dV,dO,B,S,H,causal,0); };
  for(int i=0;i<20;++i) call();
  CHECK_CUDA(cudaDeviceSynchronize()); CHECK_CUDA(cudaGetLastError());
  std::vector<float> ts;
  for(int r=0;r<40;++r){ cudaEvent_t s,e; cudaEventCreate(&s); cudaEventCreate(&e);
    cudaEventRecord(s); call(); cudaEventRecord(e); cudaEventSynchronize(e);
    float ms; cudaEventElapsedTime(&ms,s,e); ts.push_back(ms);
    cudaEventDestroy(s); cudaEventDestroy(e); }
  std::sort(ts.begin(),ts.end()); double ms=ts[ts.size()/2];
  double tflops=fa_flops(B,H,S,D,causal)/(ms*1e-3)/1e12;
  printf("%-18s B=%d H=%d S=%d D=%d causal=%d  time=%.4f ms  %.2f TFLOPS\n",
         fatc::fa_tc_name(id),B,H,S,D,causal,ms,tflops);
  cudaFree(dQ);cudaFree(dK);cudaFree(dV);cudaFree(dO);
  return 0;
}
