// FA tensor-core bench — drives TinyFA's tensor-core forward (tfa::flashAttn) and
// reports TFLOPS with the same FLOP model + median timing as TinyFA's benchmark.py,
// so numbers are directly comparable to the TinyFA repo.
//   usage: bench <fp16|bf16> [B H S D causal]   default: fp16 2 32 4096 128 0
#include "fa_common.h"
#include "flash_api.cuh"   // from vendor/tfa (-I)

template<typename T> __host__ T f2t(float x);
template<> __host__ __half        f2t<__half>(float x){ return __float2half(x); }
template<> __host__ __nv_bfloat16 f2t<__nv_bfloat16>(float x){ return __float2bfloat16(x); }

template<typename T>
double run_bench(int B,int H,int S,int D,bool causal){
  size_t n=(size_t)B*S*H*D;
  std::vector<float> hf(n);
  for(size_t i=0;i<n;++i) hf[i]=(float)((i*2654435761u+11u)%1000)/500.f-1.f;
  std::vector<T> ht(n); for(size_t i=0;i<n;++i) ht[i]=f2t<T>(hf[i]);
  T *dQ,*dK,*dV,*dO;
  CHECK_CUDA(cudaMalloc(&dQ,n*sizeof(T))); CHECK_CUDA(cudaMalloc(&dK,n*sizeof(T)));
  CHECK_CUDA(cudaMalloc(&dV,n*sizeof(T))); CHECK_CUDA(cudaMalloc(&dO,n*sizeof(T)));
  CHECK_CUDA(cudaMemcpy(dQ,ht.data(),n*sizeof(T),cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dK,ht.data(),n*sizeof(T),cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dV,ht.data(),n*sizeof(T),cudaMemcpyHostToDevice));
  auto call=[&](){ tfa::flashAttn<T>(dQ,dK,dV,dO,B,S,S,H,H,D,causal,0); };
  for(int i=0;i<20;++i) call();                 // warmup
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());
  std::vector<float> times;
  for(int r=0;r<40;++r){
    cudaEvent_t s,e; cudaEventCreate(&s); cudaEventCreate(&e);
    cudaEventRecord(s); call(); cudaEventRecord(e); cudaEventSynchronize(e);
    float ms; cudaEventElapsedTime(&ms,s,e); times.push_back(ms);
    cudaEventDestroy(s); cudaEventDestroy(e);
  }
  std::sort(times.begin(),times.end());
  double ms=times[times.size()/2];             // median
  cudaFree(dQ);cudaFree(dK);cudaFree(dV);cudaFree(dO);
  return ms;
}

int main(int argc,char**argv){
  FADtype dt = fa_parse_dtype(argc>1?argv[1]:"fp16");
  int B=argc>2?atoi(argv[2]):2, H=argc>3?atoi(argv[3]):32;
  int S=argc>4?atoi(argv[4]):4096, D=argc>5?atoi(argv[5]):128;
  bool causal=argc>6?atoi(argv[6])!=0:false;
  double ms = (dt==FA_BF16)? run_bench<__nv_bfloat16>(B,H,S,D,causal)
                           : run_bench<__half>(B,H,S,D,causal);
  double tflops = fa_flops(B,H,S,D,causal)/(ms*1e-3)/1e12;
  printf("tfa_mma %-4s B=%d H=%d S=%d D=%d causal=%d  time=%.4f ms  %.2f TFLOPS\n",
         fa_dtype_name(dt),B,H,S,D,causal,ms,tflops);
  return 0;
}
