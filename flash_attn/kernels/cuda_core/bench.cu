// FA cuda_core bench — fp32 scaffold kernels. Same FLOP model as the tensor_core
// bench so the two engines are directly comparable (cuda_core will be far slower —
// that is the lesson: attention's matmuls belong on tensor cores).
//   usage: bench <id> [B H S D causal]   default id=2, 2 16 2048 64 0
#include "fa_cc.h"

int main(int argc,char**argv){
  if(argc<2){ fa_cc_print(); return 0; }
  int id=atoi(argv[1]);
  int B=argc>2?atoi(argv[2]):2, H=argc>3?atoi(argv[3]):16;
  int S=argc>4?atoi(argv[4]):2048, D=argc>5?atoi(argv[5]):64;
  bool causal=argc>6?atoi(argv[6])!=0:false;
  const FaCcCase* c=fa_cc_find(id); if(!c){printf("bad id\n");fa_cc_print();return 1;}
  size_t n=(size_t)B*S*H*D;
  std::vector<float> hf(n);
  for(size_t i=0;i<n;++i) hf[i]=(float)((i*2654435761u+11u)%1000)/500.f-1.f;
  float *dQ,*dK,*dV,*dO;
  CHECK_CUDA(cudaMalloc(&dQ,n*4));CHECK_CUDA(cudaMalloc(&dK,n*4));
  CHECK_CUDA(cudaMalloc(&dV,n*4));CHECK_CUDA(cudaMalloc(&dO,n*4));
  CHECK_CUDA(cudaMemcpy(dQ,hf.data(),n*4,cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dK,hf.data(),n*4,cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dV,hf.data(),n*4,cudaMemcpyHostToDevice));
  for(int i=0;i<5;++i) c->fn(dQ,dK,dV,dO,B,S,H,D,causal,0);
  CHECK_CUDA(cudaDeviceSynchronize()); CHECK_CUDA(cudaGetLastError());
  std::vector<float> times;
  for(int r=0;r<20;++r){ cudaEvent_t s,e;cudaEventCreate(&s);cudaEventCreate(&e);
    cudaEventRecord(s); c->fn(dQ,dK,dV,dO,B,S,H,D,causal,0); cudaEventRecord(e);
    cudaEventSynchronize(e); float ms;cudaEventElapsedTime(&ms,s,e); times.push_back(ms);
    cudaEventDestroy(s);cudaEventDestroy(e); }
  std::sort(times.begin(),times.end()); double ms=times[times.size()/2];
  double tflops=fa_flops(B,H,S,D,causal)/(ms*1e-3)/1e12;
  printf("%-16s B=%d H=%d S=%d D=%d causal=%d  time=%.4f ms  %.2f TFLOPS\n",
         c->name,B,H,S,D,causal,ms,tflops);
  cudaFree(dQ);cudaFree(dK);cudaFree(dV);cudaFree(dO); return 0;
}
