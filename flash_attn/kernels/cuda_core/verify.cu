// FA cuda_core verify — fp32 scaffold vs fp32 CPU reference attention. Tight
// tolerance since both are fp32 (atol/rtol=1e-3).
//   usage: verify <id> [B H S D causal]   default id=2, 2 4 256 64 0
#include "fa_cc.h"

int main(int argc,char**argv){
  if(argc<2){ fa_cc_print(); return 0; }
  int id=atoi(argv[1]);
  int B=argc>2?atoi(argv[2]):2, H=argc>3?atoi(argv[3]):4;
  int S=argc>4?atoi(argv[4]):256, D=argc>5?atoi(argv[5]):64;
  bool causal=argc>6?atoi(argv[6])!=0:false;
  const FaCcCase* c=fa_cc_find(id); if(!c){printf("bad id\n");fa_cc_print();return 1;}
  size_t n=(size_t)B*S*H*D;
  std::vector<float> hf(n),ref(n),ho(n);
  for(size_t i=0;i<n;++i) hf[i]=(float)((i*40503u+7u)%1000)/500.f-1.f;
  fa_cpu_ref(hf.data(),hf.data(),hf.data(),ref.data(),B,S,H,H,D,causal);
  float *dQ,*dK,*dV,*dO;
  CHECK_CUDA(cudaMalloc(&dQ,n*4));CHECK_CUDA(cudaMalloc(&dK,n*4));
  CHECK_CUDA(cudaMalloc(&dV,n*4));CHECK_CUDA(cudaMalloc(&dO,n*4));
  CHECK_CUDA(cudaMemcpy(dQ,hf.data(),n*4,cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dK,hf.data(),n*4,cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dV,hf.data(),n*4,cudaMemcpyHostToDevice));
  c->fn(dQ,dK,dV,dO,B,S,H,D,causal,0);
  CHECK_CUDA(cudaGetLastError()); CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaMemcpy(ho.data(),dO,n*4,cudaMemcpyDeviceToHost));
  double atol=1e-3,rtol=1e-3,maxa=0,maxr=0; size_t bad=0;
  for(size_t i=0;i<n;++i){ double a=ho[i],b=ref[i],dd=fabs(a-b);
    maxa=dd>maxa?dd:maxa; double rl=dd/(fabs(b)+1e-30); maxr=rl>maxr?rl:maxr;
    if(dd>atol+rtol*fabs(b))++bad; }
  printf("%-16s B=%d H=%d S=%d D=%d causal=%d  max_abs=%.3e max_rel=%.3e bad=%zu/%zu  %s\n",
         c->name,B,H,S,D,causal,maxa,maxr,bad,n,bad==0?"PASS":"FAIL");
  cudaFree(dQ);cudaFree(dK);cudaFree(dV);cudaFree(dO); return bad==0?0:1;
}
