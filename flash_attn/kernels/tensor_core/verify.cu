// FA tensor-core verify — run tfa::flashAttn, compare against fp32 CPU attention.
// fp16/bf16 tensor-core attention: atol/rtol ~2e-2 (matches TinyFA's tolerance).
//   usage: verify <fp16|bf16> [B H S D causal]   default: fp16 2 8 512 128 0
#include "fa_common.h"
#include "flash_api.cuh"

template<typename T> __host__ T   f2t(float x);
template<> __host__ __half        f2t<__half>(float x){ return __float2half(x); }
template<> __host__ __nv_bfloat16 f2t<__nv_bfloat16>(float x){ return __float2bfloat16(x); }
template<typename T> __host__ float t2f(T x);
template<> __host__ float t2f<__half>(__half x){ return __half2float(x); }
template<> __host__ float t2f<__nv_bfloat16>(__nv_bfloat16 x){ return __bfloat162float(x); }

template<typename T>
int run_verify(int B,int H,int S,int D,bool causal){
  size_t n=(size_t)B*S*H*D;
  std::vector<float> hf(n), ref(n);
  for(size_t i=0;i<n;++i) hf[i]=(float)((i*40503u+7u)%1000)/500.f-1.f;
  fa_cpu_ref(hf.data(),hf.data(),hf.data(),ref.data(),B,S,H,H,D,causal);
  std::vector<T> ht(n); for(size_t i=0;i<n;++i) ht[i]=f2t<T>(hf[i]);
  T *dQ,*dK,*dV,*dO;
  CHECK_CUDA(cudaMalloc(&dQ,n*sizeof(T))); CHECK_CUDA(cudaMalloc(&dK,n*sizeof(T)));
  CHECK_CUDA(cudaMalloc(&dV,n*sizeof(T))); CHECK_CUDA(cudaMalloc(&dO,n*sizeof(T)));
  CHECK_CUDA(cudaMemcpy(dQ,ht.data(),n*sizeof(T),cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dK,ht.data(),n*sizeof(T),cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dV,ht.data(),n*sizeof(T),cudaMemcpyHostToDevice));
  tfa::flashAttn<T>(dQ,dK,dV,dO,B,S,S,H,H,D,causal,0);
  CHECK_CUDA(cudaGetLastError()); CHECK_CUDA(cudaDeviceSynchronize());
  std::vector<T> hoT(n); CHECK_CUDA(cudaMemcpy(hoT.data(),dO,n*sizeof(T),cudaMemcpyDeviceToHost));
  double atol=2e-2,rtol=2e-2,maxa=0,maxr=0; size_t bad=0;
  for(size_t i=0;i<n;++i){ double a=t2f<T>(hoT[i]),b=ref[i],dd=fabs(a-b);
    maxa=dd>maxa?dd:maxa; double rl=dd/(fabs(b)+1e-30); maxr=rl>maxr?rl:maxr;
    if(dd>atol+rtol*fabs(b))++bad; }
  printf("tfa_mma verify B=%d H=%d S=%d D=%d causal=%d  max_abs=%.3e max_rel=%.3e bad=%zu/%zu  %s\n",
         B,H,S,D,causal,maxa,maxr,bad,n, bad==0?"PASS":"FAIL");
  cudaFree(dQ);cudaFree(dK);cudaFree(dV);cudaFree(dO);
  return bad==0?0:1;
}

int main(int argc,char**argv){
  FADtype dt=fa_parse_dtype(argc>1?argv[1]:"fp16");
  int B=argc>2?atoi(argv[2]):2, H=argc>3?atoi(argv[3]):8;
  int S=argc>4?atoi(argv[4]):512, D=argc>5?atoi(argv[5]):128;
  bool causal=argc>6?atoi(argv[6])!=0:false;
  return (dt==FA_BF16)? run_verify<__nv_bfloat16>(B,H,S,D,causal)
                      : run_verify<__half>(B,H,S,D,causal);
}
