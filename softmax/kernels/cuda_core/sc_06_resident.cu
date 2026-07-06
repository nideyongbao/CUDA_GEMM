// sc_06 register-resident — the single-DRAM-read softmax. THE lesson: sc_02..sc_05
// are DRAM-bound (ncu: sc_05 hits 87% DRAM throughput), so the only way to go faster
// is FEWER DRAM passes. sc_05 online already cut 4·MN -> 3·MN (fused max+sum pass,
// then a re-read to write). sc_06 caches the whole row in REGISTERS on the first read,
// so it reads x ONCE and writes y ONCE: 2·MN traffic (the roofline ideal).
//   pass structure (all after a single DRAM load into regs):
//     load reg[] <- x           (1 DRAM read)
//     m = blockReduceMax(reg)   (compute, no DRAM)
//     l = blockReduceSum(exp(reg-m))
//     y <- exp(reg-m)/l         (1 DRAM write, from regs — NO second read of x)
// This is the "keep the row on-chip" principle borrowed from TinyFA (softmax on
// register-resident MMA fragments) and flash-attention's Triton layer_norm/cross_entropy
// (tl.load the whole row once, reduce, write once). exp2f + folded log2e as in TinyFA.
#include "softmax.h"

static constexpr int kBlock = 256;
static constexpr int kVptMax = 16;              // float4 per thread -> N <= 256*4*16 = 16384
static constexpr float kLog2e = 1.4426950408889634f;

__device__ __forceinline__ float warpMax(float v){
  for(int o=16;o>0;o>>=1) v=fmaxf(v,__shfl_xor_sync(0xffffffff,v,o)); return v; }
__device__ __forceinline__ float warpSum(float v){
  for(int o=16;o>0;o>>=1) v+=__shfl_xor_sync(0xffffffff,v,o); return v; }

__device__ __forceinline__ float blockReduce(float v, bool isMax){
  __shared__ float part[kBlock/32];
  int lane=threadIdx.x&31, wid=threadIdx.x>>5;
  v = isMax? warpMax(v) : warpSum(v);
  if(lane==0) part[wid]=v;
  __syncthreads();
  v = (threadIdx.x < (blockDim.x>>5)) ? part[lane] : (isMax?-INFINITY:0.f);
  if(wid==0) v = isMax? warpMax(v) : warpSum(v);
  __shared__ float bcast;
  if(threadIdx.x==0) bcast=v;
  __syncthreads();
  return bcast;
}

__global__ void sc06_resident_kernel(const float* __restrict__ x, float* __restrict__ y, int N){
  int r=blockIdx.x;
  const float4* xr=reinterpret_cast<const float4*>(x+(size_t)r*N);
  float4* yr=reinterpret_cast<float4*>(y+(size_t)r*N);
  int N4=N>>2, t=threadIdx.x, nt=blockDim.x;

  float4 reg[kVptMax];
  // ---- single DRAM read: load this thread's strided float4 into registers ----
  #pragma unroll
  for(int k=0;k<kVptMax;k++){ int j=t+k*nt; if(j<N4) reg[k]=xr[j]; }

  // ---- max over registers -> block max ----
  float m=-INFINITY;
  #pragma unroll
  for(int k=0;k<kVptMax;k++){ int j=t+k*nt; if(j<N4){ float4 v=reg[k];
    m=fmaxf(m,fmaxf(fmaxf(v.x,v.y),fmaxf(v.z,v.w))); } }
  m=blockReduce(m,true);

  // ---- sum exp2((x-m)*log2e) over registers -> block sum ----
  float s=0.f;
  #pragma unroll
  for(int k=0;k<kVptMax;k++){ int j=t+k*nt; if(j<N4){ float4 v=reg[k];
    s+=exp2f((v.x-m)*kLog2e)+exp2f((v.y-m)*kLog2e)+exp2f((v.z-m)*kLog2e)+exp2f((v.w-m)*kLog2e); } }
  s=blockReduce(s,false);
  float inv=1.f/s;

  // ---- single DRAM write: y from registers, NO second read of x ----
  #pragma unroll
  for(int k=0;k<kVptMax;k++){ int j=t+k*nt; if(j<N4){ float4 v=reg[k],o;
    o.x=exp2f((v.x-m)*kLog2e)*inv; o.y=exp2f((v.y-m)*kLog2e)*inv;
    o.z=exp2f((v.z-m)*kLog2e)*inv; o.w=exp2f((v.w-m)*kLog2e)*inv; yr[j]=o; } }
}

void softmax_resident(const float* x, float* y, int M, int N, cudaStream_t st){
  // register capacity: N4 must fit in kVptMax float4 per thread
  if(N%4!=0 || (N>>2) > kBlock*kVptMax){ softmax_online(x,y,M,N,st); return; } // fallback
  sc06_resident_kernel<<<M,kBlock,0,st>>>(x,y,N);
}
