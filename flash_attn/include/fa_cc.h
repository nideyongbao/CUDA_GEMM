// FA cuda_core scaffold — registry of from-scratch fp32 FlashAttention kernels that
// teach the FA2 algorithm (online softmax, tiling, delayed normalization) WITHOUT
// tensor cores, before the tensor_core path fuses it onto mma.sync. Layout [B,S,H,D].
#pragma once
#include "fa_common.h"

using FaCcFn = void(*)(const float*Q,const float*K,const float*V,float*O,
                       int B,int S,int H,int D,bool causal,cudaStream_t);

void fa_cc_stream(const float*,const float*,const float*,float*,int,int,int,int,bool,cudaStream_t); // fa_cc_01
void fa_cc_tiled (const float*,const float*,const float*,float*,int,int,int,int,bool,cudaStream_t); // fa_cc_02

struct FaCcCase{ int id; const char* name; FaCcFn fn; };
inline const std::vector<FaCcCase>& fa_cc_registry(){
  static const std::vector<FaCcCase> r={
    {1,"fa_cc_01_stream", fa_cc_stream},
    {2,"fa_cc_02_tiled",  fa_cc_tiled},
  };
  return r;
}
inline const FaCcCase* fa_cc_find(int id){
  for(auto&c:fa_cc_registry()) if(c.id==id) return &c; return nullptr;
}
inline void fa_cc_print(){ printf("FA cuda_core ids:\n");
  for(auto&c:fa_cc_registry()) printf("  %d  %s\n",c.id,c.name); }
