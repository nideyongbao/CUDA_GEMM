// Canonical raw-PTX wrappers for the from-scratch tensor-core FA2 ladder.
// These asm strings are the one-and-only way to emit these Ampere instructions
// (identical across CUTLASS / TinyFA / flash_attention_from_scratch) — they are
// the hardware ABI, not creative code.
#pragma once
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace fatc {

__device__ __forceinline__ void cp_async_cg16(void* smem, const void* gmem) {
  unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(smem));
  asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n" ::"r"(s), "l"(gmem));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n"); }
template <int N> __device__ __forceinline__ void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

// ldmatrix.x4 (non-transposed): load a 16x16 region (four 8x8 b16 matrices).
__device__ __forceinline__ void ldmatrix_x4(const void* smem, uint32_t& r0, uint32_t& r1,
                                            uint32_t& r2, uint32_t& r3) {
  unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(smem));
  asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(s));
}
// ldmatrix.x4.trans: transposed load (for the V operand of P@V).
__device__ __forceinline__ void ldmatrix_x4_trans(const void* smem, uint32_t& r0, uint32_t& r1,
                                                  uint32_t& r2, uint32_t& r3) {
  unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(smem));
  asm volatile("ldmatrix.sync.aligned.x4.trans.m8n8.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(s));
}

// mma.sync.m16n8k16.row.col.f32.{f16|bf16}: D[m16,n8] = A[m16,k16] * B[k16,n8] + C
template <typename T>
__device__ __forceinline__ void mma_m16n8k16(float& d0, float& d1, float& d2, float& d3,
                                             uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                             uint32_t b0, uint32_t b1) {
  if constexpr (sizeof(T) == 2 && !__is_same(T, __nv_bfloat16)) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
  } else {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
  }
}

__device__ __forceinline__ float warpMax4(float v) {  // reduce over the 4-lane quad (a row)
  v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, 2));
  v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, 1));
  return v;
}
__device__ __forceinline__ float warpSum4(float v) {
  v += __shfl_xor_sync(0xffffffff, v, 2);
  v += __shfl_xor_sync(0xffffffff, v, 1);
  return v;
}

}  // namespace fatc
