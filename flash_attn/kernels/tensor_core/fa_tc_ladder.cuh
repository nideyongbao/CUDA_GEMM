// The incremental raw-CUDA tensor-core FA2 ladder: id -> config-gated rung.
// Each rung flips ONE flag on vs the previous (one delta per rung), mirroring the
// GEMM tensor_core ladder (tc_01..tc_06). Shared kernel = fa_tc.cuh.
//   id 1 fa_tc_01_base     : raw mma.sync + ldmatrix, sync copy, no swizzle, expf
//   id 2 fa_tc_02_swizzle  : + XOR-swizzle smem (kill ldmatrix bank conflicts)
//   id 3 fa_tc_03_cpasync  : + cp.async GMEM->SMEM
//   id 4 fa_tc_04_exp2     : + exp2f with folded log2e
//   id 5 fa_tc_05_occ      : + __launch_bounds__ min-3-blocks (regs 184->168, occ 12.5%->18.4%)
//   id 6 fa_tc_06_pipeline : + double-buffered cp.async (prefetch tile j+1 while computing j)
#pragma once
#include "fa_tc.cuh"
#include <vector>

namespace fatc {

template <typename T, bool SWZ, bool CPA, bool EXP2, bool DBUF, int MINBLK>
void launch_cfg(const T* Q, const T* K, const T* V, T* O, int B, int S, int H,
                bool causal, cudaStream_t st) {
  size_t smem = (BR * DH + (DBUF ? 4 : 2) * BC * DH) * sizeof(T);
  dim3 grid(S / BR, H, B), block(WARPS * 32);
  if (causal) {
    auto k = fa_tc_kernel<T, SWZ, CPA, EXP2, true, DBUF, MINBLK>;
    cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    k<<<grid, block, smem, st>>>(Q, K, V, O, B, S, H);
  } else {
    auto k = fa_tc_kernel<T, SWZ, CPA, EXP2, false, DBUF, MINBLK>;
    cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    k<<<grid, block, smem, st>>>(Q, K, V, O, B, S, H);
  }
}

template <typename T>
void launch_id(int id, const T* Q, const T* K, const T* V, T* O, int B, int S, int H,
               bool causal, cudaStream_t st) {
  switch (id) {
    case 1: launch_cfg<T, false, false, false, false, 1>(Q, K, V, O, B, S, H, causal, st); break;
    case 2: launch_cfg<T, true,  false, false, false, 1>(Q, K, V, O, B, S, H, causal, st); break;
    case 3: launch_cfg<T, true,  true,  false, false, 1>(Q, K, V, O, B, S, H, causal, st); break;
    case 4: launch_cfg<T, true,  true,  true,  false, 1>(Q, K, V, O, B, S, H, causal, st); break;
    case 5: launch_cfg<T, true,  true,  true,  false, 3>(Q, K, V, O, B, S, H, causal, st); break;
    case 6: launch_cfg<T, true,  true,  true,  true,  2>(Q, K, V, O, B, S, H, causal, st); break;
    default: break;
  }
}

inline const std::vector<std::pair<int, const char*>>& fa_tc_registry() {
  static const std::vector<std::pair<int, const char*>> r = {
      {1, "fa_tc_01_base"}, {2, "fa_tc_02_swizzle"}, {3, "fa_tc_03_cpasync"},
      {4, "fa_tc_04_exp2"}, {5, "fa_tc_05_occ"}, {6, "fa_tc_06_pipeline"}};
  return r;
}
inline const char* fa_tc_name(int id) {
  for (auto& p : fa_tc_registry()) if (p.first == id) return p.second;
  return nullptr;
}
inline void fa_tc_print() { printf("fa_tc ids:\n"); for (auto& p : fa_tc_registry()) printf("  %d  %s\n", p.first, p.second); }

}  // namespace fatc
