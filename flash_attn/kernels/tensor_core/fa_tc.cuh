// From-scratch raw-CUDA FlashAttention-2 forward (tensor-core, mma.sync m16n8k16).
// Incremental-ladder shared kernel; template flags turn on one optimization per rung.
// Canonical Ampere FA2 layout (as in CUTLASS / TinyFA / lubits.ch flash_attention_
// from_scratch, credited); our own assembly, validated vs fa_cpu_ref.
//
// Scope: head_dim=128, MHA, bf16/fp16 in/out + fp32 softmax, seqlen % Bc == 0.
// Block = 4 warps x 32; each warp owns 16 query rows (FA2, no cross-warp split-K).
// Operand loads are STREAMED per k16-tile (all-resident K/V would blow the RF).
//
// Fragment layout (m16n8k16, 32 lanes; g=lane/4 in 0..7, tig=lane%4 in 0..3):
//   C(m16n8) ownership: c0=(g,2tig) c1=(g,2tig+1) c2=(g+8,2tig) c3=(g+8,2tig+1).
//   S stored as S[2 rowgroups][Bc/4]; (c0,c1)->half2,(c2,c3)->half2 across 2 adjacent
//   n8-tiles == the A-fragment for the P@V mma (register-direct S->P relayout).
#pragma once
#include "fa_tc_ptx.cuh"

namespace fatc {

static constexpr int WARPS = 4, ROWS_W = 16, BR = 64, BC = 64, DH = 128;
static constexpr int KT_QK = DH / 16;   // 8  k16-tiles over D for S=QK^T
static constexpr int NT_QK = BC / 8;    // 8  n8-tiles over Bc
static constexpr int KT_PV = BC / 16;   // 4  k16-tiles over Bc for O=PV
static constexpr int NT_PV = DH / 8;    // 16 n8-tiles over D
static constexpr int CF = DH / 8;       // 16 col-fragments (8 elems) per smem row
static constexpr float LOG2E = 1.4426950408889634f;

__device__ __forceinline__ int swz(int row, int cf, bool on) { return on ? (cf ^ (row & 7)) : cf; }

// cooperative GMEM->SMEM copy of a [rows x DH] tile (sync uint4 or cp.async)
template <typename T, bool ASYNC, bool SWZ>
__device__ __forceinline__ void load_tile(const T* g, int gstride, T* s, int rows, int tid) {
  for (int idx = tid; idx < rows * CF; idx += WARPS * 32) {
    int r = idx / CF, cf = idx % CF;
    const T* gp = g + (size_t)r * gstride + cf * 8;
    T* sp = s + (size_t)r * DH + swz(r, cf, SWZ) * 8;
    if constexpr (ASYNC) cp_async_cg16(sp, gp);
    else *reinterpret_cast<uint4*>(sp) = *reinterpret_cast<const uint4*>(gp);
  }
}

// ldmatrix.x4 non-transposed A/B fragment for 16 rows @ rowbase, k16-tile kt
template <typename T, bool SWZ>
__device__ __forceinline__ void ld_Afrag(const T* s, int rowbase, int kt, int lane,
                                         uint32_t& a0, uint32_t& a1, uint32_t& a2, uint32_t& a3) {
  int trow = lane % 16, thi = lane / 16;
  int cf = kt * 2 + thi;
  const T* p = s + (size_t)(rowbase + trow) * DH + swz(rowbase + trow, cf, SWZ) * 8;
  ldmatrix_x4(p, a0, a1, a2, a3);
}

// ---- compute one KV tile: S=QK^T -> (causal) -> online softmax -> O += P@V ----
// Factored out so the single-buffer and double-buffer loops share it (readability).
template <typename T, bool SWZ, bool EXP2, bool CAUSAL>
__device__ __forceinline__ void compute_kv_tile(
    const T* sK, const T* sV, const uint32_t aQ[KT_QK][4], float oC[2][NT_PV * 2],
    float rmax[2], float rsum[2], int j, int qtile, int wrow0, int g, int tig, int lane, float scale) {
  // S = Q @ K^T  -> sC[2 rowgroups][NT_QK*2], accumulate over k-tiles
  float sC[2][NT_QK * 2];
  #pragma unroll
  for (int m = 0; m < 2; ++m) for (int n = 0; n < NT_QK * 2; ++n) sC[m][n] = 0.f;
  #pragma unroll
  for (int kt = 0; kt < KT_QK; ++kt) {
    uint32_t bK[NT_QK][2];
    #pragma unroll
    for (int rb = 0; rb < NT_QK / 2; ++rb) {
      uint32_t r0, r1, r2, r3;
      ld_Afrag<T, SWZ>(sK, rb * 16, kt, lane, r0, r1, r2, r3);
      bK[rb * 2][0] = r0; bK[rb * 2][1] = r2;
      bK[rb * 2 + 1][0] = r1; bK[rb * 2 + 1][1] = r3;
    }
    #pragma unroll
    for (int n = 0; n < NT_QK; ++n)
      mma_m16n8k16<T>(sC[0][2 * n], sC[0][2 * n + 1], sC[1][2 * n], sC[1][2 * n + 1],
                      aQ[kt][0], aQ[kt][1], aQ[kt][2], aQ[kt][3], bK[n][0], bK[n][1]);
  }
  // causal mask: drop keys j*Bc+col > query row
  if constexpr (CAUSAL) {
    #pragma unroll
    for (int m = 0; m < 2; ++m) {
      int qrow = qtile * BR + wrow0 + g + m * 8;
      #pragma unroll
      for (int n = 0; n < NT_QK; ++n) {
        int kcol = j * BC + n * 8 + tig * 2;
        if (kcol > qrow) sC[m][2 * n] = -INFINITY;
        if (kcol + 1 > qrow) sC[m][2 * n + 1] = -INFINITY;
      }
    }
  }
  // online softmax (quad rows, width-4 shuffle)
  #pragma unroll
  for (int m = 0; m < 2; ++m) {
    float cur = -INFINITY;
    #pragma unroll
    for (int n = 0; n < NT_QK * 2; ++n) cur = fmaxf(cur, sC[m][n] * scale);
    cur = warpMax4(cur);
    float mnew = fmaxf(rmax[m], cur);
    float corr = (rmax[m] == -INFINITY) ? 0.f : (EXP2 ? exp2f(rmax[m] - mnew) : expf(rmax[m] - mnew));
    rsum[m] *= corr;
    #pragma unroll
    for (int n = 0; n < NT_PV * 2; ++n) oC[m][n] *= corr;
    float s = 0.f;
    #pragma unroll
    for (int n = 0; n < NT_QK * 2; ++n) {
      float e = EXP2 ? exp2f(sC[m][n] * scale - mnew) : expf(sC[m][n] * scale - mnew);
      sC[m][n] = e; s += e;
    }
    rsum[m] += warpSum4(s);
    rmax[m] = mnew;
  }
  // S(f32) -> P(16-bit) A-fragments, register-direct
  uint32_t pA[2][NT_QK];
  #pragma unroll
  for (int m = 0; m < 2; ++m)
    #pragma unroll
    for (int n = 0; n < NT_QK; ++n) {
      if constexpr (!__is_same(T, __nv_bfloat16)) {
        __half2 h = __floats2half2_rn(sC[m][2 * n], sC[m][2 * n + 1]);
        pA[m][n] = *reinterpret_cast<uint32_t*>(&h);
      } else {
        __nv_bfloat162 h = __floats2bfloat162_rn(sC[m][2 * n], sC[m][2 * n + 1]);
        pA[m][n] = *reinterpret_cast<uint32_t*>(&h);
      }
    }
  // O += P @ V  (V streamed per k-tile, ldmatrix.trans)
  #pragma unroll
  for (int kt = 0; kt < KT_PV; ++kt) {
    uint32_t bV[NT_PV][2];
    #pragma unroll
    for (int nb = 0; nb < NT_PV / 2; ++nb) {
      int thi = lane / 16, vrow = kt * 16 + (lane % 16), cf = nb * 2 + thi;
      const T* p = sV + (size_t)vrow * DH + swz(vrow, cf, SWZ) * 8;
      uint32_t r0, r1, r2, r3;
      ldmatrix_x4_trans(p, r0, r1, r2, r3);
      bV[nb * 2][0] = r0; bV[nb * 2][1] = r1;
      bV[nb * 2 + 1][0] = r2; bV[nb * 2 + 1][1] = r3;
    }
    #pragma unroll
    for (int n = 0; n < NT_PV; ++n)
      mma_m16n8k16<T>(oC[0][2 * n], oC[0][2 * n + 1], oC[1][2 * n], oC[1][2 * n + 1],
                      pA[0][2 * kt], pA[1][2 * kt], pA[0][2 * kt + 1], pA[1][2 * kt + 1],
                      bV[n][0], bV[n][1]);
  }
}

// ---- the kernel: DBUF selects single- vs double-buffered (pipelined) KV loop ----
template <typename T, bool SWZ, bool CPA, bool EXP2, bool CAUSAL, bool DBUF = false, int MINBLK = 1>
__global__ void __launch_bounds__(WARPS * 32, MINBLK)
fa_tc_kernel(const T* __restrict__ Qg, const T* __restrict__ Kg, const T* __restrict__ Vg,
             T* __restrict__ Og, int B, int S, int H) {
  const int qtile = blockIdx.x, h = blockIdx.y, b = blockIdx.z;
  const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
  const int g = lane / 4, tig = lane % 4;
  const int gstride = H * DH;
  const int wrow0 = warp * ROWS_W;
  const float scale = rsqrtf((float)DH) * (EXP2 ? LOG2E : 1.f);

  extern __shared__ char smem_raw[];
  T* sQ = reinterpret_cast<T*>(smem_raw);
  T* sK = sQ + BR * DH;                       // DBUF: sK[buf] = sK + buf*BC*DH
  T* sV = sK + (DBUF ? 2 : 1) * BC * DH;       // DBUF: sV[buf] = sV + buf*BC*DH

  const T* Qbase = Qg + ((size_t)(b * S + qtile * BR) * H + h) * DH;
  const T* Kbase = Kg + ((size_t)(b * S) * H + h) * DH;
  const T* Vbase = Vg + ((size_t)(b * S) * H + h) * DH;

  load_tile<T, CPA, SWZ>(Qbase, gstride, sQ, BR, tid);
  if constexpr (CPA) { cp_async_commit(); cp_async_wait<0>(); }
  __syncthreads();
  uint32_t aQ[KT_QK][4];
  #pragma unroll
  for (int kt = 0; kt < KT_QK; ++kt)
    ld_Afrag<T, SWZ>(sQ, wrow0, kt, lane, aQ[kt][0], aQ[kt][1], aQ[kt][2], aQ[kt][3]);

  float oC[2][NT_PV * 2];
  #pragma unroll
  for (int m = 0; m < 2; ++m) for (int n = 0; n < NT_PV * 2; ++n) oC[m][n] = 0.f;
  float rmax[2] = {-INFINITY, -INFINITY}, rsum[2] = {0.f, 0.f};

  int nKV = S / BC;
  if (CAUSAL) { int lim = (qtile * BR + BR + BC - 1) / BC; if (lim < nKV) nKV = lim; }

  if constexpr (DBUF) {
    // software pipeline: prefetch tile j+1 into the other buffer while computing tile j.
    auto Kbuf = [&](int bfr) { return sK + bfr * BC * DH; };
    auto Vbuf = [&](int bfr) { return sV + bfr * BC * DH; };
    load_tile<T, true, SWZ>(Kbase, gstride, Kbuf(0), BC, tid);
    load_tile<T, true, SWZ>(Vbase, gstride, Vbuf(0), BC, tid);
    cp_async_commit();
    for (int j = 0; j < nKV; ++j) {
      int cur = j & 1;
      cp_async_wait<0>();
      __syncthreads();
      if (j + 1 < nKV) {   // prefetch next tile into the OTHER buffer (overlaps compute below)
        load_tile<T, true, SWZ>(Kbase + (size_t)(j + 1) * BC * gstride, gstride, Kbuf(cur ^ 1), BC, tid);
        load_tile<T, true, SWZ>(Vbase + (size_t)(j + 1) * BC * gstride, gstride, Vbuf(cur ^ 1), BC, tid);
        cp_async_commit();
      }
      compute_kv_tile<T, SWZ, EXP2, CAUSAL>(Kbuf(cur), Vbuf(cur), aQ, oC, rmax, rsum,
                                            j, qtile, wrow0, g, tig, lane, scale);
      __syncthreads();   // free buf `cur` before it is overwritten at j+2
    }
  } else {
    for (int j = 0; j < nKV; ++j) {
      load_tile<T, CPA, SWZ>(Kbase + (size_t)j * BC * gstride, gstride, sK, BC, tid);
      load_tile<T, CPA, SWZ>(Vbase + (size_t)j * BC * gstride, gstride, sV, BC, tid);
      if constexpr (CPA) { cp_async_commit(); cp_async_wait<0>(); }
      __syncthreads();
      compute_kv_tile<T, SWZ, EXP2, CAUSAL>(sK, sV, aQ, oC, rmax, rsum,
                                            j, qtile, wrow0, g, tig, lane, scale);
      __syncthreads();
    }
  }

  // finalize + write O
  #pragma unroll
  for (int m = 0; m < 2; ++m) {
    float inv = (rsum[m] > 0.f) ? 1.f / rsum[m] : 0.f;
    int row = qtile * BR + wrow0 + g + m * 8;
    if (row >= S) continue;
    #pragma unroll
    for (int n = 0; n < NT_PV; ++n) {
      int col = n * 8 + tig * 2;
      T* op = Og + (((size_t)(b * S + row) * H + h) * DH) + col;
      op[0] = (T)(oC[m][2 * n] * inv);
      op[1] = (T)(oC[m][2 * n + 1] * inv);
    }
  }
}

}  // namespace fatc
