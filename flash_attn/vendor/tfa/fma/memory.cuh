/*
 * TinyFA
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include "../ptx.cuh"
#include "../utils.cuh"
#include "layout.cuh"

namespace tfa::fma {

template <typename Config>
struct MemLoader {
  using DType = typename Config::DType;
  using Layout = TileLayout<Config>;
  using VecT = int4;  // 128-bit

  static constexpr int kElemsPerVec = Config::kElemsPerVecLoad;
  static constexpr int kHeadDim = Config::kHeadDim;
  static constexpr int kNumVecs = kHeadDim / kElemsPerVec;

  static_assert(kHeadDim % kElemsPerVec == 0, "kHeadDim must be divisible by kElemsPerVec");

  template <int Rows, typename Context, bool IsLoad, typename SmemPtr, typename GmemPtr>
  __device__ __forceinline__ static void copy(SmemPtr smem, GmemPtr gmem, int stride, int validRows,
                                              const Context& ctx) {
    for (int idx = ctx.threadId; idx < Rows * kNumVecs; idx += ctx.blockSize) {
      int row = idx / kNumVecs;
      int col = (idx % kNumVecs) * kElemsPerVec;
      int smemIdx = Layout::map(row, col, kHeadDim);

      if constexpr (IsLoad) {
        // gmem -> smem
        const VecT* gPtr = reinterpret_cast<const VecT*>(gmem + row * stride + col);
        *reinterpret_cast<VecT*>(&smem[smemIdx]) = (row < validRows) ? *gPtr : make_int4(0, 0, 0, 0);
      } else if (row < validRows) {
        // smem -> gmem
        *reinterpret_cast<VecT*>(gmem + row * stride + col) = *reinterpret_cast<const VecT*>(&smem[smemIdx]);
      }
    }
  }

  template <int Rows, typename Context>
  __device__ __forceinline__ static void gm2sm(DType* __restrict__ smem, const DType* __restrict__ gmem, int stride,
                                               int validRows, const Context& ctx) {
    copy<Rows, Context, true>(smem, gmem, stride, validRows, ctx);
  }

  // async load via cp.async
  template <int Rows, typename Context>
  __device__ __forceinline__ static void gm2smAsync(DType* __restrict__ smem, const DType* __restrict__ gmem,
                                                    int stride, int validRows, const Context& ctx) {
    for (int idx = ctx.threadId; idx < Rows * kNumVecs; idx += ctx.blockSize) {
      int row = idx / kNumVecs;
      int col = (idx % kNumVecs) * kElemsPerVec;
      int smemIdx = Layout::map(row, col, kHeadDim);

      DType* sPtr = &smem[smemIdx];
      const DType* gPtr = gmem + row * stride + col;

      cpAsync128bZfill(sPtr, gPtr, row < validRows);
    }
  }

  template <int Rows, typename Context>
  __device__ __forceinline__ static void sm2gm(DType* __restrict__ gmem, const DType* __restrict__ smem, int stride,
                                               int validRows, const Context& ctx) {
    copy<Rows, Context, false>(smem, gmem, stride, validRows, ctx);
  }

  __device__ __forceinline__ static void sm2reg(DType* __restrict__ reg, const DType* __restrict__ smem) {
    *reinterpret_cast<VecT*>(reg) = *reinterpret_cast<const VecT*>(smem);
  }
};

template <typename Config>
struct MemStore {
  using DType = typename Config::DType;
  using Layout = TileLayout<Config>;

  static constexpr int kHeadDim = Config::kHeadDim;

  template <typename AccO, typename Context>
  __device__ __forceinline__ static void storeO(DType* __restrict__ gmem, DType* __restrict__ smem, int stride,
                                                const AccO& accO, const Context& ctx) {
    static constexpr int kRowsPerWarp = Config::kRowsPerWarp;
    static constexpr int kDimsPerLane = kHeadDim / Config::kWarpSize;

    // accO -> smem
    const int validRows = ctx.validWarpRows();
#pragma unroll
    for (int m = 0; m < kRowsPerWarp; m++) {
      if (m < validRows) {
        int row = ctx.warpRowOffset + m;
#pragma unroll
        for (int k = 0; k < kDimsPerLane; k++) {
          int col = ctx.laneId * kDimsPerLane + k;
          int smemIdx = Layout::map(row, col, kHeadDim);
          smem[smemIdx] = fromFloat<DType>(accO[m][k]);
        }
      }
    }
    __syncthreads();

    // smem -> gmem
    MemLoader<Config>::template sm2gm<Config::kBr>(gmem, smem, stride, ctx.tileSizeQ, ctx);
  }
};

template <typename Config>
struct MemLoaderPagedVarLen {
  using DType = typename Config::DType;
  using Layout = TileLayout<Config>;
  using VecT = int4;  // 128-bit

  static constexpr int kElemsPerVec = Config::kElemsPerVecLoad;
  static constexpr int kHeadDim = Config::kHeadDim;
  static constexpr int kNumVecs = kHeadDim / kElemsPerVec;

  static_assert(kHeadDim % kElemsPerVec == 0, "kHeadDim must be divisible by kElemsPerVec");

  template <int Rows, bool kAsync, typename Context, typename Params>
  __device__ __forceinline__ static void gm2smPagedVarLenImpl(DType* __restrict__ smem, const Params& params,
                                                              const Context& ctx, int tileKVStart, int tileSizeKV,
                                                              bool isK) {
    const int pageSize = params.pageSize;
    const bool isPow2 = (pageSize & (pageSize - 1)) == 0;
    const int pageShift = isPow2 ? (__ffs(pageSize) - 1) : 0;
    const int pageMask = isPow2 ? (pageSize - 1) : 0;

    const int firstPage = isPow2 ? (tileKVStart >> pageShift) : (tileKVStart / pageSize);
    const int lastRowKV = tileKVStart + tileSizeKV - 1;
    const int lastPage = (tileSizeKV > 0) ? (isPow2 ? (lastRowKV >> pageShift) : (lastRowKV / pageSize)) : firstPage;
    const bool singlePage = (firstPage == lastPage) && (tileSizeKV > 0);

    if (singlePage) {
      const DType* pagePtr = isK ? params.kPagePtr(ctx, firstPage) : params.vPagePtr(ctx, firstPage);
      const int firstOffset = isPow2 ? (tileKVStart & pageMask) : (tileKVStart % pageSize);
      const DType* basePtr = pagePtr + firstOffset * kHeadDim;

      for (int idx = ctx.threadId; idx < Rows * kNumVecs; idx += ctx.blockSize) {
        int row = idx / kNumVecs;
        int col = (idx % kNumVecs) * kElemsPerVec;
        int smemIdx = Layout::map(row, col, kHeadDim);

        if constexpr (kAsync) {
          DType* sPtr = &smem[smemIdx];
          if (row < tileSizeKV) {
            cpAsync128bZfill(sPtr, basePtr + row * kHeadDim + col, true);
          } else {
            cpAsync128bZfill(sPtr, sPtr, false);
          }
        } else {
          if (row < tileSizeKV) {
            const VecT* gPtr = reinterpret_cast<const VecT*>(basePtr + row * kHeadDim + col);
            *reinterpret_cast<VecT*>(&smem[smemIdx]) = *gPtr;
          } else {
            *reinterpret_cast<VecT*>(&smem[smemIdx]) = make_int4(0, 0, 0, 0);
          }
        }
      }
    } else if (isPow2) {
      for (int idx = ctx.threadId; idx < Rows * kNumVecs; idx += ctx.blockSize) {
        int row = idx / kNumVecs;
        int col = (idx % kNumVecs) * kElemsPerVec;
        int smemIdx = Layout::map(row, col, kHeadDim);

        if (row < tileSizeKV) {
          int kvPos = tileKVStart + row;
          int pageIdx = kvPos >> pageShift;
          int pageOffset = kvPos & pageMask;
          const DType* pagePtr = isK ? params.kPagePtr(ctx, pageIdx) : params.vPagePtr(ctx, pageIdx);
          const DType* gAddr = pagePtr + pageOffset * kHeadDim + col;

          if constexpr (kAsync) {
            cpAsync128bZfill(&smem[smemIdx], gAddr, true);
          } else {
            *reinterpret_cast<VecT*>(&smem[smemIdx]) = *reinterpret_cast<const VecT*>(gAddr);
          }
        } else {
          if constexpr (kAsync) {
            DType* sPtr = &smem[smemIdx];
            cpAsync128bZfill(sPtr, sPtr, false);
          } else {
            *reinterpret_cast<VecT*>(&smem[smemIdx]) = make_int4(0, 0, 0, 0);
          }
        }
      }
    } else {
      for (int idx = ctx.threadId; idx < Rows * kNumVecs; idx += ctx.blockSize) {
        int row = idx / kNumVecs;
        int col = (idx % kNumVecs) * kElemsPerVec;
        int smemIdx = Layout::map(row, col, kHeadDim);

        if (row < tileSizeKV) {
          int kvPos = tileKVStart + row;
          int pageIdx = kvPos / pageSize;
          int pageOffset = kvPos % pageSize;
          const DType* pagePtr = isK ? params.kPagePtr(ctx, pageIdx) : params.vPagePtr(ctx, pageIdx);
          const DType* gAddr = pagePtr + pageOffset * kHeadDim + col;

          if constexpr (kAsync) {
            cpAsync128bZfill(&smem[smemIdx], gAddr, true);
          } else {
            *reinterpret_cast<VecT*>(&smem[smemIdx]) = *reinterpret_cast<const VecT*>(gAddr);
          }
        } else {
          if constexpr (kAsync) {
            DType* sPtr = &smem[smemIdx];
            cpAsync128bZfill(sPtr, sPtr, false);
          } else {
            *reinterpret_cast<VecT*>(&smem[smemIdx]) = make_int4(0, 0, 0, 0);
          }
        }
      }
    }
  }

  template <int Rows, typename Context, typename Params>
  __device__ __forceinline__ static void gm2smPagedVarLen(DType* __restrict__ smem, const Params& params,
                                                          const Context& ctx, int tileKVStart, int tileSizeKV,
                                                          bool isK) {
    gm2smPagedVarLenImpl<Rows, false>(smem, params, ctx, tileKVStart, tileSizeKV, isK);
  }

  template <int Rows, typename Context, typename Params>
  __device__ __forceinline__ static void gm2smPagedVarLenAsync(DType* __restrict__ smem, const Params& params,
                                                               const Context& ctx, int tileKVStart, int tileSizeKV,
                                                               bool isK) {
    gm2smPagedVarLenImpl<Rows, true>(smem, params, ctx, tileKVStart, tileSizeKV, isK);
  }
};

}  // namespace tfa::fma
