/*
 * TinyFA
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include <cutlass/numeric_conversion.h>

#include "../ptx.cuh"
#include "layout.cuh"

namespace tfa::mma {

template <typename Config>
struct MemConfig {
  using Smem = SmemLayout<Config>;
  using Element = typename Smem::Element;

  static constexpr int kBlockKSmem = Smem::kBlockKSmem;
  static constexpr int kNumThreads = Config::kNumThreads;

  static constexpr int kGmemElemsPerLoad = sizeof(cute::uint128_t) / sizeof(Element);
  static constexpr int kGmemThreadsPerRow = kBlockKSmem / kGmemElemsPerLoad;

  using GmemLayoutAtom =
      cute::Layout<cute::Shape<cute::Int<kNumThreads / kGmemThreadsPerRow>, cute::Int<kGmemThreadsPerRow>>,
                   cute::Stride<cute::Int<kGmemThreadsPerRow>, cute::_1>>;

  // ignore Config::kUseCpAsync
  static_assert(Config::kUseCpAsync, "MMA need cp.async");
  using GmemCopyStruct = cute::SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>;

  using GmemTiledCopyQKV = decltype(cute::make_tiled_copy(cute::Copy_Atom<GmemCopyStruct, Element>{}, GmemLayoutAtom{},
                                                          cute::Layout<cute::Shape<cute::_1, cute::_8>>{}));

  using GmemTiledCopyO =
      decltype(cute::make_tiled_copy(cute::Copy_Atom<cute::AutoVectorizingCopyWithAssumedAlignment<128>, Element>{},
                                     GmemLayoutAtom{}, cute::Layout<cute::Shape<cute::_1, cute::_8>>{}));

  using SmemCopyAtomO = cute::Copy_Atom<cute::AutoVectorizingCopyWithAssumedAlignment<128>, Element>;
};

struct MemLoader {
  template <typename GmemCopy, typename TGg, typename TGs, typename TCo>
  __device__ __forceinline__ static void gm2sm(const GmemCopy& gmemCopy, TGs& tGs, const TGg& tGg, const TCo& tCo,
                                               int maxRows, int tileIdx) {
    using namespace cute;

#pragma unroll
    for (int m = 0; m < size<1>(tGs); m++) {
      if (get<0>(tCo(0, m, 0)) < maxRows) {
#pragma unroll
        for (int k = 0; k < size<2>(tGs); k++) {
          copy(gmemCopy, tGg(_, m, k, tileIdx), tGs(_, m, k));
        }
      } else {
        clear(tGs(_, m, _));
      }
    }
  }

  template <typename GmemCopy, typename TGg, typename TGs, typename TCo>
  __device__ __forceinline__ static void gm2sm(const GmemCopy& gmemCopy, TGs& tGs, const TGg& tGg, const TCo& tCo,
                                               int maxRows) {
    using namespace cute;

#pragma unroll
    for (int m = 0; m < size<1>(tGs); m++) {
      if (get<0>(tCo(0, m, 0)) < maxRows) {
#pragma unroll
        for (int k = 0; k < size<2>(tGs); k++) {
          copy(gmemCopy, tGg(_, m, k), tGs(_, m, k));
        }
      } else {
        clear(tGs(_, m, _));
      }
    }
  }
};

struct MemStore {
  template <typename Config, typename AccO, typename TiledMma, typename GO, typename BlockInfo_>
  __device__ __forceinline__ static void storeO(AccO& accO, typename SmemLayout<Config>::Element* smemQ,
                                                const TiledMma& tiledMma, int tidx, const GO& gO,
                                                const BlockInfo_& blockInfo) {
    using namespace cute;

    using Smem = SmemLayout<Config>;
    using Mem = MemConfig<Config>;
    using ElemType = typename Smem::Element;

    // float -> (fp16/bf16)
    constexpr int numelO = decltype(size(accO))::value;
    cutlass::NumericArrayConverter<ElemType, float, numelO> converterO;
    auto fragO = converterO(*reinterpret_cast<const cutlass::Array<float, numelO>*>(accO.data()));
    auto rO = make_tensor(make_rmem_ptr<ElemType>(&fragO), accO.layout());

    // regs -> smem
    auto sO = make_tensor(make_smem_ptr(smemQ), typename Smem::SmemLayoutO{});
    auto smemTiledCopyO = make_tiled_copy_C(typename Mem::SmemCopyAtomO{}, tiledMma);
    auto smemThrCopyO = smemTiledCopyO.get_thread_slice(tidx);
    auto taccOrO = smemThrCopyO.retile_S(rO);
    auto taccOsO = smemThrCopyO.partition_D(sO);
    copy(smemTiledCopyO, taccOrO, taccOsO);
    __syncthreads();

    // smem -> gmem
    typename Mem::GmemTiledCopyO gmemTiledCopyO;
    auto gmemThrCopyO = gmemTiledCopyO.get_thread_slice(tidx);
    auto tOsO = gmemThrCopyO.partition_S(sO);
    auto tOgO = gmemThrCopyO.partition_D(gO);

    auto tOrOOut = make_tensor<ElemType>(shape(tOgO));
    copy(gmemTiledCopyO, tOsO, tOrOOut);

    auto cO = make_identity_tensor(make_shape(size<0>(sO), size<1>(sO)));
    auto tOcO = gmemThrCopyO.partition_D(cO);
#pragma unroll
    for (int m = 0; m < size<1>(tOgO); m++) {
      if (get<0>(tOcO(0, m, 0)) < blockInfo.tileSizeQ) {
#pragma unroll
        for (int k = 0; k < size<2>(tOgO); k++) {
          copy(tOrOOut(_, m, k), tOgO(_, m, k));
        }
      }
    }
  }
};

struct PagedMemLoader {
  static constexpr int kMaxPagesPerTile = 16;

  template <typename Config, typename Params, typename BlockInfo_>
  __device__ __forceinline__ static void precomputePagePtrs(void* __restrict__ scratchSmem, const Params& params,
                                                            const BlockInfo_& blockInfo, int tileStartKV,
                                                            int tileSizeKV, int tidx) {
    if (tileSizeKV <= 0) {
      return;
    }

    const int pageSize = params.pageSize;
    const bool isPow2 = (pageSize & (pageSize - 1)) == 0;
    const int pageShift = isPow2 ? (__ffs(pageSize) - 1) : 0;

    const int firstPage = isPow2 ? (tileStartKV >> pageShift) : (tileStartKV / pageSize);
    const int lastRowKV = tileStartKV + tileSizeKV - 1;
    const int lastPage = isPow2 ? (lastRowKV >> pageShift) : (lastRowKV / pageSize);
    const int numPages = lastPage - firstPage + 1;

    auto* scratch = reinterpret_cast<uintptr_t*>(scratchSmem);

    if (tidx < numPages) {
      int pageIdx = firstPage + tidx;
      scratch[tidx] = reinterpret_cast<uintptr_t>(params.kPagePtr(blockInfo, pageIdx));
      scratch[kMaxPagesPerTile + tidx] = reinterpret_cast<uintptr_t>(params.vPagePtr(blockInfo, pageIdx));
    }
  }

  template <typename Config, typename Params>
  __device__ __forceinline__ static void gm2smPagedCached(typename SmemLayout<Config>::Element* __restrict__ smemKV,
                                                          const void* __restrict__ scratchSmem, const Params& params,
                                                          int tileStartKV, int tileSizeKV, bool isK, int tidx) {
    using namespace cute;
    using Smem = SmemLayout<Config>;
    using Element = typename Smem::Element;

    static constexpr int kBc = Config::kBc;
    static constexpr int kHeadDim = Config::kHeadDim;
    static constexpr int kNumThreads = Config::kNumThreads;
    static constexpr int kElemsPerLoad = sizeof(uint128_t) / sizeof(Element);
    static constexpr int kLoadsPerRow = kHeadDim / kElemsPerLoad;

    auto sKV = make_tensor(make_smem_ptr(smemKV), typename Smem::SmemLayoutKV{});

    const int pageSize = params.pageSize;
    const bool isPow2 = (pageSize & (pageSize - 1)) == 0;
    const int pageShift = isPow2 ? (__ffs(pageSize) - 1) : 0;
    const int pageMask = isPow2 ? (pageSize - 1) : 0;
    const int firstPageOffset = isPow2 ? (tileStartKV & pageMask) : (tileStartKV % pageSize);

    const auto* scratch = reinterpret_cast<const uintptr_t*>(scratchSmem);
    const int ptrOffset = isK ? 0 : kMaxPagesPerTile;

    for (int idx = tidx; idx < kBc * kLoadsPerRow; idx += kNumThreads) {
      int row = idx / kLoadsPerRow;
      int col = (idx % kLoadsPerRow) * kElemsPerLoad;
      Element* sPtr = &sKV(row, col);

      if (row < tileSizeKV) {
        int localPos = firstPageOffset + row;
        int localPageIdx = isPow2 ? (localPos >> pageShift) : (localPos / pageSize);
        int localPageOff = isPow2 ? (localPos & pageMask) : (localPos % pageSize);
        const auto* pageBase = reinterpret_cast<const Element*>(scratch[ptrOffset + localPageIdx]);
        cpAsync128bZfill(sPtr, pageBase + localPageOff * kHeadDim + col, true);
      } else {
        cpAsync128bZfill(sPtr, sPtr, false);
      }
    }
  }

  template <typename Config, typename Params, typename BlockInfo_>
  __device__ __forceinline__ static void gm2smPaged(typename SmemLayout<Config>::Element* __restrict__ smemKV,
                                                    const Params& params, const BlockInfo_& blockInfo, int tileStartKV,
                                                    int tileSizeKV, bool isK, int tidx) {
    using namespace cute;
    using Smem = SmemLayout<Config>;
    using Element = typename Smem::Element;

    static constexpr int kBc = Config::kBc;
    static constexpr int kHeadDim = Config::kHeadDim;
    static constexpr int kNumThreads = Config::kNumThreads;
    static constexpr int kElemsPerLoad = sizeof(uint128_t) / sizeof(Element);
    static constexpr int kLoadsPerRow = kHeadDim / kElemsPerLoad;

    auto sKV = make_tensor(make_smem_ptr(smemKV), typename Smem::SmemLayoutKV{});

    const int pageSize = params.pageSize;
    const bool isPow2 = (pageSize & (pageSize - 1)) == 0;
    const int pageShift = isPow2 ? (__ffs(pageSize) - 1) : 0;
    const int pageMask = isPow2 ? (pageSize - 1) : 0;

    const int firstPage = isPow2 ? (tileStartKV >> pageShift) : (tileStartKV / pageSize);
    const int lastRowKV = tileStartKV + tileSizeKV - 1;
    const int lastPage = (tileSizeKV > 0) ? (isPow2 ? (lastRowKV >> pageShift) : (lastRowKV / pageSize)) : firstPage;
    const bool singlePage = (firstPage == lastPage) && (tileSizeKV > 0);

    const Element* basePtr = nullptr;
    if (singlePage) {
      const auto* pagePtr = reinterpret_cast<const Element*>(isK ? params.kPagePtr(blockInfo, firstPage)
                                                                 : params.vPagePtr(blockInfo, firstPage));
      int firstOffset = isPow2 ? (tileStartKV & pageMask) : (tileStartKV % pageSize);
      basePtr = pagePtr + firstOffset * kHeadDim;
    }

    for (int idx = tidx; idx < kBc * kLoadsPerRow; idx += kNumThreads) {
      int row = idx / kLoadsPerRow;
      int col = (idx % kLoadsPerRow) * kElemsPerLoad;
      Element* sPtr = &sKV(row, col);

      if (row < tileSizeKV) {
        const Element* gAddr;
        if (singlePage) {
          gAddr = basePtr + row * kHeadDim + col;
        } else {
          int kvPos = tileStartKV + row;
          int pageIdx = isPow2 ? (kvPos >> pageShift) : (kvPos / pageSize);
          int pageOffset = isPow2 ? (kvPos & pageMask) : (kvPos % pageSize);
          const auto* pagePtr = reinterpret_cast<const Element*>(isK ? params.kPagePtr(blockInfo, pageIdx)
                                                                     : params.vPagePtr(blockInfo, pageIdx));
          gAddr = pagePtr + pageOffset * kHeadDim + col;
        }
        cpAsync128bZfill(sPtr, gAddr, true);
      } else {
        cpAsync128bZfill(sPtr, sPtr, false);
      }
    }
  }
};

}  // namespace tfa::mma
