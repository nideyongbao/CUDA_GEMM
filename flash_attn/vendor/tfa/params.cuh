/*
 * TinyFA
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include "utils.cuh"

namespace tfa {

template <typename DType_, int kHeadDim_>
struct ParamsBase {
  using DType = DType_;
  static constexpr int kHeadDim = kHeadDim_;

  const DType* __restrict__ Q;
  const DType* __restrict__ K;
  const DType* __restrict__ V;
  DType* __restrict__ O;

  int seqDimQ;    // numHeadsQ * kHeadDim
  int seqDimKV;   // numHeadsKV * kHeadDim
  int groupSize;  // numHeadsQ / numHeadsKV (GQA)

  __device__ __forceinline__ int getKVHead(int headIdx) const { return headIdx / groupSize; }
};

template <typename DType_, int kHeadDim_>
struct FixLenParams : ParamsBase<DType_, kHeadDim_> {
  using DType = typename ParamsBase<DType_, kHeadDim_>::DType;
  static constexpr int kHeadDim = ParamsBase<DType_, kHeadDim_>::kHeadDim;
  using ParamsBase<DType_, kHeadDim_>::Q;
  using ParamsBase<DType_, kHeadDim_>::K;
  using ParamsBase<DType_, kHeadDim_>::V;
  using ParamsBase<DType_, kHeadDim_>::O;
  using ParamsBase<DType_, kHeadDim_>::seqDimQ;
  using ParamsBase<DType_, kHeadDim_>::seqDimKV;

  int seqLenQ;
  int seqLenKV;
  int numKVTiles;  // ceilDiv(seqLenKV, kTileKV)

  __device__ __forceinline__ int getSeqLenQ(int batchIdx) const { return seqLenQ; }
  __device__ __forceinline__ int getSeqLenKV(int batchIdx) const { return seqLenKV; }
  __device__ __forceinline__ int getKVTiles(int seqLen, int tileKV) const { return numKVTiles; }

  // Layout: [batch, seq, head, dim]
  template <typename Context>
  __device__ __forceinline__ const DType* qPtr(const Context& ctx) const {
    return Q + (ctx.batchIdx * seqLenQ + ctx.tileQ) * seqDimQ + ctx.headIdx * kHeadDim;
  }

  template <typename Context>
  __device__ __forceinline__ const DType* kPtr(const Context& ctx, int seqIdx) const {
    return K + (ctx.batchIdx * seqLenKV + seqIdx) * seqDimKV + ctx.headIdxKV * kHeadDim;
  }

  template <typename Context>
  __device__ __forceinline__ const DType* vPtr(const Context& ctx, int seqIdx) const {
    return V + (ctx.batchIdx * seqLenKV + seqIdx) * seqDimKV + ctx.headIdxKV * kHeadDim;
  }

  template <typename Context>
  __device__ __forceinline__ DType* oPtr(const Context& ctx) const {
    return O + (ctx.batchIdx * seqLenQ + ctx.tileQ) * seqDimQ + ctx.headIdx * kHeadDim;
  }
};

template <typename DType_, int kHeadDim_>
struct VarLenParamsBase : ParamsBase<DType_, kHeadDim_> {
  using DType = typename ParamsBase<DType_, kHeadDim_>::DType;
  static constexpr int kHeadDim = ParamsBase<DType_, kHeadDim_>::kHeadDim;
  using ParamsBase<DType_, kHeadDim_>::Q;
  using ParamsBase<DType_, kHeadDim_>::O;
  using ParamsBase<DType_, kHeadDim_>::seqDimQ;

  const int* __restrict__ cuSeqLensQ;   // [batch + 1], cumulative sequence lengths for Q
  const int* __restrict__ cuSeqLensKV;  // [batch + 1], cumulative sequence lengths for KV

  int maxSeqLenQ;
  int maxSeqLenKV;
  int maxKVTiles;  // ceilDiv(maxSeqLenKV, kTileKV)

  __device__ __forceinline__ int getSeqLenQ(int batchIdx) const {
    return cuSeqLensQ[batchIdx + 1] - cuSeqLensQ[batchIdx];
  }
  __device__ __forceinline__ int getSeqLenKV(int batchIdx) const {
    return cuSeqLensKV[batchIdx + 1] - cuSeqLensKV[batchIdx];
  }
  __device__ static __forceinline__ int getKVTiles(int seqLen, int tileKV) { return ceilDiv(seqLen, tileKV); }

  // Layout: [totalSeq, numHeads, headDim] (packed, no padding)
  template <typename Context>
  __device__ __forceinline__ const DType* qPtr(const Context& ctx) const {
    return Q + (cuSeqLensQ[ctx.batchIdx] + ctx.tileQ) * seqDimQ + ctx.headIdx * kHeadDim;
  }

  template <typename Context>
  __device__ __forceinline__ DType* oPtr(const Context& ctx) const {
    return O + (cuSeqLensQ[ctx.batchIdx] + ctx.tileQ) * seqDimQ + ctx.headIdx * kHeadDim;
  }
};

template <typename DType_, int kHeadDim_>
struct VarLenParams : VarLenParamsBase<DType_, kHeadDim_> {
  using DType = typename VarLenParamsBase<DType_, kHeadDim_>::DType;
  static constexpr int kHeadDim = VarLenParamsBase<DType_, kHeadDim_>::kHeadDim;
  using VarLenParamsBase<DType_, kHeadDim_>::K;
  using VarLenParamsBase<DType_, kHeadDim_>::V;
  using VarLenParamsBase<DType_, kHeadDim_>::seqDimKV;
  using VarLenParamsBase<DType_, kHeadDim_>::cuSeqLensKV;

  template <typename Context>
  __device__ __forceinline__ const DType* kPtr(const Context& ctx, int seqIdx) const {
    return K + (cuSeqLensKV[ctx.batchIdx] + seqIdx) * seqDimKV + ctx.headIdxKV * kHeadDim;
  }

  template <typename Context>
  __device__ __forceinline__ const DType* vPtr(const Context& ctx, int seqIdx) const {
    return V + (cuSeqLensKV[ctx.batchIdx] + seqIdx) * seqDimKV + ctx.headIdxKV * kHeadDim;
  }
};

template <typename Config>
struct ThreadInfo {
  static constexpr int kWarpSize = Config::kWarpSize;
  static constexpr int kRowsPerWarp = Config::kRowsPerWarp;
  static constexpr int kColsPerLane = Config::kBc / kWarpSize;
  static constexpr int kDimsPerLane = Config::kHeadDim / kWarpSize;

  const unsigned int threadId = threadIdx.x;
  static constexpr int blockSize = Config::kNumThreads;
  const int warpId = threadId / kWarpSize;
  const int laneId = threadId % kWarpSize;
  const int warpRowOffset = warpId * kRowsPerWarp;
};

template <typename Config, typename Params>
struct BlockInfo {
  static constexpr int kTileQ = Config::kBr;   // tile size for Q
  static constexpr int kTileKV = Config::kBc;  // tile size for KV

  // grid: (tileIdxQ, batch, head)
  const unsigned int batchIdx = blockIdx.y;
  unsigned int headIdx = blockIdx.z;
  const unsigned int tileIdxQ = blockIdx.x;
  const int tileQ = tileIdxQ * kTileQ;
  int headIdxKV;
  const int seqLenQ;  // sequence length for Q
  int seqLenKV;       // sequence length for KV
  const int tileSizeQ;
  const int causalQOffset;  // seqLenKV - seqLenQ

  __device__ explicit BlockInfo(const Params& params)
      : headIdxKV(params.getKVHead(headIdx)),
        seqLenQ(params.getSeqLenQ(batchIdx)),
        seqLenKV(params.getSeqLenKV(batchIdx)),
        tileSizeQ(max(0, min(kTileQ, params.getSeqLenQ(batchIdx) - tileQ))),
        causalQOffset(params.getSeqLenKV(batchIdx) - params.getSeqLenQ(batchIdx)) {}

  __device__ __forceinline__ bool isValidTile() const { return tileQ < seqLenQ; }

  template <bool kIsCausal>
  __device__ __forceinline__ int numTilesKV() const {
    int tiles = ceilDiv(seqLenKV, kTileKV);
    if constexpr (kIsCausal) {
      tiles = min(tiles, ceilDiv(causalQOffset + tileQ + tileSizeQ, kTileKV));
    }
    return tiles;
  }
};

template <typename DType_, int kHeadDim_>
struct PagedVarLenParams : VarLenParamsBase<DType_, kHeadDim_> {
  using DType = typename VarLenParamsBase<DType_, kHeadDim_>::DType;
  static constexpr int kHeadDim = VarLenParamsBase<DType_, kHeadDim_>::kHeadDim;
  using VarLenParamsBase<DType_, kHeadDim_>::Q;
  using VarLenParamsBase<DType_, kHeadDim_>::O;
  using VarLenParamsBase<DType_, kHeadDim_>::seqDimQ;
  using VarLenParamsBase<DType_, kHeadDim_>::seqDimKV;
  using VarLenParamsBase<DType_, kHeadDim_>::cuSeqLensQ;

  const DType* __restrict__ kCachePool;  // [numBlocks, numKvHeads, pageSize, headDim]
  const DType* __restrict__ vCachePool;  // [numBlocks, numKvHeads, pageSize, headDim]
  const int* __restrict__ blockTable;    // [batch, maxBlocksPerSeq]
  int blockTableStride;                  // = maxBlocksPerSeq
  int pageSize;                          // tokens per page (e.g. 16)
  int pageStride;                        // = numKvHeads * pageSize * headDim
  int kvHeadStride;                      // = pageSize * headDim

  template <typename Context>
  __device__ __forceinline__ const DType* kPagePtr(const Context& ctx, int pageIdx) const {
    int physBlock = blockTable[ctx.batchIdx * blockTableStride + pageIdx];
    return kCachePool + physBlock * pageStride + ctx.headIdxKV * kvHeadStride;
  }

  template <typename Context>
  __device__ __forceinline__ const DType* vPagePtr(const Context& ctx, int pageIdx) const {
    int physBlock = blockTable[ctx.batchIdx * blockTableStride + pageIdx];
    return vCachePool + physBlock * pageStride + ctx.headIdxKV * kvHeadStride;
  }

  // for mma
  float* __restrict__ tmpO;    // [totalQ, numHeadsQ, maxPartitions, headDim] or nullptr
  float* __restrict__ tmpLse;  // [totalQ, numHeadsQ, maxPartitions, 2] or nullptr
  int numPartitions;           // 0 or 1 = no split; >1 = split-KV
  int partitionSize;
  int totalQ;
};

template <typename Params>
struct isPagedParams_ : std::false_type {};

template <typename DType_, int kHeadDim_>
struct isPagedParams_<PagedVarLenParams<DType_, kHeadDim_>> : std::true_type {};

template <typename Params>
inline constexpr bool isPagedParams = isPagedParams_<Params>::value;

}  // namespace tfa
