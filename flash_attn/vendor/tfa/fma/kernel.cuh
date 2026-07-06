/*
 * TinyFA
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include "../params.cuh"
#include "../utils.cuh"
#include "gemm.cuh"
#include "memory.cuh"
#include "softmax.cuh"
#include "tile.cuh"

namespace tfa::fma {

template <typename Config, typename Params>
struct KernelContext : ThreadInfo<Config>, BlockInfo<Config, Params> {
  using DType = typename Config::DType;
  using Thread = ThreadInfo<Config>;
  using Block = BlockInfo<Config, Params>;

  static constexpr float kAttnScale = AttentionScale<Config::kHeadDim>::value;

  using Block::batchIdx;
  using Block::causalQOffset;
  using Block::headIdx;
  using Block::headIdxKV;
  using Block::isValidTile;
  using Block::kTileKV;
  using Block::numTilesKV;
  using Block::seqLenKV;
  using Block::seqLenQ;
  using Block::tileQ;
  using Block::tileSizeQ;

  using Thread::blockSize;
  using Thread::kRowsPerWarp;
  using Thread::laneId;
  using Thread::threadId;
  using Thread::warpRowOffset;

  __device__ explicit KernelContext(const Params& params) : Thread(), Block(params), params(params) {}

  int tileKV = 0;
  int tileSizeKV = 0;

  template <bool kIsCausal>
  __device__ __forceinline__ void setTileKV(int tileIdx) {
    tileKV = tileIdx * kTileKV;
    tileSizeKV = min(kTileKV, seqLenKV - tileKV);
    if constexpr (kIsCausal) {
      tileSizeKV = min(tileSizeKV, causalQOffset + tileQ + tileSizeQ - tileKV);
    }
  }

  template <bool kIsCausal>
  __device__ __forceinline__ bool needsCausalMask() const {
    return kIsCausal && (tileKV + kTileKV > causalQOffset + tileQ);
  }

  __device__ __forceinline__ bool isPartialTile() const { return tileSizeKV < kTileKV; }

  __device__ __forceinline__ int validWarpRows() const {
    int warpStartQ = tileQ + warpRowOffset;
    return (warpStartQ < seqLenQ) ? min(kRowsPerWarp, seqLenQ - warpStartQ) : 0;
  }

  __device__ __forceinline__ int globalRowQ(int localRow) const { return tileQ + warpRowOffset + localRow; }

  __device__ __forceinline__ int globalKV(int n) const { return tileKV + laneId + n * Config::kWarpSize; }

  __device__ __forceinline__ const DType* qPtr() const { return params.qPtr(*this); }
  __device__ __forceinline__ const DType* kPtr() const { return params.kPtr(*this, tileKV); }
  __device__ __forceinline__ const DType* vPtr() const { return params.vPtr(*this, tileKV); }
  __device__ __forceinline__ DType* oPtr() const { return params.oPtr(*this); }
  __device__ __forceinline__ int seqDimQ() const { return params.seqDimQ; }
  __device__ __forceinline__ int seqDimKV() const { return params.seqDimKV; }

  const Params& params;
};

template <typename Config, bool kUseCpAsync, bool kIsPaged>
struct KVLoadOps {
  using DType = typename Config::DType;

  template <typename Tile, typename Context, typename Params>
  __device__ __forceinline__ static void loadK(Tile& tile, const Params& params, Context& ctx, int tileVStart) {
    if constexpr (kUseCpAsync) {
      tile.gm2smAsync(ctx.kPtr(), ctx.seqDimKV(), ctx.tileSizeKV, ctx);
    } else {
      tile.gm2sm(ctx.kPtr(), ctx.seqDimKV(), ctx.tileSizeKV, ctx);
    }
  }

  template <typename Tile, typename Context, typename Params>
  __device__ __forceinline__ static void loadV(Tile& tile, const Params& params, Context& ctx, int tileVStart) {
    if constexpr (kUseCpAsync) {
      tile.gm2smAsync(ctx.vPtr(), ctx.seqDimKV(), ctx.tileSizeKV, ctx);
    } else {
      tile.gm2sm(ctx.vPtr(), ctx.seqDimKV(), ctx.tileSizeKV, ctx);
    }
  }
};

// paged KV load
template <typename Config, bool kUseCpAsync>
struct KVLoadOps<Config, kUseCpAsync, true> {
  using DType = typename Config::DType;

  template <typename Tile, typename Context, typename Params>
  __device__ __forceinline__ static void loadK(Tile& tile, const Params& params, Context& ctx, int tileKVStart) {
    if constexpr (kUseCpAsync) {
      MemLoaderPagedVarLen<Config>::template gm2smPagedVarLenAsync<Config::kBc>(tile.smem, params, ctx, tileKVStart,
                                                                                ctx.tileSizeKV, true);
    } else {
      MemLoaderPagedVarLen<Config>::template gm2smPagedVarLen<Config::kBc>(tile.smem, params, ctx, tileKVStart,
                                                                           ctx.tileSizeKV, true);
    }
  }

  template <typename Tile, typename Context, typename Params>
  __device__ __forceinline__ static void loadV(Tile& tile, const Params& params, Context& ctx, int tileVStart) {
    if constexpr (kUseCpAsync) {
      MemLoaderPagedVarLen<Config>::template gm2smPagedVarLenAsync<Config::kBc>(tile.smem, params, ctx, tileVStart,
                                                                                ctx.tileSizeKV, false);
    } else {
      MemLoaderPagedVarLen<Config>::template gm2smPagedVarLen<Config::kBc>(tile.smem, params, ctx, tileVStart,
                                                                           ctx.tileSizeKV, false);
    }
  }
};

template <typename Config, bool kIsCausal, typename Params>
__device__ void flashAttnFma(const Params& params, char* smemBuf) {
  using DType = typename Config::DType;
  using Context = KernelContext<Config, Params>;

  static constexpr bool kUseCpAsync = Config::kUseCpAsync;
  static constexpr bool kIsPaged = isPagedParams<Params>;
  static constexpr int kRowsPerWarp = Config::kRowsPerWarp;
  static constexpr int kColsPerLane = Config::kBc / Config::kWarpSize;
  static constexpr int kDimsPerLane = Config::kHeadDim / Config::kWarpSize;

  using KVLoad = KVLoadOps<Config, kUseCpAsync, kIsPaged>;

  Context ctx(params);
  if (!ctx.isValidTile()) {
    return;
  }

  // smem:
  //   async: Q | K | V
  //   sync:  Q | KV
  auto* smem = reinterpret_cast<DType*>(smemBuf);
  QTile<Config> qTile(smem);
  DType* kSmemBase = smem + qTile.numElems();
  KTile<Config> kTile(kSmemBase);
  VTile<Config> vTile(kUseCpAsync ? (kSmemBase + KTile<Config>::numElems()) : kSmemBase);

  float accO[kRowsPerWarp][kDimsPerLane]{};

  Softmax<Config> softmax;
  softmax.init();

  // load Q tile
  qTile.gm2sm(ctx.qPtr(), ctx.seqDimQ(), ctx.tileSizeQ, ctx);
  __syncthreads();

  const int numTilesKV = ctx.template numTilesKV<kIsCausal>();
  if (numTilesKV <= 0) {
    return;
  }
  int kvTileIdx = numTilesKV - 1;

  const int nMaskingSteps = Softmax<Config>::template numMaskingSteps<kIsCausal>(ctx);
  const int nNoMaskEnd = numTilesKV - nMaskingSteps;

  // load first K tile [tileIdx]
  ctx.template setTileKV<kIsCausal>(kvTileIdx);
  KVLoad::loadK(kTile, params, ctx, kvTileIdx * Config::kBc);
  if constexpr (kUseCpAsync) {
    cpAsyncCommit();
    cpAsyncWaitAll();
  }
  __syncthreads();

  // kv tile iteration
  auto tileIteration = [&](int tileIdx, auto kNeedsMaskTag) {
    constexpr bool kNeedsMask = kNeedsMaskTag.value;

    if constexpr (kUseCpAsync) {
      // async load V[tileIdx]
      KVLoad::loadV(vTile, params, ctx, tileIdx * Config::kBc);
      cpAsyncCommit();
    }

    // S = Q @ K^T
    float accS[kRowsPerWarp][kColsPerLane]{};
    GemmOp<Config>::computeScore(accS, qTile, kTile, ctx);

    if constexpr (kUseCpAsync) {
      // wait for V load
      cpAsyncWaitAll();
      __syncthreads();
    } else {
      // sync load V[tileIdx]
      __syncthreads();
      KVLoad::loadV(vTile, params, ctx, tileIdx * Config::kBc);
      __syncthreads();
    }

    int curTileKV = ctx.tileKV;
    int tileSizeKV = ctx.tileSizeKV;

    // load next K [tileIdx - 1] (begin)
    bool hasNext = tileIdx > 0;
    if (hasNext) {
      ctx.template setTileKV<kIsCausal>(tileIdx - 1);
      if constexpr (kUseCpAsync) {
        KVLoad::loadK(kTile, params, ctx, (tileIdx - 1) * Config::kBc);
        cpAsyncCommit();
      }
    }

    // masking
    if constexpr (kNeedsMask) {
      Softmax<Config>::template applyMask<kIsCausal>(accS, ctx, curTileKV, tileSizeKV);
    }

    // softmax
    softmax.update(accS, accO, ctx.kAttnScale);

    // O += P @ V
    GemmOp<Config>::computeOutput(accO, accS, vTile, ctx);

    // load next K [tileIdx - 1] (finish)
    if (hasNext) {
      if constexpr (kUseCpAsync) {
        cpAsyncWaitAll();
        __syncthreads();
      } else {
        KVLoad::loadK(kTile, params, ctx, (tileIdx - 1) * Config::kBc);
        __syncthreads();
      }
    }
  };

  // masking iterations
  for (; kvTileIdx >= nNoMaskEnd; --kvTileIdx) {
    tileIteration(kvTileIdx, std::integral_constant<bool, true>{});
  }
  // no masking iterations
  for (; kvTileIdx >= 0; --kvTileIdx) {
    tileIteration(kvTileIdx, std::integral_constant<bool, false>{});
  }

  // finalize softmax
  softmax.finalize(accO);

  // store O
  MemStore<Config>::storeO(ctx.oPtr(), smem, ctx.seqDimQ(), accO, ctx);
}

TFA_DEFINE_KERNEL_WRAPPER(flashAttentionKernel, flashAttnFma)
TFA_DEFINE_KERNEL_WRAPPER(flashAttentionPagedVarLenKernel, flashAttnFma)

}  // namespace tfa::fma
