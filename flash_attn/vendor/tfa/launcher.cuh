/*
 * TinyFA
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include "config.cuh"
#include "fma/kernel.cuh"
#include "mma/kernel.cuh"
#include "mma/paged.cuh"
#include "params.cuh"
#include "utils.cuh"

namespace tfa {

namespace detail {

template <typename Config>
size_t getSmemSize() {
  if constexpr (Config::kUseTensorCore) {
    return mma::SmemLayout<Config>::kSmemSize;
  } else {
    return fma::SmemSize<Config>::kSmemSize;
  }
}

template <typename Config, typename Params>
void launchKernel(const Params& params, int gridX, int numHeads, int batchSize, bool isCausal, cudaStream_t stream) {
  size_t smemSize = getSmemSize<Config>();

  dim3 grid(gridX, batchSize, numHeads);
  dim3 block(Config::kNumThreads);

  auto launch = [&](auto kernel) {
    if (!configureSmem(kernel, smemSize)) {
      return;
    }
    kernel<<<grid, block, smemSize, stream>>>(params);
  };

  if constexpr (Config::kUseTensorCore) {
    launch(isCausal ? mma::flashAttentionKernel<Config, true, Params>
                    : mma::flashAttentionKernel<Config, false, Params>);
  } else {
    launch(isCausal ? fma::flashAttentionKernel<Config, true, Params>
                    : fma::flashAttentionKernel<Config, false, Params>);
  }
}

template <typename Config, typename Params>
void launchPagedVarLenKernel(const Params& paramsIn, int gridX, int numHeads, int batchSize, bool isCausal,
                             cudaStream_t stream) {
  size_t smemSize = getSmemSize<Config>();

  // dynamically adjust partition count for SM saturation
  Params params = paramsIn;
  if (params.numPartitions > 1) {
    int numBlocksNoSplit = gridX * batchSize * numHeads;
    int numSMs = getNumSMs();
    if (numBlocksNoSplit >= numSMs) {
      params.numPartitions = 0;
    } else {
      // target: total blocks -> numSMs
      int desiredParts = ceilDiv(numSMs, numBlocksNoSplit);
      if (desiredParts > paramsIn.numPartitions) {
        desiredParts = paramsIn.numPartitions;
      }
      int maxPossible = ceilDiv(params.maxSeqLenKV, static_cast<int>(Config::kBc));
      if (desiredParts > maxPossible) {
        desiredParts = maxPossible;
      }
      if (desiredParts <= 1) {
        params.numPartitions = 0;
      } else {
        params.numPartitions = desiredParts;
        params.partitionSize = ceilDiv(params.maxSeqLenKV, desiredParts);
      }
    }
  }

  // check tmpO, tmpLse
  if (params.numPartitions > 1) {
    if (params.tmpO == nullptr || params.tmpLse == nullptr) {
      fprintf(stderr, "TinyFA: split kv (numPartitions=%d) requires non-null tmpO and tmpLse buffers.\n",
              params.numPartitions);
      abort();
    }
  }

  auto launch = [&](auto kernel, int gridZ) {
    dim3 grid(gridX, batchSize, gridZ);
    dim3 block(Config::kNumThreads);
    if (!configureSmem(kernel, smemSize)) {
      return;
    }
    kernel<<<grid, block, smemSize, stream>>>(params);
  };

  if constexpr (Config::kUseTensorCore) {
    int gridZ = (params.numPartitions > 1) ? (numHeads * params.numPartitions) : numHeads;

    if (isCausal) {
      launch(mma::flashAttentionPagedVarLenKernel<Config, true, Params>, gridZ);
    } else {
      launch(mma::flashAttentionPagedVarLenKernel<Config, false, Params>, gridZ);
    }

    // reduce
    if (params.numPartitions > 1) {
      using DType = typename Params::DType;
      static constexpr int kHeadDim = Config::kHeadDim;
      int totalQ = params.totalQ;
      int numHeadsQ = params.seqDimQ / kHeadDim;

      constexpr int kReduceThreads = 128;
      dim3 reduceGrid(ceilDiv(kHeadDim, kReduceThreads), totalQ, numHeadsQ);
      dim3 reduceBlock(kReduceThreads);

      mma::flashAttnPagedVarLenReduceKernel<DType, kHeadDim><<<reduceGrid, reduceBlock, 0, stream>>>(
          params.O, params.tmpO, params.tmpLse, params.cuSeqLensQ, params.seqDimQ, params.numPartitions,
          params.numPartitions, numHeadsQ);
    }
  } else {
    dim3 grid(gridX, batchSize, numHeads);
    dim3 block(Config::kNumThreads);
    auto launchFma = [&](auto kernel) {
      if (!configureSmem(kernel, smemSize)) {
        return;
      }
      kernel<<<grid, block, smemSize, stream>>>(params);
    };
    if (isCausal) {
      launchFma(fma::flashAttentionPagedVarLenKernel<Config, true, Params>);
    } else {
      launchFma(fma::flashAttentionPagedVarLenKernel<Config, false, Params>);
    }
  }
}

}  // namespace detail

namespace impl {

template <typename Params, typename DType>
void initParams(Params& params, const DType* Q, const DType* K, const DType* V, DType* O, int numHeadsQ,
                int numHeadsKV) {
  static constexpr int kHeadDim = Params::kHeadDim;
  params.Q = Q;
  params.K = K;
  params.V = V;
  params.O = O;
  params.seqDimQ = numHeadsQ * kHeadDim;
  params.seqDimKV = numHeadsKV * kHeadDim;
  params.groupSize = numHeadsQ / numHeadsKV;
}

template <typename ArchTag, typename DType, int kHeadDim, bool IsCausal>
void runFwd(const DType* Q, const DType* K, const DType* V, DType* O, int batchSize, int seqLenQ, int seqLenKV,
            int numHeadsQ, int numHeadsKV, cudaStream_t stream) {
  using Config = typename ConfigForArch<ArchTag, DType, kHeadDim, IsCausal>::Config;
  using Params = FixLenParams<DType, kHeadDim>;

  Params params;
  initParams(params, Q, K, V, O, numHeadsQ, numHeadsKV);
  params.seqLenQ = seqLenQ;
  params.seqLenKV = seqLenKV;
  params.numKVTiles = ceilDiv(seqLenKV, Config::kBc);

  detail::launchKernel<Config>(params, ceilDiv(seqLenQ, Config::kBr), numHeadsQ, batchSize, IsCausal, stream);
}

template <typename ArchTag, typename DType, int kHeadDim, bool IsCausal>
void runFwdVarLen(const DType* Q, const DType* K, const DType* V, DType* O, const int* cuSeqLensQ,
                  const int* cuSeqLensKV, int batchSize, int maxSeqLenQ, int maxSeqLenKV, int numHeadsQ, int numHeadsKV,
                  cudaStream_t stream) {
  using Config = typename ConfigForArch<ArchTag, DType, kHeadDim, IsCausal>::Config;
  using Params = VarLenParams<DType, kHeadDim>;

  Params params;
  initParams(params, Q, K, V, O, numHeadsQ, numHeadsKV);
  params.cuSeqLensQ = cuSeqLensQ;
  params.cuSeqLensKV = cuSeqLensKV;
  params.maxSeqLenQ = maxSeqLenQ;
  params.maxSeqLenKV = maxSeqLenKV;
  params.maxKVTiles = ceilDiv(maxSeqLenKV, Config::kBc);

  detail::launchKernel<Config>(params, ceilDiv(maxSeqLenQ, Config::kBr), numHeadsQ, batchSize, IsCausal, stream);
}

template <typename ArchTag, typename DType, int kHeadDim, bool IsCausal>
void runFwdPagedVarLen(const DType* Q, DType* O, const DType* kCachePool, const DType* vCachePool,
                       const int* cuSeqLensQ, const int* cuSeqLensKV, const int* blockTable, int batchSize,
                       int maxSeqLenQ, int maxSeqLenKV, int numHeadsQ, int numHeadsKV, int pageSize,
                       int maxBlocksPerSeq, float* tmpO, float* tmpLse, int partitionSize, int totalQ,
                       cudaStream_t stream) {
  using Config = typename ConfigForArch<ArchTag, DType, kHeadDim, IsCausal>::Config;
  using Params = PagedVarLenParams<DType, kHeadDim>;

  Params params;
  initParams<Params, DType>(params, Q, nullptr, nullptr, O, numHeadsQ, numHeadsKV);

  params.cuSeqLensQ = cuSeqLensQ;
  params.cuSeqLensKV = cuSeqLensKV;
  params.kCachePool = kCachePool;
  params.vCachePool = vCachePool;
  params.blockTable = blockTable;
  params.blockTableStride = maxBlocksPerSeq;
  params.pageSize = pageSize;
  params.pageStride = numHeadsKV * pageSize * kHeadDim;
  params.kvHeadStride = pageSize * kHeadDim;
  params.maxSeqLenQ = maxSeqLenQ;
  params.maxSeqLenKV = maxSeqLenKV;
  params.maxKVTiles = ceilDiv(maxSeqLenKV, Config::kBc);

  // split kv
  params.tmpO = tmpO;
  params.tmpLse = tmpLse;
  params.numPartitions = splitKvNumPartitions(maxSeqLenKV, partitionSize);
  params.partitionSize = partitionSize;
  params.totalQ = totalQ;

  detail::launchPagedVarLenKernel<Config>(params, ceilDiv(maxSeqLenQ, Config::kBr), numHeadsQ, batchSize, IsCausal,
                                          stream);
}

}  // namespace impl

}  // namespace tfa
