/*
 * TinyFA
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include "../params.cuh"
#include "../ptx.cuh"
#include "../utils.cuh"
#include "gemm.cuh"
#include "layout.cuh"
#include "memory.cuh"
#include "softmax.cuh"

namespace tfa::mma {

template <typename Config, bool kIsCausal, typename Params>
__device__ void flashAttnPagedVarLenMma(const Params& params, char* smemBuf) {
  using namespace cute;

  using Block = BlockInfo<Config, Params>;
  using Smem = SmemLayout<Config>;
  using Gemm = GemmConfig<Config>;
  using Mem = MemConfig<Config>;
  using ElemType = typename Smem::Element;

  static constexpr int kBr = Config::kBr;
  static constexpr int kBc = Config::kBc;
  static constexpr int kHeadDim = Config::kHeadDim;

  const unsigned int tidx = threadIdx.x;

  const bool useSplitKV = params.numPartitions > 1;
  const int numHeadsQ = params.seqDimQ / kHeadDim;

  Block blockInfo(params);
  int partIdx = 0;
  if (useSplitKV) {
    blockInfo.headIdx = blockIdx.z % numHeadsQ;
    blockInfo.headIdxKV = params.getKVHead(blockInfo.headIdx);
    partIdx = blockIdx.z / numHeadsQ;
  }
  if (!blockInfo.isValidTile()) {
    return;
  }

  int partStartKV = 0;
  int partEndKV = blockInfo.seqLenKV;
  if (useSplitKV) {
    partStartKV = partIdx * params.partitionSize;
    partEndKV = min(partStartKV + params.partitionSize, blockInfo.seqLenKV);
    if (partStartKV >= blockInfo.seqLenKV) {
      return;
    }
    blockInfo.seqLenKV = partEndKV;
  }
  const int partSeqLenKV = partEndKV - partStartKV;

  const auto* qBase = reinterpret_cast<const ElemType*>(params.qPtr(blockInfo));
  auto* oBase = reinterpret_cast<ElemType*>(params.oPtr(blockInfo));
  const int rowStrideQ = params.seqDimQ;

  //  smem tensors
  auto* smemQ = reinterpret_cast<ElemType*>(smemBuf);
  auto* smemK = smemQ + size(typename Smem::SmemLayoutQ{});
  auto* smemV = smemK + size(typename Smem::SmemLayoutKV{});

  auto sQ = make_tensor(make_smem_ptr(smemQ), typename Smem::SmemLayoutQ{});
  auto sK = make_tensor(make_smem_ptr(smemK), typename Smem::SmemLayoutKV{});
  auto sV = make_tensor(make_smem_ptr(smemV), typename Smem::SmemLayoutKV{});
  auto sVt = make_tensor(make_smem_ptr(smemV), typename Smem::SmemLayoutVTransposed{});
  auto sVtNoSwizzle = make_tensor(smemV, typename Smem::SmemLayoutVTransposedNoSwizzle{});

  //  gmem tensor
  auto mQ = make_tensor(make_gmem_ptr(qBase), make_shape(blockInfo.seqLenQ - blockInfo.tileQ, Int<kHeadDim>{}),
                        make_stride(rowStrideQ, _1{}));
  auto gQ = local_tile(mQ, Shape<Int<kBr>, Int<kHeadDim>>{}, make_coord(0, 0));

  auto mO = make_tensor(make_gmem_ptr(oBase), make_shape(blockInfo.seqLenQ - blockInfo.tileQ, Int<kHeadDim>{}),
                        make_stride(rowStrideQ, _1{}));
  auto gO = local_tile(mO, Shape<Int<kBr>, Int<kHeadDim>>{}, make_coord(0, 0));

  // gmem copy partitions
  typename Mem::GmemTiledCopyQKV gmemTiledCopyQKV;
  auto gmemThrCopyQKV = gmemTiledCopyQKV.get_thread_slice(tidx);

  auto tQgQ = gmemThrCopyQKV.partition_S(gQ);
  auto tQsQ = gmemThrCopyQKV.partition_D(sQ);

  auto cQ = make_identity_tensor(make_shape(size<0>(sQ), size<1>(sQ)));
  auto tQcQ = gmemThrCopyQKV.partition_S(cQ);

  // MMA partitions
  typename Gemm::TiledMma tiledMma;
  auto thrMma = tiledMma.get_thread_slice(tidx);

  auto accO = partition_fragment_C(tiledMma, Shape<Int<kBr>, Int<kHeadDim>>{});
  clear(accO);

  auto tSrQ = thrMma.partition_fragment_A(sQ);
  auto tSrK = thrMma.partition_fragment_B(sK);

  auto smemTiledCopyQ = make_tiled_copy_A(typename Gemm::SmemCopyAtom{}, tiledMma);
  auto smemThrCopyQ = smemTiledCopyQ.get_thread_slice(tidx);
  auto tSsQ = smemThrCopyQ.partition_S(sQ);
  auto tSrQCopy = smemThrCopyQ.retile_D(tSrQ);

  auto smemTiledCopyK = make_tiled_copy_B(typename Gemm::SmemCopyAtom{}, tiledMma);
  auto smemThrCopyK = smemTiledCopyK.get_thread_slice(tidx);
  auto tSsK = smemThrCopyK.partition_S(sK);
  auto tSrKCopy = smemThrCopyK.retile_D(tSrK);

  auto tOrVt = thrMma.partition_fragment_B(sVtNoSwizzle);

  auto smemTiledCopyV = make_tiled_copy_B(typename Gemm::SmemCopyAtomTransposed{}, tiledMma);
  auto smemThrCopyV = smemTiledCopyV.get_thread_slice(tidx);
  auto tOsVt = smemThrCopyV.partition_S(sVt);
  auto tOrVtCopy = smemThrCopyV.retile_D(tOrVt);

  constexpr int kNRows = 2 * size<1>(accO);
  Softmax<Config, kNRows> softmax;
  softmax.init();

  constexpr float kAttnScale = AttentionScale<kHeadDim>::value;

  // load Q tile
  MemLoader::gm2sm(gmemTiledCopyQKV, tQsQ, tQgQ, tQcQ, blockInfo.tileSizeQ);
  cp_async_fence();

  // page pointer scratch for precomputed K/V page base pointers
  __shared__ uintptr_t pagePtrScratchBuf[PagedMemLoader::kMaxPagesPerTile * 2];
  void* pagePtrScratch = pagePtrScratchBuf;

  int numTilesKV = ceilDiv(partSeqLenKV, kBc);
  if constexpr (kIsCausal) {
    numTilesKV =
        min(numTilesKV, ceilDiv(blockInfo.causalQOffset + blockInfo.tileQ + blockInfo.tileSizeQ - partStartKV, kBc));
  }
  if (numTilesKV <= 0) {
    return;
  }
  int kvTileIdx = numTilesKV - 1;

  // load first K tile [tileIdx]
  int kvStart = partStartKV + kvTileIdx * kBc;
  int kvValidRows = min(kBc, partEndKV - kvStart);
  PagedMemLoader::precomputePagePtrs<Config>(pagePtrScratch, params, blockInfo, kvStart, kvValidRows, tidx);
  __syncthreads();
  PagedMemLoader::gm2smPagedCached<Config>(smemK, pagePtrScratch, params, kvStart, kvValidRows, true, tidx);
  cp_async_fence();
  cpAsyncWaitGroup<0>();
  __syncthreads();

  const int nMaskingSteps = [&]() -> int {
    if constexpr (!kIsCausal) {
      return ((partEndKV - partStartKV) % kBc == 0) ? 0 : 1;
    } else {
      bool isAlignedKV = (partSeqLenKV % kBc == 0);
      int numMasking = ceilDiv(kBr, kBc) + (isAlignedKV ? 0 : 1);
      return min(numMasking, numTilesKV);
    }
  }();
  const int nNoMaskEnd = numTilesKV - nMaskingSteps;

  auto tileIteration = [&](int tileIdx, auto kNeedsMaskTag) {
    constexpr bool kNeedsMask = kNeedsMaskTag.value;

    int vStart = partStartKV + tileIdx * kBc;
    int vValidRows = min(kBc, partEndKV - vStart);
    PagedMemLoader::precomputePagePtrs<Config>(pagePtrScratch, params, blockInfo, vStart, vValidRows, tidx);
    __syncthreads();

    // async load V[tileIdx]
    PagedMemLoader::gm2smPagedCached<Config>(smemV, pagePtrScratch, params, vStart, vValidRows, false, tidx);
    cp_async_fence();

    // S = Q @ K^T
    auto accS = partition_fragment_C(tiledMma, Shape<Int<kBr>, Int<kBc>>{});
    clear(accS);
    GemmOp::computeScore(accS, tiledMma, smemTiledCopyQ, smemTiledCopyK, tSsQ, tSsK, tSrQCopy, tSrKCopy, tSrQ, tSrK);

    // wait for V load
    cpAsyncWaitGroup<0>();
    __syncthreads();

    // async load next K [tileIdx - 1]
    bool hasNext = tileIdx > 0;
    if (hasNext) {
      int nextKVStart = partStartKV + (tileIdx - 1) * kBc;
      int nextValidRows = min(kBc, partEndKV - nextKVStart);
      PagedMemLoader::precomputePagePtrs<Config>(pagePtrScratch, params, blockInfo, nextKVStart, nextValidRows, tidx);
      __syncthreads();
      PagedMemLoader::gm2smPagedCached<Config>(smemK, pagePtrScratch, params, nextKVStart, nextValidRows, true, tidx);
      cp_async_fence();
    }

    // masking
    if constexpr (kNeedsMask) {
      int colOffset = partStartKV + tileIdx * kBc;
      Softmax<Config, kNRows>::template applyMask<kIsCausal>(accS, tiledMma, tidx, blockInfo, 0, colOffset);
    }

    // softmax
    softmax.update(accS, accO, kAttnScale);

    // O += P @ V
    GemmOp::computeOutput<Config>(accO, accS, tiledMma, smemTiledCopyV, tOsVt, tOrVtCopy, tOrVt);

    // wait for next K load
    if (hasNext) {
      cpAsyncWaitGroup<0>();
      __syncthreads();
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

  if (!useSplitKV) {
    // finalize softmax
    softmax.finalize(accO);

    // store O
    MemStore::storeO<Config>(accO, smemQ, tiledMma, tidx, gO, blockInfo);
  } else {
#pragma unroll
    for (int mi = 0; mi < kNRows; mi++) {
      softmax.rowSum[mi] = warpReduceSum<4>(softmax.rowSum[mi]);
    }

    auto accORowCol = make_tensor(accO.data(), convertAccRowCol(accO.layout()));

    auto cO = make_identity_tensor(Shape<Int<kBr>, Int<kHeadDim>>{});
    auto tOcO = thrMma.partition_C(cO);
    auto tOcORowCol = make_tensor(tOcO.data(), convertAccRowCol(tOcO.layout()));

    const int cuSeqStart = params.cuSeqLensQ[blockInfo.batchIdx];
    const int maxPartitions = params.numPartitions;
    const int numHeadsQ_ = params.seqDimQ / kHeadDim;

    // vectorized output via smem staging
    static_assert(sizeof(ElemType) * (kBr + 2 * kBc) * kHeadDim >= sizeof(float) * kBr * kHeadDim,
                  "smem buffer too small for float staging; need (kBr + 2*kBc)*sizeof(ElemType) >= kBr*sizeof(float)");
    auto* smemStage = reinterpret_cast<float*>(smemBuf);

    // scatters accO elements into the staging buffer
#pragma unroll
    for (int mi = 0; mi < size<0>(accORowCol); mi++) {
      int localRow = get<0>(tOcORowCol(mi, 0));
#pragma unroll
      for (int ni = 0; ni < size<1>(accORowCol); ni++) {
        int d = get<1>(tOcORowCol(mi, ni));
        smemStage[localRow * kHeadDim + d] = accORowCol(mi, ni);
      }
    }
    __syncthreads();

    // write tmpO
    static constexpr int kFloatsPerVec = 4;  // float4 = 128 bits
    static constexpr int kVecsPerRow = kHeadDim / kFloatsPerVec;
    static constexpr int kNumThreads = Config::kNumThreads;

    for (unsigned int idx = tidx; idx < kBr * kVecsPerRow; idx += kNumThreads) {
      int row = idx / kVecsPerRow;
      int vecIdx = idx % kVecsPerRow;

      int globalQRow = blockInfo.tileQ + row;
      if (globalQRow < blockInfo.seqLenQ) {
        int qAbsIdx = cuSeqStart + globalQRow;
        float* oPtr = params.tmpO + ((qAbsIdx * numHeadsQ_ + blockInfo.headIdx) * maxPartitions + partIdx) * kHeadDim;
        int col = vecIdx * kFloatsPerVec;

        *reinterpret_cast<float4*>(oPtr + col) = *reinterpret_cast<const float4*>(smemStage + row * kHeadDim + col);
      }
    }

    // write tmpLse
#pragma unroll
    for (int mi = 0; mi < size<0>(accORowCol); mi++) {
      int globalQRow = blockInfo.tileQ + get<0>(tOcORowCol(mi, 0));
      if (globalQRow < blockInfo.seqLenQ && get<1>(tOcORowCol(mi, 0)) == 0) {
        int qAbsIdx = cuSeqStart + globalQRow;
        float* lsePtr = params.tmpLse + ((qAbsIdx * numHeadsQ_ + blockInfo.headIdx) * maxPartitions + partIdx) * 2;
        lsePtr[0] = softmax.rowMax[mi];
        lsePtr[1] = softmax.rowSum[mi];
      }
    }
  }
}

TFA_DEFINE_KERNEL_WRAPPER(flashAttentionPagedVarLenKernel, flashAttnPagedVarLenMma)

template <typename DType, int kHeadDim, int kMaxPartitions = 128>
__global__ void flashAttnPagedVarLenReduceKernel(DType* __restrict__ O, float* __restrict__ tmpO,
                                                 float* __restrict__ tmpLse, const int* __restrict__ cuSeqLensQ,
                                                 int seqDimQ, int numPartitions, int maxPartitions, int numHeadsQ) {
  // grid: (ceilDiv(headDim, blockDim.x), totalQ, numHeadsQ)
  const unsigned int d = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int qIdx = blockIdx.y;
  const unsigned int headIdx = blockIdx.z;

  constexpr float kAttnScale = AttentionScale<kHeadDim>::value;

  const unsigned int baseIdx = (qIdx * numHeadsQ + headIdx) * maxPartitions;

  __shared__ float sharedMax[kMaxPartitions];    // rowMax per partition
  __shared__ float sharedSum[kMaxPartitions];    // rowSum per partition
  __shared__ float sharedScale[kMaxPartitions];  // exp2 rescale factor

  if (numPartitions > kMaxPartitions) {
    return;
  }

  // load tmpLse
  for (unsigned int p = threadIdx.x; p < numPartitions; p += blockDim.x) {
    sharedMax[p] = tmpLse[(baseIdx + p) * 2 + 0];
    sharedSum[p] = tmpLse[(baseIdx + p) * 2 + 1];
  }
  __syncthreads();

  // global max reduction
  float localMax = -INFINITY;
  for (unsigned int p = threadIdx.x; p < numPartitions; p += blockDim.x) {
    localMax = max(localMax, sharedMax[p]);
  }
  // warp-level max reduce
  localMax = warpReduceMax(localMax);

  // block-level reduce
  constexpr int kMaxWarps = 4;  // blockDim.x=128 → 4 warps
  __shared__ float warpMaxBuf[kMaxWarps];
  const unsigned int warpId = threadIdx.x / 32;
  const unsigned int laneId = threadIdx.x % 32;
  if (laneId == 0) {
    warpMaxBuf[warpId] = localMax;
  }
  __syncthreads();
  float globalMax = -INFINITY;
  if (threadIdx.x < kMaxWarps) {
    globalMax = warpMaxBuf[threadIdx.x];
  }
  globalMax = warpReduceMax<kMaxWarps>(globalMax);
  if (threadIdx.x == 0) {
    warpMaxBuf[0] = globalMax;
  }
  __syncthreads();
  globalMax = warpMaxBuf[0];

  // per-partition scale computation
  for (unsigned int p = threadIdx.x; p < numPartitions; p += blockDim.x) {
    sharedScale[p] = (sharedMax[p] == -INFINITY) ? 0.0f : exp2f((sharedMax[p] - globalMax) * kAttnScale);
  }
  __syncthreads();

  if (d >= kHeadDim) {
    return;
  }

  float sumO = 0.0f;
  float sumExp = 0.0f;
  for (int p = 0; p < numPartitions; p++) {
    float partO = tmpO[(baseIdx + p) * kHeadDim + d];
    float scale = sharedScale[p];
    sumO += partO * scale;
    sumExp += sharedSum[p] * scale;
  }

  // normalize & write O
  float invSum = (sumExp > 0.0f) ? (1.0f / sumExp) : 0.0f;
  O[qIdx * seqDimQ + headIdx * kHeadDim + d] = fromFloat<DType>(sumO * invSum);
}

}  // namespace tfa::mma
