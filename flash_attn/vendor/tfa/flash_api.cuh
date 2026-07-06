/*
 * TinyFA
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include "config.cuh"
#include "utils.cuh"

namespace tfa {

namespace impl {

template <typename ArchTag, typename DType, int kHeadDim, bool IsCausal>
void runFwd(const DType* Q, const DType* K, const DType* V, DType* O, int batchSize, int seqLenQ, int seqLenKV,
            int numHeadsQ, int numHeadsKV, cudaStream_t stream);

template <typename ArchTag, typename DType, int kHeadDim, bool IsCausal>
void runFwdVarLen(const DType* Q, const DType* K, const DType* V, DType* O, const int* cuSeqLensQ,
                  const int* cuSeqLensKV, int batchSize, int maxSeqLenQ, int maxSeqLenKV, int numHeadsQ, int numHeadsKV,
                  cudaStream_t stream);

template <typename ArchTag, typename DType, int kHeadDim, bool IsCausal>
void runFwdPagedVarLen(const DType* Q, DType* O, const DType* kCachePool, const DType* vCachePool,
                       const int* cuSeqLensQ, const int* cuSeqLensKV, const int* blockTable, int batchSize,
                       int maxSeqLenQ, int maxSeqLenKV, int numHeadsQ, int numHeadsKV, int pageSize,
                       int maxBlocksPerSeq, float* tmpO, float* tmpLse, int partitionSize, int totalQ,
                       cudaStream_t stream);

}  // namespace impl

#define TFA_DISPATCH_KERNEL(headDim, isCausal, ...) \
  do {                                              \
    const int arch_ = getRuntimeArch();             \
    TFA_DISPATCH_ARCH(arch_, {                      \
      TFA_DISPATCH_HEAD_DIM(headDim, kHeadDim, {    \
        if (isCausal) {                             \
          constexpr bool IsCausal = true;           \
          __VA_ARGS__                               \
        } else {                                    \
          constexpr bool IsCausal = false;          \
          __VA_ARGS__                               \
        }                                           \
      });                                           \
    });                                             \
  } while (0)

template <typename DType>
void flashAttn(const DType* Q, const DType* K, const DType* V, DType* O, int batchSize, int seqLenQ, int seqLenKV,
               int numHeadsQ, int numHeadsKV, int headDim, bool isCausal = false, cudaStream_t stream = nullptr) {
  TFA_DISPATCH_KERNEL(headDim, isCausal, {
    impl::runFwd<ArchTag, DType, kHeadDim, IsCausal>(Q, K, V, O, batchSize, seqLenQ, seqLenKV, numHeadsQ, numHeadsKV,
                                                     stream);
  });
}

template <typename DType>
void flashAttnVarLen(const DType* Q, const DType* K, const DType* V, DType* O, const int* cuSeqLensQ,
                     const int* cuSeqLensKV, int batchSize, int maxSeqLenQ, int maxSeqLenKV, int numHeadsQ,
                     int numHeadsKV, int headDim, bool isCausal = false, cudaStream_t stream = nullptr) {
  TFA_DISPATCH_KERNEL(headDim, isCausal, {
    impl::runFwdVarLen<ArchTag, DType, kHeadDim, IsCausal>(Q, K, V, O, cuSeqLensQ, cuSeqLensKV, batchSize, maxSeqLenQ,
                                                           maxSeqLenKV, numHeadsQ, numHeadsKV, stream);
  });
}

template <typename DType>
void flashAttnPagedVarLen(const DType* Q, DType* O, const DType* kCachePool, const DType* vCachePool,
                          const int* cuSeqLensQ, const int* cuSeqLensKV, const int* blockTable, int batchSize,
                          int maxSeqLenQ, int maxSeqLenKV, int numHeadsQ, int numHeadsKV, int headDim, int pageSize,
                          int maxBlocksPerSeq, int totalQ, bool isCausal = false, float* tmpO = nullptr,
                          float* tmpLse = nullptr, int partitionSize = 0, cudaStream_t stream = nullptr) {
  TFA_DISPATCH_KERNEL(headDim, isCausal, {
    impl::runFwdPagedVarLen<ArchTag, DType, kHeadDim, IsCausal>(
        Q, O, kCachePool, vCachePool, cuSeqLensQ, cuSeqLensKV, blockTable, batchSize, maxSeqLenQ, maxSeqLenKV,
        numHeadsQ, numHeadsKV, pageSize, maxBlocksPerSeq, tmpO, tmpLse, partitionSize, totalQ, stream);
  });
}

inline size_t splitKvTmpOSize(int totalQ, int numHeadsQ, int numPartitions, int headDim) {
  return static_cast<size_t>(totalQ) * numHeadsQ * numPartitions * headDim;
}

inline size_t splitKvTmpLseSize(int totalQ, int numHeadsQ, int numPartitions) {
  return static_cast<size_t>(totalQ) * numHeadsQ * numPartitions * 2;
}

}  // namespace tfa
