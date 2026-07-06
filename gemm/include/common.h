#pragma once

#include <cuda_runtime.h>   // cudaError_t, cudaGetErrorString, cudaEvent_t 等
#include <cstdio>           // fprintf, printf(CUDA_CHECK 里报错要用)
#include <cstdlib>          // exit / abort(CUDA_CHECK 出错后终止)


#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))

#define CHECK_CUDA(x) do {                                      \
    cudaError_t err_ = (x);                                      \
    if (err_ != cudaSuccess) {                                   \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err_));   \
        exit(1);                                                \
    }                                                           \
} while (0)

#define CHECK_CUBLAS(call) do {                                              \
    cublasStatus_t status = (call);                                          \
    if (status != CUBLAS_STATUS_SUCCESS) {                                   \
        printf("cuBLAS error %s:%d: status = %d\n",                         \
               __FILE__, __LINE__, static_cast<int>(status));                \
        std::exit(1);                                                        \
    }                                                                        \
} while (0)

typedef void (*KernelFn)(int M, int N, int K, float alpha,
                         const float* A, const float* B,
                         float beta, float* C);

