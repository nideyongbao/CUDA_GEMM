#include "../../include/common.h"

__global__ void smem_kernel(int M, int N, int K,
                             float alpha,
                             const float* A,
                             const float* B,
                             float beta,
                             float* C)
{
    __shared__ float As[32][32];
    __shared__ float Bs[32][32];
    int row = threadIdx.y +blockDim.y*blockIdx.y;
    int col = threadIdx.x +blockDim.x*blockIdx.x;
    int ty = threadIdx.y;
    int tx = threadIdx.x;
    float temp=0;

    for(int bk=0;bk<K;bk+=32)
    {
        As[ty][tx] = A[row*K+bk+tx];
        Bs[ty][tx] = B[(bk+ty)*N+col];
        __syncthreads();
        for(int k=0;k<32;k++)
        {
            temp += As[ty][k] * Bs[k][tx];
        }
        __syncthreads();
    }
    C[row*N+col] = alpha * temp + beta * C[row*N+col];    
}


void launch_smem_kernel(int M, int N, int K,
                         float alpha,
                         const float* A,
                         const float* B,
                         float beta,
                         float* C)
{

    dim3 block(32,32,1);
    dim3 grid(CEIL_DIV(N, 32), CEIL_DIV(M, 32), 1);

    smem_kernel<<<grid, block>>>(M, N, K, alpha, A, B, beta, C);

    CHECK_CUDA(cudaGetLastError());
}


