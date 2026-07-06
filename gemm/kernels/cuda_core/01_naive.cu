#include "../../include/common.h"



__global__ void naive_kernel(int M, int N, int K,
                             float alpha,
                             const float* A,
                             const float* B,
                             float beta,
                             float* C)
{
    int row = threadIdx.y +blockDim.y*blockIdx.y;
    int col = threadIdx.x +blockDim.x*blockIdx.x;
    if(row<M&&col<N)
    {
        float temp=0;
        for(int k=0;k<K;k++)
        {
             temp+= A[row*K+k]*B[N*k+col];
        }
        C[row*N+col] = alpha*temp + beta*C[row*N+col];
    }
}


void launch_naive_kernel(int M, int N, int K,
                         float alpha,
                         const float* A,
                         const float* B,
                         float beta,
                         float* C)
{

    dim3 block(32,32,1);
    dim3 grid(CEIL_DIV(N, 32), CEIL_DIV(M, 32), 1);

    naive_kernel<<<grid, block>>>(M, N, K, alpha, A, B, beta, C);

    CHECK_CUDA(cudaGetLastError());
}

