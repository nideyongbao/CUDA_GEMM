#include "../../include/common.h"


__global__ void blocktiling_kernel(int M, int N, int K,
                             float alpha,
                             const float* A,
                             const float* B,
                             float beta,
                             float* C)
{
    constexpr int BM=64;
    constexpr int BK =8;
    constexpr int BN = 64;
    constexpr int TM = 8;
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];
    
    int tid = threadIdx.x + threadIdx.y*blockDim.x;


    float acc[TM] = {0};
    for(int bk=0;bk<K;bk+=BK)
    {
        int a_row = tid / BK + BM * blockIdx.y;
        int a_col = bk + tid % BK;
        int b_row = bk + tid / BN;
        int b_col = blockIdx.x * BN + tid % BN;

        As[tid / BK][tid % BK] = (a_row < M && a_col < K) ? A[a_row * K + a_col] : 0.0f;
        Bs[tid / BN][tid % BN] = (b_row < K && b_col < N) ? B[b_row * N + b_col] : 0.0f;
        __syncthreads();
        for(int k=0;k<BK;k++)
        {
            float b = Bs[k][tid%BN];
            for(int i=0;i<TM;i++)
            {
                acc[i]+=As[tid/BN*TM + i][k] * b;
            }
        }
        __syncthreads();
    }
    int c_col=tid%64;
    int c_row_block=tid/64;
    for (int i = 0; i < TM; i++) {
        int global_row = blockIdx.y * BM + c_row_block * TM + i;
        int global_col = blockIdx.x * BN + c_col;
        if (global_row < M && global_col < N) {
            C[global_row * N + global_col] = alpha * acc[i] + beta * C[global_row * N + global_col];
        }
    }
    
}


void launch_blocktiling_kernel(int M, int N, int K,
                         float alpha,
                         const float* A,
                         const float* B,
                         float beta,
                         float* C)
{

    constexpr int BM = 64;
    constexpr int BN = 64;

    dim3 block(64,8,1);
    dim3 grid(CEIL_DIV(N, BN), CEIL_DIV(M, BM), 1);

    blocktiling_kernel<<<grid, block>>>(M, N, K, alpha, A, B, beta, C);

    CHECK_CUDA(cudaGetLastError());
}
