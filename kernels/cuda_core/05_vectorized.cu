#include "../../include/common.h"


__global__ void vectorized_kernel(int M, int N, int K,
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
    constexpr int TN = 4;
    __shared__ float As[BK][BM];
    __shared__ float Bs[BK][BN];
    
    int tid = threadIdx.x + threadIdx.y*blockDim.x;


    float acc[TM*TN] = {0};
    for(int bk=0;bk<K;bk+=BK)
    {

        
        
        int idx = tid*4;        // 0~511 全覆盖
        int as_r = idx / BK, as_c = idx % BK;   // As: 64×8
        int bs_r = idx / BN, bs_c = idx % BN;
        // 算 global 地址、边界判断、填 As[as_r][as_c]
        int a_row = blockIdx.y * BM + as_r;
        int a_col = bk + as_c;
        int b_row = bk + bs_r;
        int b_col = blockIdx.x * BN + bs_c;
        float4 t = reinterpret_cast<const float4*>(&A[a_row*K + a_col])[0];
        As[as_c][as_r] = t.x;
        As[as_c+1][as_r] = t.y;
        As[as_c+2][as_r] = t.z;
        As[as_c+3][as_r] = t.w;
        
        t = reinterpret_cast<const float4*>(&B[b_row*N + b_col])[0];
        reinterpret_cast<float4*>(&Bs[bs_r][bs_c])[0] = t;

        
        __syncthreads();
        for(int k=0;k<BK;k++)
        {
            float regA[TM];
            float regB[TN];
            int row_in_tile = (tid / 16) * TM;   // 这个 thread 负责的行起点
            int col_in_tile = (tid % 16) * TN;
            for(int i=0;i<TM;i+=4)
            {
                float4 tmp = reinterpret_cast<float4*>(&As[k][row_in_tile + i])[0];
                regA[i] = tmp.x;
                regA[i+1] = tmp.y;
                regA[i+2] = tmp.z;
                regA[i+3] = tmp.w;
            }
            for(int i=0;i<TN;i+=4)
            {
                float4 tmp = reinterpret_cast<float4*>(&Bs[k][col_in_tile + i])[0];
                regB[i] = tmp.x;
                regB[i+1] = tmp.y;
                regB[i+2] = tmp.z;
                regB[i+3] = tmp.w;
            }
            for(int i=0;i<TN;i++)
            {
                for(int j=0;j<TM;j++)
                {
                    acc[i*TM+j]+=regA[j] * regB[i];
                }
            }
        }
        __syncthreads();
    }
    int c_col_block=tid%16;
    int c_row_block=tid/16;


    for(int i=0;i<TN;i++)
    {
        for(int j=0;j<TM;j++)
        {
            int global_row = blockIdx.y * BM + c_row_block * TM + j;
            int global_col = blockIdx.x * BN + c_col_block*TN +i;
            if (global_row < M && global_col < N) {
                C[global_row * N + global_col] = alpha * acc[i*TM+j] + beta * C[global_row * N + global_col];
            }
        }
    }

}


void launch_vectorized_kernel(int M, int N, int K,
                         float alpha,
                         const float* A,
                         const float* B,
                         float beta,
                         float* C)
{

    constexpr int BM = 64;
    constexpr int BN = 64;

    dim3 block(16,8,1);
    dim3 grid(CEIL_DIV(N, BN), CEIL_DIV(M, BM), 1);

    vectorized_kernel<<<grid, block>>>(M, N, K, alpha, A, B, beta, C);

    CHECK_CUDA(cudaGetLastError());
}
