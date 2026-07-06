#include "../../include/common.h"


__global__ void Dblocktiling_kernel(int M, int N, int K,
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
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];
    
    int tid = threadIdx.x + threadIdx.y*blockDim.x;


    float acc[TM*TN] = {0};
    for(int bk=0;bk<K;bk+=BK)
    {

        
        for(int load=0; load<4; load++){
            int idx = tid + load*128;        // 0~511 全覆盖
            int as_r = idx / BK, as_c = idx % BK;   // As: 64×8
            int bs_r = idx / BN, bs_c = idx % BN;
            // 算 global 地址、边界判断、填 As[as_r][as_c]
            int a_row = blockIdx.y * BM + as_r;
            int a_col = bk + as_c;
            int b_row = bk + bs_r;
            int b_col = blockIdx.x * BN + bs_c;
            As[as_r][as_c] = (a_row<M && a_col<K) ? A[a_row*K + a_col] : 0.0f;
            Bs[bs_r][bs_c] = (b_row<K && b_col<N) ? B[b_row*N + b_col] : 0.0f;

        }
        __syncthreads();
        for(int k=0;k<BK;k++)
        {
            float regA[TM];
            float regB[TN];
            int row_in_tile = (tid / 16) * TM;   // 这个 thread 负责的行起点
            int col_in_tile = (tid % 16) * TN;
            for(int i=0;i<TM;i++)
                regA[i] = As[row_in_tile + i][k];
            for(int i=0;i<TN;i++)
                regB[i] = Bs[k][col_in_tile + i];

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


void launch_2Dblocktiling_kernel(int M, int N, int K,
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

    Dblocktiling_kernel<<<grid, block>>>(M, N, K, alpha, A, B, beta, C);

    CHECK_CUDA(cudaGetLastError());
}
