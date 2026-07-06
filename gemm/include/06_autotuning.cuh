#pragma once
#include "common.h"

template<int BM,int BN,int BK,int TM,int TN>
__global__ void autotuning_kernel(int M, int N, int K,
                             float alpha,
                             const float* A,
                             const float* B,
                             float beta,
                             float* C)
{
    constexpr int threads   = (BM/TM)*(BN/TN);
    constexpr int As_loads  = (BM*BK)/(threads*4);   // 每线程搬几个 float4 (As)
    constexpr int Bs_loads  = (BN*BK)/(threads*4);   // 同理 Bs
    constexpr int col_threads = BN / TN;

    static_assert(BK % 4 == 0, "As float4 store crosses BK boundary");
    static_assert((BM*BK) % (threads*4) == 0, "As load not divisible by float4");
    static_assert((BN*BK) % (threads*4) == 0, "Bs load not divisible by float4");
    static_assert(BN % 4 == 0 && BM % 4 == 0, "float4 row-cross");
    
    __shared__ float As[BK][BM];
    __shared__ float Bs[BK][BN];
    
    int tid = threadIdx.x + threadIdx.y*blockDim.x;


    float acc[TM*TN] = {0};
    for(int bk=0;bk<K;bk+=BK)
    {

        for (int load = 0; load < As_loads; load++) {
            int idx = (tid + load*threads) * 4;   // 注意 *threads 步进，再 *4
            int as_r = idx / BK, as_c = idx % BK;
            int a_row = blockIdx.y * BM + as_r;
            int a_col = bk + as_c;
            float4 t = reinterpret_cast<const float4*>(&A[a_row*K + a_col])[0];
            As[as_c][as_r] = t.x;
            As[as_c+1][as_r] = t.y;
            As[as_c+2][as_r] = t.z;
            As[as_c+3][as_r] = t.w;
        }
        for (int load = 0; load < Bs_loads; load++) {
            int idx = (tid + load*threads) * 4;   // 注意 *threads 步进，再 *4
            int bs_r = idx / BN, bs_c = idx % BN;
            int b_row = bk + bs_r;
            int b_col = blockIdx.x * BN + bs_c;
            float4 t = reinterpret_cast<const float4*>(&B[b_row*N + b_col])[0];
            reinterpret_cast<float4*>(&Bs[bs_r][bs_c])[0] = t;
        }
            
        

        
        __syncthreads();
        for(int k=0;k<BK;k++)
        {
            float regA[TM];
            float regB[TN];
            int col_in_tile = (tid % col_threads) * TN;
            int row_in_tile = (tid / col_threads) * TM;
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
    int c_col_block = tid % col_threads;   // 列序号
    int c_row_block = tid / col_threads;   // 行序号


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


// 1. 先完整定义模板 launcher(不是声明!)
template<int BM,int BN,int BK,int TM,int TN>
void launch_at(int M,int N,int K,float alpha,
               const float*A,const float*B,float beta,float*C)
{
    constexpr int threads = (BM/TM)*(BN/TN);
    dim3 block(threads);
    dim3 grid(CEIL_DIV(N,BN), CEIL_DIV(M,BM));
    autotuning_kernel<BM,BN,BK,TM,TN><<<grid,block>>>(M,N,K,alpha,A,B,beta,C);
    CHECK_CUDA(cudaGetLastError());
}