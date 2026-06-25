#pragma once
// BF16 版 GEMM：A/B 以 __nv_bfloat16 存于 global memory，载入时转 FP32，
// shared memory / 寄存器 / 累加 全程 FP32，C 输出 FP32。
// 目的：在 H20 上测“BF16 输入 + FP32 累加”的 CUDA core kernel 对 BF16
// Tensor Core 峰值(148 TFLOPS)的利用率，并与真正的 WMMA / cuBLAS BF16 对比。
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include "common.h"

// 统一的 BF16 kernel launcher 签名
typedef void (*BF16KernelFn)(int M, int N, int K, float alpha,
                             const __nv_bfloat16* A, const __nv_bfloat16* B,
                             float beta, float* C);

// 读 4 个连续 bf16(8 字节，需 8 字节对齐)→ 4 个 float。
__device__ __forceinline__ void ld4_bf16(const __nv_bfloat16* p,
                                         float& a, float& b, float& c, float& d) {
    float2 raw = *reinterpret_cast<const float2*>(p);          // 8B = 4×bf16
    const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&raw);
    float2 f0 = __bfloat1622float2(h[0]);
    float2 f1 = __bfloat1622float2(h[1]);
    a = f0.x; b = f0.y; c = f1.x; d = f1.y;
}

// ---- 非模板 kernel 的 launcher 声明 ----
void launch_naive_bf16(int,int,int,float,const __nv_bfloat16*,const __nv_bfloat16*,float,float*);
void launch_smem_bf16(int,int,int,float,const __nv_bfloat16*,const __nv_bfloat16*,float,float*);
void launch_blocktiling_bf16(int,int,int,float,const __nv_bfloat16*,const __nv_bfloat16*,float,float*);
void launch_2Dblocktiling_bf16(int,int,int,float,const __nv_bfloat16*,const __nv_bfloat16*,float,float*);
void launch_vectorized_bf16(int,int,int,float,const __nv_bfloat16*,const __nv_bfloat16*,float,float*);
void launch_warptile_bf16(int,int,int,float,const __nv_bfloat16*,const __nv_bfloat16*,float,float*);
void launch_warptile_vec_bf16(int,int,int,float,const __nv_bfloat16*,const __nv_bfloat16*,float,float*);
void launch_bank_conflict_bf16(int,int,int,float,const __nv_bfloat16*,const __nv_bfloat16*,float,float*);
void launch_double_buffer_bf16(int,int,int,float,const __nv_bfloat16*,const __nv_bfloat16*,float,float*);
// 注：tensor core (WMMA / WGMMA / FP8) 不在本头文件，已拆成 kernels/tensor_core/ 下
// 一个个独立最简用例（自带 main），见 docs/ 对应文档。

// ---- autotuning 模板(BF16) ----
template<int BM,int BN,int BK,int TM,int TN>
__global__ void autotuning_kernel_bf16(int M, int N, int K, float alpha,
                                       const __nv_bfloat16* A, const __nv_bfloat16* B,
                                       float beta, float* C)
{
    constexpr int threads   = (BM/TM)*(BN/TN);
    constexpr int As_loads  = (BM*BK)/(threads*4);
    constexpr int Bs_loads  = (BN*BK)/(threads*4);
    constexpr int col_threads = BN / TN;

    static_assert(BK % 4 == 0, "As bf16 load crosses BK boundary");
    static_assert((BM*BK) % (threads*4) == 0, "As load not divisible by 4");
    static_assert((BN*BK) % (threads*4) == 0, "Bs load not divisible by 4");

    __shared__ float As[BK][BM];
    __shared__ float Bs[BK][BN];

    int tid = threadIdx.x + threadIdx.y*blockDim.x;

    float acc[TM*TN] = {0};
    for(int bk=0;bk<K;bk+=BK)
    {
        for (int load = 0; load < As_loads; load++) {
            int idx = (tid + load*threads) * 4;
            int as_r = idx / BK, as_c = idx % BK;
            int a_row = blockIdx.y * BM + as_r;
            int a_col = bk + as_c;
            float a0,a1,a2,a3; ld4_bf16(&A[a_row*K + a_col], a0,a1,a2,a3);
            As[as_c][as_r]   = a0;
            As[as_c+1][as_r] = a1;
            As[as_c+2][as_r] = a2;
            As[as_c+3][as_r] = a3;
        }
        for (int load = 0; load < Bs_loads; load++) {
            int idx = (tid + load*threads) * 4;
            int bs_r = idx / BN, bs_c = idx % BN;
            int b_row = bk + bs_r;
            int b_col = blockIdx.x * BN + bs_c;
            float b0,b1,b2,b3; ld4_bf16(&B[b_row*N + b_col], b0,b1,b2,b3);
            reinterpret_cast<float4*>(&Bs[bs_r][bs_c])[0] = make_float4(b0,b1,b2,b3);
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
                regA[i]=tmp.x; regA[i+1]=tmp.y; regA[i+2]=tmp.z; regA[i+3]=tmp.w;
            }
            for(int i=0;i<TN;i+=4)
            {
                float4 tmp = reinterpret_cast<float4*>(&Bs[k][col_in_tile + i])[0];
                regB[i]=tmp.x; regB[i+1]=tmp.y; regB[i+2]=tmp.z; regB[i+3]=tmp.w;
            }
            for(int i=0;i<TN;i++)
                for(int j=0;j<TM;j++)
                    acc[i*TM+j]+=regA[j] * regB[i];
        }
        __syncthreads();
    }
    int c_col_block = tid % col_threads;
    int c_row_block = tid / col_threads;
    for(int i=0;i<TN;i++)
        for(int j=0;j<TM;j++)
        {
            int global_row = blockIdx.y * BM + c_row_block * TM + j;
            int global_col = blockIdx.x * BN + c_col_block*TN +i;
            if (global_row < M && global_col < N)
                C[global_row * N + global_col] = alpha * acc[i*TM+j] + beta * C[global_row * N + global_col];
        }
}

template<int BM,int BN,int BK,int TM,int TN>
void launch_at_bf16(int M,int N,int K,float alpha,
                    const __nv_bfloat16*A,const __nv_bfloat16*B,float beta,float*C)
{
    constexpr int threads = (BM/TM)*(BN/TN);
    dim3 block(threads);
    dim3 grid(CEIL_DIV(N,BN), CEIL_DIV(M,BM));
    autotuning_kernel_bf16<BM,BN,BK,TM,TN><<<grid,block>>>(M,N,K,alpha,A,B,beta,C);
    CHECK_CUDA(cudaGetLastError());
}
