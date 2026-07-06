// ============================================================================
// tc_03 — WMMA + cp.async 多级流水线（第 ③ 级，Ampere 风格）
//
// 相对 tc_02 的改动：① 更大 block tile(128×64) 提高复用；② shared memory 改成
// NSTAGES=3 级 ring buffer；③ 用 cp.async 异步预取后续 K-tile，让"搬下一块"和
// "算当前块"在不同硬件上重叠。这就是"异步搬运"第一次出现。
//
// 它正是 Hopper TMA+mbarrier 多级流水线的同步原语版前身：
//   cp.async ↔ TMA，__pipeline_commit/wait_prior ↔ mbarrier 的 phase 等待。
// 布局：BM128×BN64×BK32，block=256(8 warp，排 4×2)，每 warp 32×32 = 2×2 fragment。
// 约束：M 为 128 倍数，N 为 64 倍数，K 为 32 倍数。
// ============================================================================
#include <mma.h>
#include <cuda_pipeline.h>
#include "../../include/tc_common.cuh"
using namespace nvcuda;

// 128 线程协作把 A(128×32)/B(32×64) tile 用 cp.async 异步搬进 smem（16B=8×bf16 一发）
__device__ __forceinline__ void pipe_load(
        __nv_bfloat16* As_s, __nv_bfloat16* Bs_s,
        const __nv_bfloat16* A, const __nv_bfloat16* B,
        int blockRow,int blockCol,int k0,int K,int N){
    for(int c=threadIdx.x; c<128*32/8; c+=blockDim.x){ int r=c/4, cc=c%4;   // BK/8=4
        __pipeline_memcpy_async(&As_s[r*32 + cc*8], &A[(blockRow+r)*K + k0 + cc*8], 16); }
    for(int c=threadIdx.x; c<32*64/8; c+=blockDim.x){ int r=c/8, cc=c%8;     // BN/8=8
        __pipeline_memcpy_async(&Bs_s[r*64 + cc*8], &B[(k0+r)*N + blockCol + cc*8], 16); }
}

__global__ void wmma_pipe_kernel(int M,int N,int K,
        const __nv_bfloat16* A,const __nv_bfloat16* B,float* C){
    constexpr int BM=128,BN=64,BK=32,NSTAGES=3,KSTEP=BK/16;   // KSTEP=2
    __shared__ __align__(16) __nv_bfloat16 As[NSTAGES][BM*BK];
    __shared__ __align__(16) __nv_bfloat16 Bs[NSTAGES][BK*BN];
    int tid=threadIdx.x, warpId=tid/32, warpRow=warpId/2, warpCol=warpId%2;  // 4×2 warps
    int blockRow=blockIdx.y*BM, blockCol=blockIdx.x*BN, K_TILES=K/BK;

    wmma::fragment<wmma::accumulator,16,16,16,float> acc[2][2];
    for(int i=0;i<2;i++) for(int j=0;j<2;j++) wmma::fill_fragment(acc[i][j],0.0f);

    // 预取前 NSTAGES-1 个 stage
    for(int s=0;s<NSTAGES-1;s++){
        if(s<K_TILES) pipe_load(As[s],Bs[s],A,B,blockRow,blockCol,s*BK,K,N);
        __pipeline_commit();
    }
    for(int kt=0;kt<K_TILES;kt++){
        __pipeline_wait_prior(NSTAGES-2);   // 当前 stage 的搬运已完成
        __syncthreads();
        int stage=kt%NSTAGES;
        wmma::fragment<wmma::matrix_a,16,16,16,__nv_bfloat16,wmma::row_major> a_frag[2];
        wmma::fragment<wmma::matrix_b,16,16,16,__nv_bfloat16,wmma::row_major> b_frag[2];
        for(int ks=0;ks<KSTEP;ks++){
            for(int i=0;i<2;i++){ int row=warpRow*32+i*16;
                wmma::load_matrix_sync(a_frag[i], &As[stage][row*BK + ks*16], BK); }
            for(int j=0;j<2;j++){ int col=warpCol*32+j*16;
                wmma::load_matrix_sync(b_frag[j], &Bs[stage][(ks*16)*BN + col], BN); }
            for(int i=0;i<2;i++) for(int j=0;j<2;j++)
                wmma::mma_sync(acc[i][j], a_frag[i], b_frag[j], acc[i][j]);
        }
        __syncthreads();
        int next=kt+NSTAGES-1;               // 预取 NSTAGES-1 步之后的 tile 到刚空出的 stage
        if(next<K_TILES) pipe_load(As[next%NSTAGES],Bs[next%NSTAGES],A,B,blockRow,blockCol,next*BK,K,N);
        __pipeline_commit();
    }
    for(int i=0;i<2;i++) for(int j=0;j<2;j++){
        int row=blockRow+warpRow*32+i*16, col=blockCol+warpCol*32+j*16;
        wmma::store_matrix_sync(C + row*N + col, acc[i][j], N, wmma::mem_row_major);
    }
}

void launch_wmma_pipe(int M,int N,int K,
        const __nv_bfloat16* A,const __nv_bfloat16* B,float* C){
    dim3 block(256);   // 8 warps
    dim3 grid((N+63)/64, (M+127)/128, 1);
    wmma_pipe_kernel<<<grid,block>>>(M,N,K,A,B,C);
}

// 派发入口：由 tensor_core/{verify,bench} 按 id 调用（不再各自带 main）。
void tc03_verify(int M,int N,int K){ tc_verify("tc_03 WMMA_pipe", launch_wmma_pipe, M,N,K); }
void tc03_bench (int M,int N,int K){ tc_bench ("tc_03 WMMA_pipe", launch_wmma_pipe, M,N,K); }
