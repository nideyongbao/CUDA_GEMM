// ============================================================================
// tc_01 — WMMA 朴素版（tensor core 的 "hello world"）
//
// 用 WMMA (warp-level matrix multiply-accumulate) 让 **一个 warp(32 线程)** 协作
// 算一个 16×16 的 C tile。和 CUDA core 的本质区别：不再是"一个线程算一个 C 元素"，
// 而是 32 个线程把数据装进 wmma::fragment，由 Tensor Core 一条指令吃掉 16×16×16。
//
// 朴素在哪：A/B 的 fragment 直接从 global memory 取（load_matrix_sync 读 global），
// 没有 shared memory 复用 —— 所以会被访存卡死，喂不饱 Tensor Core（见 ncu）。
// 这是 tensor core 阶梯的第 ① 级，用来和后面 smem/流水线/WGMMA 对比。
//
// 约束：M、N 为 16 倍数，K 为 16 倍数（测试尺寸满足）。
// ============================================================================
#include <mma.h>
#include "../../include/tc_common.cuh"
using namespace nvcuda;

__global__ void wmma_naive_kernel(int M,int N,int K,
        const __nv_bfloat16* A,const __nv_bfloat16* B,float* C){
    // 该 warp 负责的 C tile 坐标（每 tile 16×16）
    int warpM = blockIdx.y*blockDim.y + threadIdx.y;       // tile 行号
    int warpN = (blockIdx.x*blockDim.x + threadIdx.x)/32;  // tile 列号

    wmma::fragment<wmma::matrix_a,16,16,16,__nv_bfloat16,wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b,16,16,16,__nv_bfloat16,wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator,16,16,16,float> acc;
    wmma::fill_fragment(acc,0.0f);

    int aRow=warpM*16, bCol=warpN*16;
    if(aRow<M && bCol<N){
        for(int k=0;k<K;k+=16){
            // 直接从 global 取 16×16 fragment：A 行主 MxK(lda=K)，B 行主 KxN(ldb=N)
            wmma::load_matrix_sync(a_frag, A + aRow*K + k, K);
            wmma::load_matrix_sync(b_frag, B + k*N + bCol, N);
            wmma::mma_sync(acc, a_frag, b_frag, acc);       // Tensor Core: 16×16×16
        }
        wmma::store_matrix_sync(C + aRow*N + bCol, acc, N, wmma::mem_row_major);
    }
}

void launch_wmma_naive(int M,int N,int K,
        const __nv_bfloat16* A,const __nv_bfloat16* B,float* C){
    // blockDim: x=128(=4 warps 沿 N), y=4(4 tile 行) → 每 block 覆盖 64×64
    dim3 block(128,4,1);
    dim3 grid((N+63)/64, (M+63)/64, 1);
    wmma_naive_kernel<<<grid,block>>>(M,N,K,A,B,C);
}

// 派发入口：由 tensor_core/{verify,bench} 按 id 调用（不再各自带 main）。
void tc01_verify(int M,int N,int K){ tc_verify("tc_01 WMMA_naive", launch_wmma_naive, M,N,K); }
void tc01_bench (int M,int N,int K){ tc_bench ("tc_01 WMMA_naive", launch_wmma_naive, M,N,K); }
