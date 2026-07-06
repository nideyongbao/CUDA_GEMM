// ============================================================================
// tc_02 — WMMA + shared memory staging（第 ② 级）
//
// 相对 tc_01 的唯一改动：每个 K 步先把 A(64×16)/B(16×64) tile **搬进 shared memory**，
// 再让 4 个 warp 从 smem 取 fragment 反复复用。和 CUDA core 教程里 "naive→smem"
// 是同一个思想：把 global 的重复读取换成 smem 内部复用，喂饱 Tensor Core。
//
// 布局：block tile 64×64，block=128 线程=4 warp，排成 2×2；每个 warp 负责 32×32 =
// 2×2 个 16×16 wmma fragment。smem 直接用 bf16（wmma fragment 要 bf16）。
// 约束：M、N 为 64 倍数，K 为 16 倍数。
// ============================================================================
#include <mma.h>
#include "../../include/tc_common.cuh"
using namespace nvcuda;

__global__ void wmma_smem_kernel(int M,int N,int K,
        const __nv_bfloat16* A,const __nv_bfloat16* B,float* C){
    constexpr int BM=64,BN=64,BK=16;
    __shared__ __nv_bfloat16 As[BM*BK];   // [BM][BK] row-major, ld=BK
    __shared__ __nv_bfloat16 Bs[BK*BN];   // [BK][BN] row-major, ld=BN
    int tid=threadIdx.x, warpId=tid/32;
    int warpRow=warpId/2, warpCol=warpId%2;          // 2×2 warps
    int blockRow=blockIdx.y*BM, blockCol=blockIdx.x*BN;

    wmma::fragment<wmma::accumulator,16,16,16,float> acc[2][2];
    for(int i=0;i<2;i++) for(int j=0;j<2;j++) wmma::fill_fragment(acc[i][j],0.0f);

    for(int k0=0;k0<K;k0+=BK){
        // 128 线程协作把 A/B tile 搬进 smem（每个各搬 BM*BK/128、BK*BN/128 个元素）
        for(int t=tid;t<BM*BK;t+=128){ int r=t/BK,c=t%BK; As[t]=A[(blockRow+r)*K + k0 + c]; }
        for(int t=tid;t<BK*BN;t+=128){ int r=t/BN,c=t%BN; Bs[t]=B[(k0+r)*N + blockCol + c]; }
        __syncthreads();
        // 每 warp 从 smem 取 2×2 fragment 做 wmma
        wmma::fragment<wmma::matrix_a,16,16,16,__nv_bfloat16,wmma::row_major> a_frag[2];
        wmma::fragment<wmma::matrix_b,16,16,16,__nv_bfloat16,wmma::row_major> b_frag[2];
        for(int i=0;i<2;i++){ int row=warpRow*32+i*16; wmma::load_matrix_sync(a_frag[i], &As[row*BK], BK); }
        for(int j=0;j<2;j++){ int col=warpCol*32+j*16; wmma::load_matrix_sync(b_frag[j], &Bs[col], BN); }
        for(int i=0;i<2;i++) for(int j=0;j<2;j++)
            wmma::mma_sync(acc[i][j], a_frag[i], b_frag[j], acc[i][j]);
        __syncthreads();
    }
    for(int i=0;i<2;i++) for(int j=0;j<2;j++){
        int row=blockRow+warpRow*32+i*16, col=blockCol+warpCol*32+j*16;
        wmma::store_matrix_sync(C + row*N + col, acc[i][j], N, wmma::mem_row_major);
    }
}

void launch_wmma_smem(int M,int N,int K,
        const __nv_bfloat16* A,const __nv_bfloat16* B,float* C){
    dim3 block(128);   // 4 warps
    dim3 grid((N+63)/64, (M+63)/64, 1);
    wmma_smem_kernel<<<grid,block>>>(M,N,K,A,B,C);
}

// 派发入口：由 tensor_core/{verify,bench} 按 id 调用（不再各自带 main）。
void tc02_verify(int M,int N,int K){ tc_verify("tc_02 WMMA_smem", launch_wmma_smem, M,N,K); }
void tc02_bench (int M,int N,int K){ tc_bench ("tc_02 WMMA_smem", launch_wmma_smem, M,N,K); }
