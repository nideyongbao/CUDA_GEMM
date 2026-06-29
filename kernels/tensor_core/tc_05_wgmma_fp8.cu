// ============================================================================
// tc_05 — WGMMA + TMA + warp specialization（第 ⑤ 级，Hopper 原生 FP8 e4m3）
//
// 在 tc_04(BF16) 基础上换成 FP8 (e4m3)：H20 FP8 Tensor Core 峰值 ~296 TFLOPS，是
// BF16(148T) 的两倍。关键改动很少，正说明 Hopper 这套异步流水线对精度是"可插拔"的：
//   ① 元素类型 bf16(2B) -> e4m3(1B)；② WGMMA 指令 m64n128k16.bf16 -> m64n128k32.e4m3
//      （FP8 的 K 维是 32，且没有 transA/transB 立即数——FP8/INT8 只支持 TN）；
//   ③ BK 64 -> 128（让 smem 每行仍是 128 字节，对齐 128B swizzle）；TMA dtype 用 UINT8。
// 因为 smem 的"字节布局"和 BF16 完全一致（128 行 × 128 字节，每条 wgmma 跨 32 字节），
// 所以 128B-swizzle 描述符的魔数(leading=16, stride=1024)直接沿用。
//
// 正确性：FP8 没有简单的 cuBLAS GemmEx 路径（需 cuBLASLt），故用 CPU double 参考
// （解码同一份 fp8 输入做双精度累加）在小尺寸上对拍；大尺寸只测吞吐。
// 约束：M、N 为 128 倍数，K 为 128 倍数。
// ============================================================================
#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cuda.h>
#include <cuda/barrier>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

#define WARP_SIZE 32
#define WARPGROUP_SIZE 128
#define DEVICE_INLINE __device__ inline
#define SMEM_DESC_ENCODE(x) ((((uint64_t)(x)) & 0x3FFFF) >> 0x4)
#define WGMMA_FENCE()        asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory")
#define WGMMA_COMMIT_GROUP() asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory")
#define WGMMA_WAIT_GROUP(n)  asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(n) : "memory")

#define CHECK_CUDA(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} }while(0)

using barrier = cuda::barrier<cuda::thread_scope_block>;
namespace cde = cuda::device::experimental;
__host__ __device__ inline int div_ceil(int a,int b){ return (a%b)?(a/b+1):(a/b); }

DEVICE_INLINE uint64_t make_smem_desc(__nv_fp8_e4m3* ptr){
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(ptr);
    uint64_t desc = 0;
    desc |= SMEM_DESC_ENCODE(addr);
    desc |= SMEM_DESC_ENCODE((uint64_t)16) << 16;
    desc |= SMEM_DESC_ENCODE((uint64_t)1024) << 32;
    desc |= 1llu << 62;   // 128B swizzle
    return desc;
}

// wgmma.mma_async.m64n128k32.f32.e4m3.e4m3 — 64 个 f32 累加寄存器；FP8 无 trans 立即数
#define WGMMA_M64N128K32_F32E4M3E4M3(d, sA, sB, ScaleD, ScaleA, ScaleB) {                      \
    uint64_t desc_a = make_smem_desc(&(sA)[0]);                                               \
    uint64_t desc_b = make_smem_desc(&(sB)[0]);                                               \
    asm volatile("{\n"                                                                        \
      "wgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e4m3 "                               \
      "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"                              \
      " %16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,"                    \
      " %32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,"                    \
      " %48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63},"                   \
      " %64, %65, %66, %67, %68;\n}\n"                                                        \
      : "+f"((d)[0][0]),"+f"((d)[0][1]),"+f"((d)[0][2]),"+f"((d)[0][3]),                      \
        "+f"((d)[0][4]),"+f"((d)[0][5]),"+f"((d)[0][6]),"+f"((d)[0][7]),                      \
        "+f"((d)[1][0]),"+f"((d)[1][1]),"+f"((d)[1][2]),"+f"((d)[1][3]),                      \
        "+f"((d)[1][4]),"+f"((d)[1][5]),"+f"((d)[1][6]),"+f"((d)[1][7]),                      \
        "+f"((d)[2][0]),"+f"((d)[2][1]),"+f"((d)[2][2]),"+f"((d)[2][3]),                      \
        "+f"((d)[2][4]),"+f"((d)[2][5]),"+f"((d)[2][6]),"+f"((d)[2][7]),                      \
        "+f"((d)[3][0]),"+f"((d)[3][1]),"+f"((d)[3][2]),"+f"((d)[3][3]),                      \
        "+f"((d)[3][4]),"+f"((d)[3][5]),"+f"((d)[3][6]),"+f"((d)[3][7]),                      \
        "+f"((d)[4][0]),"+f"((d)[4][1]),"+f"((d)[4][2]),"+f"((d)[4][3]),                      \
        "+f"((d)[4][4]),"+f"((d)[4][5]),"+f"((d)[4][6]),"+f"((d)[4][7]),                      \
        "+f"((d)[5][0]),"+f"((d)[5][1]),"+f"((d)[5][2]),"+f"((d)[5][3]),                      \
        "+f"((d)[5][4]),"+f"((d)[5][5]),"+f"((d)[5][6]),"+f"((d)[5][7]),                      \
        "+f"((d)[6][0]),"+f"((d)[6][1]),"+f"((d)[6][2]),"+f"((d)[6][3]),                      \
        "+f"((d)[6][4]),"+f"((d)[6][5]),"+f"((d)[6][6]),"+f"((d)[6][7]),                      \
        "+f"((d)[7][0]),"+f"((d)[7][1]),"+f"((d)[7][2]),"+f"((d)[7][3]),                      \
        "+f"((d)[7][4]),"+f"((d)[7][5]),"+f"((d)[7][6]),"+f"((d)[7][7])                       \
      : "l"(desc_a),"l"(desc_b),"n"(int32_t(ScaleD)),"n"(int32_t(ScaleA)),"n"(int32_t(ScaleB))); }

template<int BlockMajor,int BlockMinor>
static void create_tensor_map(CUtensorMap* tma, __nv_fp8_e4m3* g, int bh, int bw){
    uint64_t shape[5]  = {(uint64_t)BlockMinor*bw,(uint64_t)BlockMajor*bh,1,1,1};
    uint64_t stride[5] = {sizeof(__nv_fp8_e4m3), sizeof(__nv_fp8_e4m3)*(uint64_t)BlockMinor*bw,0,0,0};
    uint32_t box[5]    = {(uint32_t)BlockMinor,(uint32_t)BlockMajor,1,1,1};
    uint32_t bstride[5]= {1,1,1,1,1};
    CUresult r = cuTensorMapEncodeTiled(tma, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, (void*)g,
        shape, stride+1, box, bstride, CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if(r!=CUDA_SUCCESS) printf("cuTensorMapEncodeTiled failed: %d\n",(int)r);
}

template<int BM,int BN,int BK,int QSIZE> struct SMem05{
    alignas(128) __nv_fp8_e4m3 A[BM*BK*QSIZE];
    alignas(128) __nv_fp8_e4m3 B[BK*BN*QSIZE];
};

// TN: A row-major MxK, Bt row-major NxK, C row-major MxN, C=A*B, f32 accum
template<int WGMMA_M=64,int WGMMA_N=128,int WGMMA_K=32,int BM=128,int BN=128,
         int BK=128,int NUM_THREADS=256,int K_STAGE=3>
__global__ void __launch_bounds__(NUM_THREADS)
wgmma_fp8_kernel(int M,int N,int K,float* C,const CUtensorMap* tmaA,const CUtensorMap* tmaB){
    const int bx=blockIdx.x, by=blockIdx.y;
    constexpr int num_consumers=(NUM_THREADS/WARPGROUP_SIZE)-1;   // 1
    constexpr int B_WG_M=BM/num_consumers;                         // 128
    if(bx>=div_ceil(N,BN)||by>=div_ceil(M,BM)) return;

    extern __shared__ __align__(128) uint8_t smem[];
    SMem05<BM,BN,BK,K_STAGE>& s=*reinterpret_cast<SMem05<BM,BN,BK,K_STAGE>*>(smem);
    __nv_fp8_e4m3* s_a=s.A; __nv_fp8_e4m3* s_b=s.B;

#pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ barrier full[K_STAGE], empty[K_STAGE];
    const int num_blocks_k=K/BK;
    const int wg_idx=threadIdx.x/WARPGROUP_SIZE;
    const int tid=threadIdx.x%WARPGROUP_SIZE;

    if(threadIdx.x==0){
        for(int i=0;i<K_STAGE;++i){ init(&full[i],WARPGROUP_SIZE+1); init(&empty[i],WARPGROUP_SIZE+1); }
        cde::fence_proxy_async_shared_cta();
    }
    __syncthreads();

    if(wg_idx==0){   // producer: TMA
        if(tid==0){
            int q=0;
            for(int kb=0;kb<num_blocks_k;++kb,++q){
                if(q==K_STAGE) q=0;
                empty[q].wait(empty[q].arrive());
                cde::cp_async_bulk_tensor_2d_global_to_shared(&s_a[q*BK*BM], tmaA, kb*BK, by*BM, full[q]);
                cde::cp_async_bulk_tensor_2d_global_to_shared(&s_b[q*BK*BN], tmaB, kb*BK, bx*BN, full[q]);
                (void)cuda::device::barrier_arrive_tx(full[q],1,(BK*BN+BK*BM)*sizeof(__nv_fp8_e4m3));
            }
        }
    } else {         // consumer: WGMMA
        for(int i=0;i<K_STAGE;++i) (void)empty[i].arrive();
        float d[B_WG_M/WGMMA_M][WGMMA_N/16][8];
        memset(d,0,sizeof(d));
        int q=0;
        for(int kb=0;kb<num_blocks_k;++kb,++q){
            if(q==K_STAGE) q=0;
            full[q].wait(full[q].arrive());
            WGMMA_FENCE();
#pragma unroll
            for(int m_it=0;m_it<B_WG_M/WGMMA_M;++m_it){
                __nv_fp8_e4m3* wsA=s_a+q*BK*BM+BK*m_it*WGMMA_M;
#pragma unroll
                for(int k_it=0;k_it<BK/WGMMA_K;++k_it)
                    WGMMA_M64N128K32_F32E4M3E4M3(d[m_it], wsA+k_it*WGMMA_K,
                        s_b+q*BK*BN+k_it*WGMMA_K, 1,1,1);
            }
            WGMMA_COMMIT_GROUP();
            WGMMA_WAIT_GROUP(0);
            (void)empty[q].arrive();
        }
        const int lane=tid%WARP_SIZE, warp=tid/WARP_SIZE;
        const int row=warp*16+lane/4;
        float* bC=C+by*BM*N+bx*BN;
#pragma unroll
        for(int m_it=0;m_it<B_WG_M/WGMMA_M;++m_it){ int yo=m_it*WGMMA_M;
#pragma unroll
            for(int g=0;g<WGMMA_N/16;++g){ int col=g*16+2*(lane%4);
#define IDX(i,j) (((i)+yo)*N+(j))
                bC[IDX(row,col)]      =d[m_it][g][0]; bC[IDX(row,col+1)]    =d[m_it][g][1];
                bC[IDX(row+8,col)]    =d[m_it][g][2]; bC[IDX(row+8,col+1)]  =d[m_it][g][3];
                bC[IDX(row,col+8)]    =d[m_it][g][4]; bC[IDX(row,col+9)]    =d[m_it][g][5];
                bC[IDX(row+8,col+8)]  =d[m_it][g][6]; bC[IDX(row+8,col+9)]  =d[m_it][g][7];
#undef IDX
            }
        }
    }
}

__global__ void transpose_fp8(const __nv_fp8_e4m3* B,__nv_fp8_e4m3* Bt,int K,int N){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<K*N){ int k=i/N, n=i%N; Bt[(size_t)n*K+k]=B[(size_t)k*N+n]; }
}

constexpr int BM=128,BN=128,BK=128,QSIZE=3,THREADS=256;

void run_fp8(int M,int N,int K,float* dC,const __nv_fp8_e4m3* dBt,CUtensorMap* tmaA,CUtensorMap* tmaB){
    int smem=sizeof(SMem05<BM,BN,BK,QSIZE>);
    static bool set=false;
    if(!set){ CHECK_CUDA(cudaFuncSetAttribute(
        wgmma_fp8_kernel<64,128,32,BM,BN,BK,THREADS,QSIZE>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem)); set=true; }
    dim3 grid(div_ceil(N,BN),div_ceil(M,BM)); dim3 block(THREADS);
    wgmma_fp8_kernel<64,128,32,BM,BN,BK,THREADS,QSIZE><<<grid,block,smem>>>(M,N,K,dC,tmaA,tmaB);
}

// 在 size=sz 上：CPU double 参考(解码同一份 fp8)对拍
static void verify(int sz){
    int M=sz,N=sz,K=sz; size_t nA=(size_t)M*K,nB=(size_t)K*N,nC=(size_t)M*N;
    __nv_fp8_e4m3 *hA=(__nv_fp8_e4m3*)malloc(nA),*hB=(__nv_fp8_e4m3*)malloc(nB);
    srand(0);
    for(size_t i=0;i<nA;i++) hA[i]=__nv_fp8_e4m3((float)rand()/RAND_MAX*2-1);
    for(size_t i=0;i<nB;i++) hB[i]=__nv_fp8_e4m3((float)rand()/RAND_MAX*2-1);
    __nv_fp8_e4m3 *dA,*dB,*dBt; float* dC;
    CHECK_CUDA(cudaMalloc(&dA,nA)); CHECK_CUDA(cudaMalloc(&dB,nB)); CHECK_CUDA(cudaMalloc(&dBt,nB));
    CHECK_CUDA(cudaMalloc(&dC,nC*4));
    CHECK_CUDA(cudaMemcpy(dA,hA,nA,cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB,hB,nB,cudaMemcpyHostToDevice));
    transpose_fp8<<<div_ceil(K*N,256),256>>>(dB,dBt,K,N); CHECK_CUDA(cudaDeviceSynchronize());
    CUtensorMap tA,tB,*dtA,*dtB;
    create_tensor_map<BM,BK>(&tA,dA,M/BM,K/BK);
    create_tensor_map<BN,BK>(&tB,dBt,N/BN,K/BK);
    CHECK_CUDA(cudaMalloc(&dtA,sizeof(CUtensorMap))); CHECK_CUDA(cudaMalloc(&dtB,sizeof(CUtensorMap)));
    CHECK_CUDA(cudaMemcpy(dtA,&tA,sizeof(CUtensorMap),cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dtB,&tB,sizeof(CUtensorMap),cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(dC,0,nC*4));
    run_fp8(M,N,K,dC,dBt,dtA,dtB); CHECK_CUDA(cudaGetLastError()); CHECK_CUDA(cudaDeviceSynchronize());
    float* hC=(float*)malloc(nC*4);
    CHECK_CUDA(cudaMemcpy(hC,dC,nC*4,cudaMemcpyDeviceToHost));
    // CPU double reference
    double ma=0; int bad=0;
    for(int i=0;i<M;i++) for(int j=0;j<N;j++){
        double s=0; for(int k=0;k<K;k++) s+=(double)float(hA[i*K+k])*(double)float(hB[k*N+j]);
        double ae=fabs(s-(double)hC[i*N+j]); if(ae>ma)ma=ae;
        if(ae>1e-1+5e-2*fabs(s)) bad++;
    }
    printf("[tc_05 WGMMA_fp8] VERIFY(%d^3, vs CPU double) max_abs=%.3e bad=%d/%zu  %s\n",
           sz, ma, bad, nC, bad?"FAIL":"PASS");
    cudaFree(dA);cudaFree(dB);cudaFree(dBt);cudaFree(dC);cudaFree(dtA);cudaFree(dtB);
    free(hA);free(hB);free(hC);
}

// ============================================================================
// 派发入口（由 tensor_core/{verify,bench} 按 id 调用，不再各自带 main）。
// FP8 没有简单 cuBLAS 路径，verify 用 CPU double 参考（O(n^3)），故限制在 <=512^3 对拍；
// bench 在传入的 M,N,K 上测吞吐。
// ============================================================================
void tc05_verify(int M,int N,int K){
    cuInit(0);
    int sz = (M < 512 ? M : 512);
    if(sz <= 0 || sz % BM){ printf("[tc_05] verify needs size multiple of %d\n", BM); return; }
    verify(sz);
}

void tc05_bench(int M,int N,int K){
    cuInit(0);
    if(M%BM||N%BN||K%BK){ printf("[tc_05] need M,N,K multiple of 128\n"); return; }
    size_t nA=(size_t)M*K,nB=(size_t)K*N,nC=(size_t)M*N;
    __nv_fp8_e4m3 *hA=(__nv_fp8_e4m3*)malloc(nA),*hB=(__nv_fp8_e4m3*)malloc(nB);
    srand(0);
    for(size_t i=0;i<nA;i++) hA[i]=__nv_fp8_e4m3((float)rand()/RAND_MAX*2-1);
    for(size_t i=0;i<nB;i++) hB[i]=__nv_fp8_e4m3((float)rand()/RAND_MAX*2-1);
    __nv_fp8_e4m3 *dA,*dB,*dBt; float* dC;
    CHECK_CUDA(cudaMalloc(&dA,nA)); CHECK_CUDA(cudaMalloc(&dB,nB)); CHECK_CUDA(cudaMalloc(&dBt,nB));
    CHECK_CUDA(cudaMalloc(&dC,nC*4));
    CHECK_CUDA(cudaMemcpy(dA,hA,nA,cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB,hB,nB,cudaMemcpyHostToDevice));
    transpose_fp8<<<div_ceil(K*N,256),256>>>(dB,dBt,K,N); CHECK_CUDA(cudaDeviceSynchronize());
    CUtensorMap tA,tB,*dtA,*dtB;
    create_tensor_map<BM,BK>(&tA,dA,M/BM,K/BK);
    create_tensor_map<BN,BK>(&tB,dBt,N/BN,K/BK);
    CHECK_CUDA(cudaMalloc(&dtA,sizeof(CUtensorMap))); CHECK_CUDA(cudaMalloc(&dtB,sizeof(CUtensorMap)));
    CHECK_CUDA(cudaMemcpy(dtA,&tA,sizeof(CUtensorMap),cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dtB,&tB,sizeof(CUtensorMap),cudaMemcpyHostToDevice));
    int warm=3,rep=20;
    for(int i=0;i<warm;i++) run_fp8(M,N,K,dC,dBt,dtA,dtB);
    CHECK_CUDA(cudaDeviceSynchronize());
    cudaEvent_t s,e; cudaEventCreate(&s); cudaEventCreate(&e);
    cudaEventRecord(s);
    for(int i=0;i<rep;i++) run_fp8(M,N,K,dC,dBt,dtA,dtB);
    cudaEventRecord(e); cudaEventSynchronize(e);
    float ms=0; cudaEventElapsedTime(&ms,s,e); ms/=rep;
    double gf=2.0*M*N*K/(ms/1e3)/1e9;
    printf("[tc_05 WGMMA_fp8] M=%d N=%d K=%d  time=%.4f ms  GFLOPS=%.2f  util(vs296T)=%.1f%%  util(vs148T)=%.1f%%\n",
           M,N,K,ms,gf,gf/296000.0*100.0,gf/148000.0*100.0);
    cudaFree(dA);cudaFree(dB);cudaFree(dBt);cudaFree(dC);cudaFree(dtA);cudaFree(dtB);
    free(hA);free(hB);
}
