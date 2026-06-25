// ============================================================================
// tc_04 — WGMMA + TMA + warp specialization（第 ④ 级，Hopper 原生 BF16）
//
// 最小 WGMMA(Hopper warpgroup 异步张量指令)版 BF16 GEMM
//
// WGMMA 的 smem 操作数必须是 TMA 产生的 128B-swizzle "core matrix" 布局，描述符
// 的魔数也和这套布局绑死 —— 所以"最小 WGMMA"实际上是 TMA + WGMMA 这一对。本文件
// 移植自 LeetCUDA 的 hgemm_wgmma_fp32acc_stages_tn(已验证正确)：
//   TMA 异步搬运 + warp specialization(1 生产者 WG + 1 消费者 WG) + 多级流水线
//   + wgmma.mma_async.m64n128k16.f32.bf16.bf16 + f32 累加。
// 与本项目 row-major 约定对齐：A(MxK) row-major，B(KxN) row-major；内部把 B 转置成
// N×K(只做一次，不计入计时)喂给 TN 版 wgmma。对拍 cuBLAS BF16。
//
// 这是"手写 tensor core 阶梯"的最高一级：WMMA(同步) -> cp.async+WMMA -> TMA+WGMMA。
// 直观看 warpgroup 异步张量指令 + TMA 相比 warp 级 WMMA 值多少。
// ============================================================================
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda.h>
#include <cuda/barrier>
#include <cublas_v2.h>
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
#define CHECK_CUBLAS(x) do{ cublasStatus_t s=(x); if(s!=CUBLAS_STATUS_SUCCESS){ \
    fprintf(stderr,"cuBLAS %s:%d %d\n",__FILE__,__LINE__,(int)s); exit(1);} }while(0)

using barrier = cuda::barrier<cuda::thread_scope_block>;
namespace cde = cuda::device::experimental;
__host__ __device__ inline int div_ceil(int a,int b){ return (a%b)?(a/b+1):(a/b); }

// 128B-swizzle smem 矩阵描述符（leading=16, stride=1024, swizzle bit62）
DEVICE_INLINE uint64_t make_smem_desc(__nv_bfloat16* ptr){
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(ptr);
    uint64_t desc = 0;
    desc |= SMEM_DESC_ENCODE(addr);
    desc |= SMEM_DESC_ENCODE((uint64_t)16) << 16;
    desc |= SMEM_DESC_ENCODE((uint64_t)1024) << 32;
    desc |= 1llu << 62;   // 128B swizzle
    return desc;
}

// wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16  (64 个 f32 累加寄存器)
#define WGMMA_M64N128K16_F32BF16BF16(d, sA, sB, ScaleD, ScaleA, ScaleB, TransA, TransB) {     \
    uint64_t desc_a = make_smem_desc(&(sA)[0]);                                               \
    uint64_t desc_b = make_smem_desc(&(sB)[0]);                                               \
    asm volatile("{\n"                                                                        \
      "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 "                               \
      "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"                              \
      " %16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,"                    \
      " %32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,"                    \
      " %48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63},"                   \
      " %64, %65, %66, %67, %68, %69, %70;\n}\n"                                              \
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
      : "l"(desc_a),"l"(desc_b),"n"(int32_t(ScaleD)),"n"(int32_t(ScaleA)),                    \
        "n"(int32_t(ScaleB)),"n"(int32_t(TransA)),"n"(int32_t(TransB))); }

// ---- TMA tensor map (2D, row-major (H,W) -> shape=(W,H)) ----
template<int BlockMajor,int BlockMinor>
static void create_tensor_map(CUtensorMap* tma, __nv_bfloat16* g, int bh, int bw){
    uint64_t shape[5]  = {(uint64_t)BlockMinor*bw,(uint64_t)BlockMajor*bh,1,1,1};
    uint64_t stride[5] = {sizeof(__nv_bfloat16), sizeof(__nv_bfloat16)*(uint64_t)BlockMinor*bw,0,0,0};
    uint32_t box[5]    = {(uint32_t)BlockMinor,(uint32_t)BlockMajor,1,1,1};
    uint32_t bstride[5]= {1,1,1,1,1};
    CUresult r = cuTensorMapEncodeTiled(tma, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, (void*)g,
        shape, stride+1, box, bstride, CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if(r!=CUDA_SUCCESS) printf("cuTensorMapEncodeTiled failed: %d\n",(int)r);
}

template<int BM,int BN,int BK,int QSIZE> struct SMem{
    alignas(128) __nv_bfloat16 A[BM*BK*QSIZE];
    alignas(128) __nv_bfloat16 B[BK*BN*QSIZE];
};

// TN: A row-major MxK, Bt row-major NxK, C row-major MxN, C=A*B, f32 accum
template<int WGMMA_M=64,int WGMMA_N=128,int WGMMA_K=16,int BM=128,int BN=128,
         int BK=64,int NUM_THREADS=256,int K_STAGE=3>
__global__ void __launch_bounds__(NUM_THREADS)
wgmma_kernel(int M,int N,int K,float* C,
             const __nv_bfloat16* A_unused,const __nv_bfloat16* B_unused,
             const CUtensorMap* tmaA,const CUtensorMap* tmaB){
    const int bx=blockIdx.x, by=blockIdx.y;
    constexpr int num_consumers=(NUM_THREADS/WARPGROUP_SIZE)-1;   // 1
    constexpr int B_WG_M=BM/num_consumers;                         // 128
    if(bx>=div_ceil(N,BN)||by>=div_ceil(M,BM)) return;

    extern __shared__ __align__(128) uint8_t smem[];
    SMem<BM,BN,BK,K_STAGE>& s=*reinterpret_cast<SMem<BM,BN,BK,K_STAGE>*>(smem);
    __nv_bfloat16* s_a=s.A; __nv_bfloat16* s_b=s.B;

#pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ barrier full[K_STAGE], empty[K_STAGE];

    const int num_blocks_k=K/BK;
    const int wg_idx=threadIdx.x/WARPGROUP_SIZE;   // 0=producer,1=consumer
    const int tid=threadIdx.x%WARPGROUP_SIZE;

    if(threadIdx.x==0){
        for(int i=0;i<K_STAGE;++i){
            init(&full[i], num_consumers*WARPGROUP_SIZE+1);
            init(&empty[i],num_consumers*WARPGROUP_SIZE+1);
        }
        cde::fence_proxy_async_shared_cta();
    }
    __syncthreads();

    if(wg_idx==0){   // producer: TMA loads
        if(tid==0){
            int q=0;
            for(int kb=0;kb<num_blocks_k;++kb,++q){
                if(q==K_STAGE) q=0;
                empty[q].wait(empty[q].arrive());
                cde::cp_async_bulk_tensor_2d_global_to_shared(&s_a[q*BK*BM], tmaA, kb*BK, by*BM, full[q]);
                cde::cp_async_bulk_tensor_2d_global_to_shared(&s_b[q*BK*BN], tmaB, kb*BK, bx*BN, full[q]);
                (void)cuda::device::barrier_arrive_tx(full[q],1,(BK*BN+BK*BM)*sizeof(__nv_bfloat16));
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
                __nv_bfloat16* wsA=s_a+q*BK*BM+BK*m_it*WGMMA_M;
#pragma unroll
                for(int k_it=0;k_it<BK/WGMMA_K;++k_it)
                    WGMMA_M64N128K16_F32BF16BF16(d[m_it], wsA+k_it*WGMMA_K,
                        s_b+q*BK*BN+k_it*WGMMA_K, 1,1,1,0,0);
            }
            WGMMA_COMMIT_GROUP();
            WGMMA_WAIT_GROUP(0);
            (void)empty[q].arrive();
        }
        // epilogue: f32 accum -> row-major C
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

// ---- B(KxN row-major) -> Bt(NxK row-major) ----
__global__ void transpose_bf16(const __nv_bfloat16* B,__nv_bfloat16* Bt,int K,int N){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<K*N){ int k=i/N, n=i%N; Bt[(size_t)n*K+k]=B[(size_t)k*N+n]; }
}

// ---- host bf16 ----
static __nv_bfloat16 f2bf(float f){ unsigned u; memcpy(&u,&f,4); unsigned short h=(unsigned short)(u>>16);
    __nv_bfloat16 b; memcpy(&b,&h,2); return b; }

void cublas_bf16_ref(cublasHandle_t h,int M,int N,int K,const __nv_bfloat16*A,const __nv_bfloat16*B,float*C){
    float a=1.f,b=0.f;
    CHECK_CUBLAS(cublasGemmEx(h,CUBLAS_OP_N,CUBLAS_OP_N,N,M,K,&a,
        B,CUDA_R_16BF,N, A,CUDA_R_16BF,K, &b, C,CUDA_R_32F,N,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
}

constexpr int BM=128,BN=128,BK=64,QSIZE=3,THREADS=256;

void run_wgmma(int M,int N,int K,float* dC,const __nv_bfloat16* dBt,CUtensorMap* tmaA,CUtensorMap* tmaB){
    int smem=sizeof(SMem<BM,BN,BK,QSIZE>);
    static bool set=false;
    if(!set){ CHECK_CUDA(cudaFuncSetAttribute(
        wgmma_kernel<64,128,16,BM,BN,BK,THREADS,QSIZE>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem)); set=true; }
    dim3 grid(div_ceil(N,BN),div_ceil(M,BM)); dim3 block(THREADS);
    wgmma_kernel<64,128,16,BM,BN,BK,THREADS,QSIZE><<<grid,block,smem>>>(M,N,K,dC,nullptr,nullptr,tmaA,tmaB);
}

int main(int argc,char**argv){
    int M=4096,N=4096,K=4096;
    if(argc==4){ M=atoi(argv[1]);N=atoi(argv[2]);K=atoi(argv[3]); }
    cuInit(0);
    if(M%BM||N%BN||K%BK){ printf("need M%%%d==0,N%%%d==0,K%%%d==0\n",BM,BN,BK); return 1; }

    size_t nA=(size_t)M*K,nB=(size_t)K*N,nC=(size_t)M*N;
    __nv_bfloat16 *hA=(__nv_bfloat16*)malloc(nA*2),*hB=(__nv_bfloat16*)malloc(nB*2);
    srand(0); for(size_t i=0;i<nA;i++) hA[i]=f2bf((float)rand()/RAND_MAX*2-1);
    for(size_t i=0;i<nB;i++) hB[i]=f2bf((float)rand()/RAND_MAX*2-1);

    __nv_bfloat16 *dA,*dB,*dBt; float *dC,*dCref;
    CHECK_CUDA(cudaMalloc(&dA,nA*2)); CHECK_CUDA(cudaMalloc(&dB,nB*2)); CHECK_CUDA(cudaMalloc(&dBt,nB*2));
    CHECK_CUDA(cudaMalloc(&dC,nC*4)); CHECK_CUDA(cudaMalloc(&dCref,nC*4));
    CHECK_CUDA(cudaMemcpy(dA,hA,nA*2,cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB,hB,nB*2,cudaMemcpyHostToDevice));
    transpose_bf16<<<div_ceil(K*N,256),256>>>(dB,dBt,K,N); CHECK_CUDA(cudaDeviceSynchronize());

    // TMA maps: A(MxK) box(BM,BK); Bt(NxK) box(BN,BK)
    CUtensorMap tA,tB,*dtA,*dtB;
    create_tensor_map<BM,BK>(&tA,dA,M/BM,K/BK);
    create_tensor_map<BN,BK>(&tB,dBt,N/BN,K/BK);
    CHECK_CUDA(cudaMalloc(&dtA,sizeof(CUtensorMap))); CHECK_CUDA(cudaMalloc(&dtB,sizeof(CUtensorMap)));
    CHECK_CUDA(cudaMemcpy(dtA,&tA,sizeof(CUtensorMap),cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dtB,&tB,sizeof(CUtensorMap),cudaMemcpyHostToDevice));

    cublasHandle_t h; CHECK_CUBLAS(cublasCreate(&h));

    // ---- verify ----
    CHECK_CUDA(cudaMemset(dC,0,nC*4));
    run_wgmma(M,N,K,dC,dBt,dtA,dtB); CHECK_CUDA(cudaGetLastError()); CHECK_CUDA(cudaDeviceSynchronize());
    cublas_bf16_ref(h,M,N,K,dA,dB,dCref); CHECK_CUDA(cudaDeviceSynchronize());
    float *hC=(float*)malloc(nC*4),*hR=(float*)malloc(nC*4);
    CHECK_CUDA(cudaMemcpy(hC,dC,nC*4,cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(hR,dCref,nC*4,cudaMemcpyDeviceToHost));
    double ma=0,mr=0; int bad=0;
    for(size_t i=0;i<nC;i++){ double ae=fabs((double)hC[i]-hR[i]); double re=ae/(fabs(hR[i])+1e-5);
        if(ae>ma)ma=ae; if(re>mr)mr=re; if(ae>5e-2+5e-2*fabs(hR[i])) bad++; }
    printf("VERIFY: max_abs=%.3e max_rel=%.3e bad=%d/%zu  %s\n",ma,mr,bad,nC,(bad==0?"PASS":"FAIL"));

    // ---- bench ----
    int warm=3,rep=20;
    for(int i=0;i<warm;i++) run_wgmma(M,N,K,dC,dBt,dtA,dtB);
    CHECK_CUDA(cudaDeviceSynchronize());
    cudaEvent_t s,e; cudaEventCreate(&s); cudaEventCreate(&e);
    cudaEventRecord(s);
    for(int i=0;i<rep;i++) run_wgmma(M,N,K,dC,dBt,dtA,dtB);
    cudaEventRecord(e); cudaEventSynchronize(e);
    float ms=0; cudaEventElapsedTime(&ms,s,e); ms/=rep;
    double gf=2.0*M*N*K/(ms/1e3)/1e9;
    printf("WGMMA(TMA+WS)      M=%d N=%d K=%d  time=%.4f ms  GFLOPS=%.2f  util(vs148T)=%.1f%%\n",
           M,N,K,ms,gf,gf/148000.0*100.0);

    // ---- fair cuBLAS BF16: handle 复用(不在计时内 create/destroy) ----
    for(int i=0;i<warm;i++) cublas_bf16_ref(h,M,N,K,dA,dB,dCref);
    CHECK_CUDA(cudaDeviceSynchronize());
    cudaEventRecord(s);
    for(int i=0;i<rep;i++) cublas_bf16_ref(h,M,N,K,dA,dB,dCref);
    cudaEventRecord(e); cudaEventSynchronize(e);
    float msb=0; cudaEventElapsedTime(&msb,s,e); msb/=rep;
    double gfb=2.0*M*N*K/(msb/1e3)/1e9;
    printf("cuBLAS BF16 (fair) M=%d N=%d K=%d  time=%.4f ms  GFLOPS=%.2f  util(vs148T)=%.1f%%\n",
           M,N,K,msb,gfb,gfb/148000.0*100.0);
    return 0;
}
