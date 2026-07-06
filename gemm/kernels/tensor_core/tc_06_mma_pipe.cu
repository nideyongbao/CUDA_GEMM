// ============================================================================
// tc_06 — mma.sync + ldmatrix + 多级 cp.async（第 ⑥ 级，Ampere 原生 BF16）
//
// 为什么有这一级：tc_01→03 的 `nvcuda::wmma` C++ API 到头了——它固定 16×16×16
// fragment、load/store 有开销、寄存器与调度不可控，A800 上 tc_03 止步 ~43T(14% 峰)。
// 要逼近峰值必须**绕开 WMMA API**，直接用底层 PTX：
//   · mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32   —— warp 级张量核 MMA
//   · ldmatrix.sync.aligned.m8n8.x4/.x2                     —— 从 smem 直接喂 fragment
//   · cp.async.cg（16B 向量化）+ K_STAGE 级 ring buffer      —— 异步多级预取
// 这正是 CUTLASS 风格 Ampere GEMM 的核心，也是 tc_03(WMMA) 与 tc_04(Hopper WGMMA)
// 之间缺的"承上启下"一级。mma.sync/ldmatrix/cp.async 都是 sm_80+ 通用指令，故本用例
// 在 A800 与 H20 上都能编都能跑（**始终注册**，不放进 #ifndef NO_HOPPER）；在 Hopper
// 上作为"mma.sync(warp) vs WGMMA(warpgroup)"的对照。
//
// 布局：A(M×K) row-major，B(K×N) row-major；内部把 B 转成 Bt(N×K) row-major(只做一次，
//   不计入计时)，使 A、Bt 两个 smem tile 都以 K 连续 —— A、B fragment 走**同一条非转置
//   ldmatrix 路径**（对称、最不易错）。C(M×N)=A·B，row-major，f32 累加。
// 分块：BM128×BN128×BK32；block=256(8 warp 排 2×4)，warp tile 64(M)×32(N) = 4×4 条
//   m16n8k16，每线程 64 个 f32 累加寄存器。K_STAGE=3 级 cp.async 流水（骨架同 tc_03）。
// 约束：M、N 为 128 倍数，K 为 32 倍数。
// ============================================================================
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_pipeline.h>
#include <cstdint>
#include "../../include/tc_common.cuh"   // TC_CHECK_*, tc_f2bf, tc_cublas_bf16（均 static inline，无链接冲突）

// ---- 张量核配置（可编译期 -DTC06_* 覆盖以复现调优扫描）----
// A800 锁频 4096³ 调优结论(见 docs/a800，约 25 组配置)：下方默认(BM128 BN128 BK32 · W2×4 · 3 级)
//   最优，≈150T=48%峰=3.5×tc_03。更大 warp tile 的数据复用/ILP 比提高占用率(kernel 受寄存器限
//   122 regs→2 block/SM)更划算；BK64/16-warp/BN256 均更差；朴素 XOR swizzle 与 B 用 x4 装载
//   都未跑赢"padding + B 用 x2"。**寄存器级 fragment 双缓冲实测也无益**(nvcc 对完全展开的内层循环
//   本就自动软件流水)；强制 3 block/SM 也不提速(非占用率受限)——~48% 是纯 CUDA C++ 上限，再往
//   60–85% 属 CUTLASS/SASS 级寄存器分配与指令调度(cuBLAS 走此路到 85%)。详见 docs/a800 §5-6。
#ifndef TC06_BM
#define TC06_BM 128
#endif
#ifndef TC06_BN
#define TC06_BN 128
#endif
#ifndef TC06_BK
#define TC06_BK 32
#endif
#ifndef TC06_WM
#define TC06_WM 2
#endif
#ifndef TC06_WN
#define TC06_WN 4
#endif
#ifndef TC06_STAGE
#define TC06_STAGE 3
#endif
namespace {
constexpr int BM = TC06_BM, BN = TC06_BN, BK = TC06_BK;   // block tile
constexpr int WARPS_M = TC06_WM, WARPS_N = TC06_WN;       // 8 warps = 256 threads
constexpr int K_STAGE = TC06_STAGE;                       // cp.async 流水级数
constexpr int WM = BM / WARPS_M;             // 每 warp 负责 64 行(M)
constexpr int WN = BN / WARPS_N;             // 每 warp 负责 32 列(N)
constexpr int PAD = 8;                       // smem 每行填充(元素)，避 ldmatrix bank conflict
constexpr int LDS = BK + PAD;                // smem 行跨度(元素)；LDS*2=80B 为 16B 对齐
constexpr int THREADS = WARPS_M * WARPS_N * 32;

// ---- PTX 原语（内部链接，避免与 tc_04 等同名符号冲突）----
__device__ __forceinline__ uint32_t smem_ptr(const void* p){
    return (uint32_t)__cvta_generic_to_shared(p);
}
// ldmatrix.x4：从 smem 取 4 个 8×8 bf16 矩阵 → 一个 16×16 fragment（A 用）
__device__ __forceinline__ void ldmatrix_x4(uint32_t& r0,uint32_t& r1,uint32_t& r2,uint32_t& r3,uint32_t a){
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(a));
}
// ldmatrix.x2：取 2 个 8×8 → 一个 16(K)×8(N) fragment（B 用；从 Bt 的 8(N)×16(K) 区域）
__device__ __forceinline__ void ldmatrix_x2(uint32_t& r0,uint32_t& r1,uint32_t a){
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
        : "=r"(r0),"=r"(r1) : "r"(a));
}
// mma.sync.m16n8k16：D(16×8 f32) = A(16×16 bf16)·B(16×8 bf16) + C(16×8 f32)
__device__ __forceinline__ void mma_m16n8k16(float& d0,float& d1,float& d2,float& d3,
        uint32_t a0,uint32_t a1,uint32_t a2,uint32_t a3,uint32_t b0,uint32_t b1,
        float c0,float c1,float c2,float c3){
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(d0),"=f"(d1),"=f"(d2),"=f"(d3)
        : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),
          "f"(c0),"f"(c1),"f"(c2),"f"(c3));
}

// 256 线程协作把 A(BM×BK)/Bt(BN×BK) tile 用 cp.async 异步搬进 padded smem（16B=8×bf16 一发）
__device__ __forceinline__ void cp_async_tile(__nv_bfloat16* As_s,__nv_bfloat16* Bs_s,
        const __nv_bfloat16* A,const __nv_bfloat16* Bt,
        int blockRow,int blockCol,int k0,int K){
    constexpr int KV = BK / 8;   // 每行 16B 块数
    for(int c=threadIdx.x; c<BM*KV; c+=THREADS){ int m=c/KV, kk=c%KV;
        __pipeline_memcpy_async(&As_s[m*LDS + kk*8], &A[(size_t)(blockRow+m)*K + k0 + kk*8], 16); }
    for(int c=threadIdx.x; c<BN*KV; c+=THREADS){ int n=c/KV, kk=c%KV;
        __pipeline_memcpy_async(&Bs_s[n*LDS + kk*8], &Bt[(size_t)(blockCol+n)*K + k0 + kk*8], 16); }
}

__global__ void __launch_bounds__(THREADS) mma_pipe_kernel(int M,int N,int K,
        const __nv_bfloat16* A,const __nv_bfloat16* Bt,float* C){
    extern __shared__ __nv_bfloat16 smem[];
    __nv_bfloat16* As = smem;                       // K_STAGE × BM × LDS
    __nv_bfloat16* Bs = smem + K_STAGE*BM*LDS;      // K_STAGE × BN × LDS
    int tid=threadIdx.x, lane=tid%32, warp=tid/32;
    int warpM=warp/WARPS_N, warpN=warp%WARPS_N;
    int blockRow=blockIdx.y*BM, blockCol=blockIdx.x*BN, K_TILES=K/BK;

    float acc[WM/16][WN/8][4];
    for(int i=0;i<WM/16;i++) for(int j=0;j<WN/8;j++) for(int r=0;r<4;r++) acc[i][j][r]=0.f;

    // 预取前 K_STAGE-1 个 stage
    for(int s=0;s<K_STAGE-1;s++){
        if(s<K_TILES) cp_async_tile(As+s*BM*LDS, Bs+s*BN*LDS, A,Bt, blockRow,blockCol, s*BK, K);
        __pipeline_commit();
    }
    for(int kt=0;kt<K_TILES;kt++){
        __pipeline_wait_prior(K_STAGE-2);   // 当前 stage 的搬运已完成
        __syncthreads();
        int stage=kt%K_STAGE;
        __nv_bfloat16* As_s=As+stage*BM*LDS;
        __nv_bfloat16* Bs_s=Bs+stage*BN*LDS;
        #pragma unroll
        for(int ks=0;ks<BK/16;ks++){
            uint32_t a[WM/16][4], b[WN/8][2];
            #pragma unroll
            for(int mi=0;mi<WM/16;mi++){ int Am=warpM*WM+mi*16, Ak=ks*16;
                ldmatrix_x4(a[mi][0],a[mi][1],a[mi][2],a[mi][3],
                    smem_ptr(&As_s[(Am+lane%16)*LDS + Ak + (lane/16)*8])); }
            #pragma unroll
            for(int ni=0;ni<WN/8;ni++){ int Bn=warpN*WN+ni*8, Bk=ks*16; int ln=lane%16;
                ldmatrix_x2(b[ni][0],b[ni][1],
                    smem_ptr(&Bs_s[(Bn+ln%8)*LDS + Bk + (ln/8)*8])); }
            #pragma unroll
            for(int mi=0;mi<WM/16;mi++) for(int ni=0;ni<WN/8;ni++)
                mma_m16n8k16(acc[mi][ni][0],acc[mi][ni][1],acc[mi][ni][2],acc[mi][ni][3],
                    a[mi][0],a[mi][1],a[mi][2],a[mi][3], b[ni][0],b[ni][1],
                    acc[mi][ni][0],acc[mi][ni][1],acc[mi][ni][2],acc[mi][ni][3]);
        }
        __syncthreads();
        int next=kt+K_STAGE-1;              // 预取 K_STAGE-1 步之后的 tile 到刚空出的 stage
        if(next<K_TILES) cp_async_tile(As+(next%K_STAGE)*BM*LDS, Bs+(next%K_STAGE)*BN*LDS,
                                       A,Bt, blockRow,blockCol, next*BK, K);
        __pipeline_commit();
    }
    // epilogue：f32 累加寄存器 → row-major C。m16n8 输出：g=lane/4, t=lane%4
    int g=lane/4, t=lane%4;
    #pragma unroll
    for(int mi=0;mi<WM/16;mi++) for(int ni=0;ni<WN/8;ni++){
        int Cm=blockRow+warpM*WM+mi*16, Cn=blockCol+warpN*WN+ni*8;
        C[(size_t)(Cm+g)*N   + Cn+2*t+0]=acc[mi][ni][0];
        C[(size_t)(Cm+g)*N   + Cn+2*t+1]=acc[mi][ni][1];
        C[(size_t)(Cm+g+8)*N + Cn+2*t+0]=acc[mi][ni][2];
        C[(size_t)(Cm+g+8)*N + Cn+2*t+1]=acc[mi][ni][3];
    }
}

// B(K×N row-major) → Bt(N×K row-major)，只做一次（不计入计时）
__global__ void tc06_transpose(const __nv_bfloat16* B,__nv_bfloat16* Bt,int K,int N){
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
    if(i<(size_t)K*N){ int k=i/N, n=i%N; Bt[(size_t)n*K+k]=B[(size_t)k*N+n]; }
}

// 启动器：转置后的 Bt 已就绪，只跑 mma kernel（供 verify/bench 在计时内重复调用）
void tc06_launch(int M,int N,int K,const __nv_bfloat16* A,const __nv_bfloat16* Bt,float* C){
    int smem=K_STAGE*(BM+BN)*LDS*(int)sizeof(__nv_bfloat16);
    static bool set=false;
    if(!set){ TC_CHECK_CUDA(cudaFuncSetAttribute(mma_pipe_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem)); set=true; }
    dim3 grid((N+BN-1)/BN, (M+BM-1)/BM);
    mma_pipe_kernel<<<grid,THREADS,smem>>>(M,N,K,A,Bt,C);
}
} // namespace

// ============================================================================
// 派发入口（由 tensor_core/{verify,bench} 按 id=6 调用）。bespoke harness：B 转置一次
// (计时外) + 对拍 cuBLAS BF16(复用 tc_common)，与 tc_04 同风格。
// ============================================================================
static const char* TC06 = "tc_06 MMA_pipe";

void tc06_verify(int M,int N,int K){
    if(M%BM||N%BN||K%BK){ printf("[%s] need M%%%d==0,N%%%d==0,K%%%d==0\n",TC06,BM,BN,BK); return; }
    size_t nA=(size_t)M*K,nB=(size_t)K*N,nC=(size_t)M*N;
    __nv_bfloat16 *hA=(__nv_bfloat16*)malloc(nA*2),*hB=(__nv_bfloat16*)malloc(nB*2);
    srand(0);
    for(size_t i=0;i<nA;i++) hA[i]=tc_f2bf((float)rand()/RAND_MAX*2-1);
    for(size_t i=0;i<nB;i++) hB[i]=tc_f2bf((float)rand()/RAND_MAX*2-1);
    __nv_bfloat16 *dA,*dB,*dBt; float *dC,*dR;
    TC_CHECK_CUDA(cudaMalloc(&dA,nA*2)); TC_CHECK_CUDA(cudaMalloc(&dB,nB*2)); TC_CHECK_CUDA(cudaMalloc(&dBt,nB*2));
    TC_CHECK_CUDA(cudaMalloc(&dC,nC*4)); TC_CHECK_CUDA(cudaMalloc(&dR,nC*4));
    TC_CHECK_CUDA(cudaMemcpy(dA,hA,nA*2,cudaMemcpyHostToDevice));
    TC_CHECK_CUDA(cudaMemcpy(dB,hB,nB*2,cudaMemcpyHostToDevice));
    tc06_transpose<<<(int)((nB+255)/256),256>>>(dB,dBt,K,N); TC_CHECK_CUDA(cudaDeviceSynchronize());
    TC_CHECK_CUDA(cudaMemset(dC,0,nC*4));
    tc06_launch(M,N,K,dA,dBt,dC); TC_CHECK_CUDA(cudaGetLastError()); TC_CHECK_CUDA(cudaDeviceSynchronize());
    tc_cublas_bf16(M,N,K,dA,dB,dR); TC_CHECK_CUDA(cudaDeviceSynchronize());
    float *hC=(float*)malloc(nC*4),*hR=(float*)malloc(nC*4);
    TC_CHECK_CUDA(cudaMemcpy(hC,dC,nC*4,cudaMemcpyDeviceToHost));
    TC_CHECK_CUDA(cudaMemcpy(hR,dR,nC*4,cudaMemcpyDeviceToHost));
    double ma=0,mr=0; int bad=0;
    for(size_t i=0;i<nC;i++){ double ae=fabs((double)hC[i]-hR[i]); double re=ae/(fabs(hR[i])+1e-5);
        if(ae>ma)ma=ae; if(re>mr)mr=re; if(ae>5e-2+5e-2*fabs(hR[i])) bad++; }
    printf("[%s] VERIFY max_abs=%.3e max_rel=%.3e bad=%d/%zu  %s\n",TC06,ma,mr,bad,nC,bad?"FAIL":"PASS");
    cudaFree(dA);cudaFree(dB);cudaFree(dBt);cudaFree(dC);cudaFree(dR);
    free(hA);free(hB);free(hC);free(hR);
}

void tc06_bench(int M,int N,int K){
    if(M%BM||N%BN||K%BK){ printf("[%s] need M%%%d==0,N%%%d==0,K%%%d==0\n",TC06,BM,BN,BK); return; }
    size_t nA=(size_t)M*K,nB=(size_t)K*N,nC=(size_t)M*N;
    __nv_bfloat16 *hA=(__nv_bfloat16*)malloc(nA*2),*hB=(__nv_bfloat16*)malloc(nB*2);
    srand(0);
    for(size_t i=0;i<nA;i++) hA[i]=tc_f2bf((float)rand()/RAND_MAX*2-1);
    for(size_t i=0;i<nB;i++) hB[i]=tc_f2bf((float)rand()/RAND_MAX*2-1);
    __nv_bfloat16 *dA,*dB,*dBt; float *dC;
    TC_CHECK_CUDA(cudaMalloc(&dA,nA*2)); TC_CHECK_CUDA(cudaMalloc(&dB,nB*2)); TC_CHECK_CUDA(cudaMalloc(&dBt,nB*2));
    TC_CHECK_CUDA(cudaMalloc(&dC,nC*4));
    TC_CHECK_CUDA(cudaMemcpy(dA,hA,nA*2,cudaMemcpyHostToDevice));
    TC_CHECK_CUDA(cudaMemcpy(dB,hB,nB*2,cudaMemcpyHostToDevice));
    tc06_transpose<<<(int)((nB+255)/256),256>>>(dB,dBt,K,N); TC_CHECK_CUDA(cudaDeviceSynchronize());
    int warm=3, rep=20;
    for(int i=0;i<warm;i++) tc06_launch(M,N,K,dA,dBt,dC);
    TC_CHECK_CUDA(cudaDeviceSynchronize());
    cudaEvent_t s,e; cudaEventCreate(&s); cudaEventCreate(&e);
    cudaEventRecord(s);
    for(int i=0;i<rep;i++) tc06_launch(M,N,K,dA,dBt,dC);
    cudaEventRecord(e); cudaEventSynchronize(e);
    float ms=0; cudaEventElapsedTime(&ms,s,e); ms/=rep;
    double gf=2.0*M*N*K/(ms/1e3)/1e9;
    // 注：%峰值不在此硬编（tc_common 的 148T 是 H20 口径）；真实利用率由 gemm_summary.py
    //     按 gpu_specs 的本机 BF16 峰值(A800=312T)重算。这里只给 GFLOPS/TFLOPS。
    printf("[%s] M=%d N=%d K=%d  time=%.4f ms  GFLOPS=%.2f  TFLOPS=%.2f\n",
           TC06, M,N,K, ms, gf, gf/1000.0);
    cudaFree(dA);cudaFree(dB);cudaFree(dBt);cudaFree(dC);
    free(hA);free(hB);
}
