// BF16 benchmark：A/B 为 bf16，C 为 fp32；输出 GFLOPS 及对 H20 BF16
// Tensor Core 峰值(148 TFLOPS)的利用率。
#include "../include/bf16.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cublas_v2.h>

static const double PEAK_BF16_GFLOPS = 148000.0;   // H20 BF16 Tensor Core dense 峰值

// host float -> bf16（截断高 16 位，确定性；参考与 kernel 读同一份 bf16，比较公平）
static __nv_bfloat16 f2bf(float f){
    unsigned int u; std::memcpy(&u,&f,4);
    unsigned short hi=(unsigned short)(u>>16);
    __nv_bfloat16 b; std::memcpy(&b,&hi,2); return b;
}

void launch_cublas_bf16_ref(int M,int N,int K,float alpha,
        const __nv_bfloat16* A,const __nv_bfloat16* B,float beta,float* C){
    // handle 只创建一次：cublasCreate/Destroy 每次约 0.33ms，放进计时循环会严重
    // 拉低小尺寸的 cuBLAS GFLOPS（4096³ 下把 89% 假报成 68%）。
    static cublasHandle_t h = nullptr;
    if(!h) CHECK_CUBLAS(cublasCreate(&h));
    CHECK_CUBLAS(cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha,
        B, CUDA_R_16BF, N,
        A, CUDA_R_16BF, K,
        &beta,
        C, CUDA_R_32F, N,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
}

BF16KernelFn g_kernels[] = {
    launch_cublas_bf16_ref, launch_naive_bf16, launch_smem_bf16,
    launch_blocktiling_bf16, launch_2Dblocktiling_bf16, launch_vectorized_bf16,
    launch_at_bf16<64,64,8,8,4>, launch_at_bf16<64,64,16,8,4>, launch_at_bf16<64,64,8,8,8>,
    launch_warptile_bf16, launch_warptile_vec_bf16, launch_bank_conflict_bf16,
    launch_double_buffer_bf16 };
const char* g_names[] = {
    "cublas_bf16","naive","smem","blocktiling","2Dblocktiling","vectorized",
    "autotune_64x64x8_8x4","autotune_64x64x16_8x4","autotune_64x64x8_8x8",
    "warptile","warptile_vec","bank_conflict","double_buffer" };

void benchmark_kernel(const char* name, BF16KernelFn kernel, int M,int N,int K,
                      int warmup,int repeat){
    float alpha=1.0f, beta=0.0f;
    size_t nA=(size_t)M*K, nB=(size_t)K*N, nC=(size_t)M*N;
    float* hf=(float*)malloc(nA>nB?nA*4:nB*4);
    __nv_bfloat16* hA=(__nv_bfloat16*)malloc(nA*2);
    __nv_bfloat16* hB=(__nv_bfloat16*)malloc(nB*2);
    for(size_t i=0;i<nA;i++) hA[i]=f2bf(((i%17)-8)*0.1f);
    for(size_t i=0;i<nB;i++) hB[i]=f2bf(((i%17)-8)*0.1f);
    free(hf);
    __nv_bfloat16 *dA,*dB; float* dC;
    CHECK_CUDA(cudaMalloc(&dA,nA*2)); CHECK_CUDA(cudaMalloc(&dB,nB*2)); CHECK_CUDA(cudaMalloc(&dC,nC*4));
    CHECK_CUDA(cudaMemcpy(dA,hA,nA*2,cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB,hB,nB*2,cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(dC,0,nC*4));

    for(int i=0;i<warmup;i++) kernel(M,N,K,alpha,dA,dB,beta,dC);
    CHECK_CUDA(cudaDeviceSynchronize());
    cudaEvent_t s,e; CHECK_CUDA(cudaEventCreate(&s)); CHECK_CUDA(cudaEventCreate(&e));
    CHECK_CUDA(cudaEventRecord(s));
    for(int i=0;i<repeat;i++) kernel(M,N,K,alpha,dA,dB,beta,dC);
    CHECK_CUDA(cudaEventRecord(e)); CHECK_CUDA(cudaEventSynchronize(e));
    float ms=0; CHECK_CUDA(cudaEventElapsedTime(&ms,s,e)); ms/=repeat;
    double gflops = 2.0*(double)M*N*K/(ms/1e3)/1e9;
    printf("%-22s M=%d N=%d K=%d  time=%.4f ms  GFLOPS=%.2f  util(vs148T)=%.1f%%\n",
           name,M,N,K,ms,gflops, gflops/PEAK_BF16_GFLOPS*100.0);
    cudaEventDestroy(s); cudaEventDestroy(e);
    cudaFree(dA); cudaFree(dB); cudaFree(dC); free(hA); free(hB);
}

int main(int argc,char** argv){
    int M=1024,N=1024,K=1024,id=0;
    int num=sizeof(g_kernels)/sizeof(g_kernels[0]);
    if(argc==5){ id=atoi(argv[1]); M=atoi(argv[2]); N=atoi(argv[3]); K=atoi(argv[4]); }
    else if(argc!=1){ printf("Usage: %s [id M N K]\n",argv[0]); return 1; }
    if(id<0||id>=num){ for(int i=0;i<num;i++) printf("idx:%d,kernel:%s\n",i,g_names[i]); return 1; }
    benchmark_kernel(g_names[id], g_kernels[id], M,N,K, 2, 10);
    return 0;
}
