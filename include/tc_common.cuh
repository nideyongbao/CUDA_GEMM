#pragma once
// ============================================================================
// tensor_core 最简用例的公共脚手架（只服务 tc_01/02/03 这类 SS WMMA 用例）。
// 约定：C(M×N) = A(M×K) · B(K×N)，全部 row-major；A/B 为 bf16，C 为 fp32。
// 每个用例只需写好 kernel + launcher，然后 main 里一行 tc_run(...) 即可：
//   跑一次对拍 cuBLAS BF16（正确性）+ 计时（性能）+ 打印对 148 TFLOPS 的利用率。
// 这样每个 .cu 文件的注意力都集中在"这一步 tensor core 改了什么"。
// ============================================================================
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

#define TC_CHECK_CUDA(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} }while(0)
#define TC_CHECK_CUBLAS(x) do{ cublasStatus_t s=(x); if(s!=CUBLAS_STATUS_SUCCESS){ \
    fprintf(stderr,"cuBLAS %s:%d %d\n",__FILE__,__LINE__,(int)s); exit(1);} }while(0)

static const double TC_PEAK_BF16_GFLOPS = 148000.0;   // H20 BF16 Tensor Core dense 峰值

// host float -> bf16（截断高 16 位；参考与 kernel 读同一份 bf16，比较公平）
static inline __nv_bfloat16 tc_f2bf(float f){
    unsigned u; std::memcpy(&u,&f,4); unsigned short h=(unsigned short)(u>>16);
    __nv_bfloat16 b; std::memcpy(&b,&h,2); return b;
}

// cuBLAS BF16 参考（handle 复用——create/destroy 约 0.33ms/次，绝不能进计时循环）
static inline void tc_cublas_bf16(int M,int N,int K,
        const __nv_bfloat16* A,const __nv_bfloat16* B,float* C){
    static cublasHandle_t h=nullptr; if(!h) TC_CHECK_CUBLAS(cublasCreate(&h));
    float a=1.f,b=0.f;
    TC_CHECK_CUBLAS(cublasGemmEx(h,CUBLAS_OP_N,CUBLAS_OP_N,N,M,K,&a,
        B,CUDA_R_16BF,N, A,CUDA_R_16BF,K, &b, C,CUDA_R_32F,N,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
}

// 用例 kernel launcher 统一签名：C = A·B
typedef void(*TCLaunch)(int M,int N,int K,
        const __nv_bfloat16* A,const __nv_bfloat16* B,float* C);

// 跑一个用例：对拍 cuBLAS + 计时 + 利用率。
static inline void tc_run(const char* name, TCLaunch launch, int M,int N,int K){
    size_t nA=(size_t)M*K, nB=(size_t)K*N, nC=(size_t)M*N;
    __nv_bfloat16 *hA=(__nv_bfloat16*)malloc(nA*2),*hB=(__nv_bfloat16*)malloc(nB*2);
    srand(0);
    for(size_t i=0;i<nA;i++) hA[i]=tc_f2bf((float)rand()/RAND_MAX*2-1);
    for(size_t i=0;i<nB;i++) hB[i]=tc_f2bf((float)rand()/RAND_MAX*2-1);
    __nv_bfloat16 *dA,*dB; float *dC,*dR;
    TC_CHECK_CUDA(cudaMalloc(&dA,nA*2)); TC_CHECK_CUDA(cudaMalloc(&dB,nB*2));
    TC_CHECK_CUDA(cudaMalloc(&dC,nC*4)); TC_CHECK_CUDA(cudaMalloc(&dR,nC*4));
    TC_CHECK_CUDA(cudaMemcpy(dA,hA,nA*2,cudaMemcpyHostToDevice));
    TC_CHECK_CUDA(cudaMemcpy(dB,hB,nB*2,cudaMemcpyHostToDevice));

    // ---- 正确性：与 cuBLAS BF16 对拍（allclose atol=rtol=5e-2，bf16 误差略大）----
    TC_CHECK_CUDA(cudaMemset(dC,0,nC*4));
    launch(M,N,K,dA,dB,dC); TC_CHECK_CUDA(cudaGetLastError()); TC_CHECK_CUDA(cudaDeviceSynchronize());
    tc_cublas_bf16(M,N,K,dA,dB,dR); TC_CHECK_CUDA(cudaDeviceSynchronize());
    float *hC=(float*)malloc(nC*4),*hRr=(float*)malloc(nC*4);
    TC_CHECK_CUDA(cudaMemcpy(hC,dC,nC*4,cudaMemcpyDeviceToHost));
    TC_CHECK_CUDA(cudaMemcpy(hRr,dR,nC*4,cudaMemcpyDeviceToHost));
    double ma=0; int bad=0;
    for(size_t i=0;i<nC;i++){ double ae=fabs((double)hC[i]-hRr[i]);
        if(ae>ma) ma=ae; if(ae>5e-2+5e-2*fabs(hRr[i])) bad++; }
    printf("[%s] VERIFY max_abs=%.3e bad=%d/%zu  %s\n", name, ma, bad, nC, bad?"FAIL":"PASS");

    // ---- 性能：best-of，handle/分配都在计时外 ----
    int warm=3, rep=20;
    for(int i=0;i<warm;i++) launch(M,N,K,dA,dB,dC);
    TC_CHECK_CUDA(cudaDeviceSynchronize());
    cudaEvent_t s,e; cudaEventCreate(&s); cudaEventCreate(&e);
    cudaEventRecord(s);
    for(int i=0;i<rep;i++) launch(M,N,K,dA,dB,dC);
    cudaEventRecord(e); cudaEventSynchronize(e);
    float ms=0; cudaEventElapsedTime(&ms,s,e); ms/=rep;
    double gf=2.0*M*N*K/(ms/1e3)/1e9;
    printf("[%s] M=%d N=%d K=%d  time=%.4f ms  GFLOPS=%.2f  util(vs148T)=%.1f%%\n",
           name, M,N,K, ms, gf, gf/TC_PEAK_BF16_GFLOPS*100.0);
    cudaFree(dA);cudaFree(dB);cudaFree(dC);cudaFree(dR);
    free(hA);free(hB);free(hC);free(hRr);
}
