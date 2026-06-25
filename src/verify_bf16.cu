// BF16 correctness：与 cuBLAS BF16(cublasGemmEx, bf16 in / fp32 accum) 对拍。
// 所有 kernel 读同一份 bf16 A/B，差异只来自累加顺序 / Tensor Core 取整。
#include "../include/bf16.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cublas_v2.h>

static __nv_bfloat16 f2bf(float f){
    unsigned int u; std::memcpy(&u,&f,4);
    unsigned short hi=(unsigned short)(u>>16);
    __nv_bfloat16 b; std::memcpy(&b,&hi,2); return b;
}

void launch_cublas_bf16_ref(int M,int N,int K,float alpha,
        const __nv_bfloat16* A,const __nv_bfloat16* B,float beta,float* C){
    cublasHandle_t h; CHECK_CUBLAS(cublasCreate(&h));
    CHECK_CUBLAS(cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha,
        B, CUDA_R_16BF, N, A, CUDA_R_16BF, K, &beta,
        C, CUDA_R_32F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    CHECK_CUBLAS(cublasDestroy(h));
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

static int verify_result(const float* mine,const float* ref,int M,int N){
    // bf16 输入 + fp32 累加，误差略大于 fp32：用稍宽的 allclose
    const float atol=5e-2f, rtol=5e-2f;
    float ma=0, mr=0; int bad=0, bi=-1, bj=-1;
    for(int i=0;i<M;i++) for(int j=0;j<N;j++){
        int idx=i*N+j; float a=mine[idx], b=ref[idx];
        float ae=fabsf(a-b), re=ae/(fabsf(b)+1e-5f);
        if(ae>ma) ma=ae; if(re>mr) mr=re;
        if(ae > atol + rtol*fabsf(b)){ bad++; if(bi==-1){bi=i;bj=j;} }
    }
    printf("max_abs_err = %.6e\n", ma);
    printf("max_rel_err = %.6e\n", mr);
    printf("bad_count   = %d / %d\n", bad, M*N);
    if(bad>0) printf("first bad at C[%d][%d]: mine=%.6f ref=%.6f\n", bi,bj, mine[bi*N+bj], ref[bi*N+bj]);
    return bad==0;
}

void verify_kernel(BF16KernelFn kernel,int M,int N,int K){
    float alpha=1.0f, beta=0.0f;
    size_t nA=(size_t)M*K, nB=(size_t)K*N, nC=(size_t)M*N;
    __nv_bfloat16* hA=(__nv_bfloat16*)malloc(nA*2);
    __nv_bfloat16* hB=(__nv_bfloat16*)malloc(nB*2);
    for(size_t i=0;i<nA;i++) hA[i]=f2bf((float)rand()/RAND_MAX*2-1);
    for(size_t i=0;i<nB;i++) hB[i]=f2bf((float)rand()/RAND_MAX*2-1);
    float* hC=(float*)malloc(nC*4); float* hCref=(float*)malloc(nC*4);
    __nv_bfloat16 *dA,*dB; float *dC,*dCref;
    CHECK_CUDA(cudaMalloc(&dA,nA*2)); CHECK_CUDA(cudaMalloc(&dB,nB*2));
    CHECK_CUDA(cudaMalloc(&dC,nC*4)); CHECK_CUDA(cudaMalloc(&dCref,nC*4));
    CHECK_CUDA(cudaMemcpy(dA,hA,nA*2,cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB,hB,nB*2,cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(dC,0,nC*4)); CHECK_CUDA(cudaMemset(dCref,0,nC*4));
    kernel(M,N,K,alpha,dA,dB,beta,dC); CHECK_CUDA(cudaDeviceSynchronize());
    launch_cublas_bf16_ref(M,N,K,alpha,dA,dB,beta,dCref); CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(hC,dC,nC*4,cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(hCref,dCref,nC*4,cudaMemcpyDeviceToHost));
    int ok=verify_result(hC,hCref,M,N);
    printf(ok?"PASS\n":"FAIL\n");
    cudaFree(dA);cudaFree(dB);cudaFree(dC);cudaFree(dCref);
    free(hA);free(hB);free(hC);free(hCref);
}

int main(int argc,char** argv){
    int M=1024,N=1024,K=1024,id=0;
    int num=sizeof(g_kernels)/sizeof(g_kernels[0]);
    if(argc==5){ id=atoi(argv[1]); M=atoi(argv[2]); N=atoi(argv[3]); K=atoi(argv[4]); }
    else if(argc!=1){ printf("Usage: %s [id M N K]\n",argv[0]); return 1; }
    if(id<0||id>=num){ for(int i=0;i<num;i++) printf("idx:%d,kernel:%s\n",i,g_names[i]); return 1; }
    verify_kernel(g_kernels[id], M,N,K);
    return 0;
}
