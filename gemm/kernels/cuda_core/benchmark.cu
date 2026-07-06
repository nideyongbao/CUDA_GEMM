#include "../../include/common.h"
#include "../../include/kernels.h"
#include "../../include/06_autotuning.cuh"
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>

struct Cfg { KernelFn fn; const char* name; };
static const Cfg g_cfgs[] = {
    { launch_at<64, 64,  8, 8, 4>, "64x64x8_8x4"   },
    { launch_at<64, 64,  8, 8, 8>, "64x64x8_8x8"   },
    { launch_at<128,128, 8, 8, 8>, "128x128x8_8x8" },
    { launch_at<64, 64, 16, 8, 4>, "64x64x16_8x4"  },
};
static const int g_cfg_count = sizeof(g_cfgs)/sizeof(g_cfgs[0]);

void run_autotune(int M,int N,int K){
    size_t bytes_A=(size_t)M*K*sizeof(float);   // size_t! 防溢出(你文档的规矩)
    size_t bytes_B=(size_t)K*N*sizeof(float);
    size_t bytes_C=(size_t)M*N*sizeof(float);
    float *dA,*dB,*dC;
    CHECK_CUDA(cudaMalloc(&dA,bytes_A));
    CHECK_CUDA(cudaMalloc(&dB,bytes_B));
    CHECK_CUDA(cudaMalloc(&dC,bytes_C));
    // init dA,dB(随便填或复用 init_matrix)

    cudaEvent_t s,e; cudaEventCreate(&s); cudaEventCreate(&e);
    double flop = 2.0*M*N*K;

    for(int i=0;i<g_cfg_count;i++){
        // warmup 一发(丢弃首发抖动)
        g_cfgs[i].fn(M,N,K,1.0f,dA,dB,0.0f,dC);
        cudaDeviceSynchronize();
        // 计时(跑几次取最优)
        float best=1e30f;
        for(int r=0;r<3;r++){
            cudaEventRecord(s);
            g_cfgs[i].fn(M,N,K,1.0f,dA,dB,0.0f,dC);
            cudaEventRecord(e); cudaEventSynchronize(e);
            float ms; cudaEventElapsedTime(&ms,s,e);
            if(ms<best) best=ms;
        }
        printf("%-16s %8.2f GFLOPS  (%.2f ms)\n",
               g_cfgs[i].name, flop/(best/1e3)/1e9, best);
    }
    cudaFree(dA);cudaFree(dB);cudaFree(dC);
}


void launch_cublas_ref(int M, int N, int K,
                       float alpha,
                       const float* A,
                       const float* B,
                       float beta,
                       float* C)
{
    // handle 只创建一次：cublasCreate/Destroy 每次约 0.33ms，放进计时循环会
    // 严重拉低 cuBLAS 的 GFLOPS（小尺寸尤甚），造成"手写超过 cuBLAS"的假象。
    static cublasHandle_t handle = nullptr;
    if(!handle) CHECK_CUBLAS(cublasCreate(&handle));

    CHECK_CUBLAS(cublasSgemm(handle,
                             CUBLAS_OP_N,
                             CUBLAS_OP_N,
                             N, M, K,
                             &alpha,
                             B, N,
                             A, K,
                             &beta,
                             C, N));
}


__global__ void dummy_kernel(int M, int N, int K,
                             float alpha,
                             const float* A,
                             const float* B,
                             float beta,
                             float* C)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int size = M * N;

    if (idx < size) {
        C[idx] = 0.0f;
    }
}

void launch_dummy_kernel(int M, int N, int K,
                         float alpha,
                         const float* A,
                         const float* B,
                         float beta,
                         float* C)
{
    int size = M * N;
    int block_size = 256;
    int grid_size = (size + block_size - 1) / block_size;

    dummy_kernel<<<grid_size, block_size>>>(M, N, K, alpha, A, B, beta, C);

    CHECK_CUDA(cudaGetLastError());
}

void init_matrix(float* mat, int size)
{
    for (int i = 0; i < size; i++) {
        mat[i] = ((i % 17) - 8) * 0.1f;
    }
}




void benchmark_kernel(const char* name,
                      KernelFn kernel,
                      int M, int N, int K,
                      int warmup_iters,
                      int repeat_iters)
{
    float alpha = 1.0f;
    float beta = 0.0f;

    size_t bytes_A = (size_t)M * K * sizeof(float);
    size_t bytes_B = (size_t)K * N * sizeof(float);
    size_t bytes_C = (size_t)M * N * sizeof(float);

    float* h_A = (float*)malloc(bytes_A);
    float* h_B = (float*)malloc(bytes_B);
    float* h_C = (float*)malloc(bytes_C);

    if (!h_A || !h_B || !h_C) {
        printf("Host malloc failed\n");
        exit(1);
    }

    init_matrix(h_A, M * K);
    init_matrix(h_B, K * N);
    init_matrix(h_C, M * N);

    float* d_A = nullptr;
    float* d_B = nullptr;
    float* d_C = nullptr;

    CHECK_CUDA(cudaMalloc(&d_A, bytes_A));
    CHECK_CUDA(cudaMalloc(&d_B, bytes_B));
    CHECK_CUDA(cudaMalloc(&d_C, bytes_C));

    CHECK_CUDA(cudaMemcpy(d_A, h_A, bytes_A, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, bytes_B, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_C, h_C, bytes_C, cudaMemcpyHostToDevice));

    for (int i = 0; i < warmup_iters; i++) {
        kernel(M, N, K, alpha, d_A, d_B, beta, d_C);
    }

    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start;
    cudaEvent_t stop;

    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));

    for (int i = 0; i < repeat_iters; i++) {
        kernel(M, N, K, alpha, d_A, d_B, beta, d_C);
    }

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float total_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&total_ms, start, stop));

    float avg_ms = total_ms / repeat_iters;

    double flops = 2.0 * (double)M * N * K;
    double gflops = flops / (avg_ms / 1000.0) / 1e9;

    printf("%s: M=%d N=%d K=%d, time=%.4f ms, GFLOPS=%.2f\n",
           name, M, N, K, avg_ms, gflops);

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));

    free(h_A);
    free(h_B);
    free(h_C);
}

KernelFn g_kernels[] = {launch_cublas_ref,launch_naive_kernel,launch_smem_kernel,launch_blocktiling_kernel,
    launch_2Dblocktiling_kernel,launch_vectorized_kernel,
    launch_at<64,64,8,8,4>,launch_at<64,64,16,8,4>,launch_at<64,64,8,8,8>,
    launch_warptile_kernel,launch_warptile_vec_kernel,launch_bank_conflict_kernel,
    launch_double_buffer_kernel};
const char* g_names[] = {"cublas_ref","naive_kernel",
    "smem_kernel","blocktiling_kernel","Dblocktiling_kernel",
    "vectorized_kernel","autotune_64x64x8_8x4",
    "autotune_64x64x16_8x4","autotune_64x64x8_8x8","warptile_kernel","warptile_vec_kernel",
    "bank_conflict_kernel","double_buffer_kernel"};

int main(int argc,char** argv)
{
    int M = 1024;
    int N = 1024;
    int K = 1024;
    int id=0;
    int num_kernels = sizeof(g_kernels) / sizeof(g_kernels[0]);
    if(argc>=2 && strcmp(argv[1],"autotune")==0){
        int M=argc>2?atoi(argv[2]):4096;
        int N=argc>3?atoi(argv[3]):4096;
        int K=argc>4?atoi(argv[4]):4096;
        run_autotune(M,N,K);
        return 0;
    }
    else if (argc == 5) {
        id = atoi(argv[1]);
        M = atoi(argv[2]);
        N = atoi(argv[3]);
        K = atoi(argv[4]);
    } else if (argc != 1) {
        printf("Usage: %s [kernel_id M N K]\n", argv[0]);
        printf("Example: %s 4 1024 1024 1024\n", argv[0]);
        return 1;
    }
    if (id < 0 || id >= num_kernels) {
        for(int i=0;i<num_kernels;i++)
        {
            printf("idx:%d,kernel:%s\n",i,g_names[i]);
        }
        return 1;
    }

    int warmup_iters = 2;
    int repeat_iters = 10;

    benchmark_kernel(g_names[id],
                     g_kernels[id],
                     M, N, K,
                     warmup_iters,
                     repeat_iters);

    return 0;
}
