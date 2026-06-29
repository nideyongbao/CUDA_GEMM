#include "../../include/common.h"
#include "../../include/kernels.h"
#include <cstdio>
#include <cstdlib>
#include "../../include/06_autotuning.cuh"
#include <math.h>
#include <cublas_v2.h>

void launch_cublas_ref(int M, int N, int K,
                       float alpha,
                       const float* A,
                       const float* B,
                       float beta,
                       float* C)
{
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    CHECK_CUBLAS(cublasSgemm(handle,
                             CUBLAS_OP_N,
                             CUBLAS_OP_N,
                             N, M, K,
                             &alpha,
                             B, N,
                             A, K,
                             &beta,
                             C, N));

    CHECK_CUBLAS(cublasDestroy(handle));
}

void init_matrix(float* mat, int size)
{
    for (int i = 0; i < size; i++) {
        float r = (float)rand() / (float)RAND_MAX;
        mat[i] = r * 2.0f - 1.0f;   // [-1, 1]
    }
}

static int verify_result(const float* mine,
                         const float* ref,
                         int M,
                         int N)
{
    const float abs_eps = 1e-2f;
    const float rel_eps = 1e-2f;

    float max_abs_err = 0.0f;
    float max_rel_err = 0.0f;

    int bad_count = 0;
    int bad_i = -1;
    int bad_j = -1;

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            int idx = i * N + j;

            float a = mine[idx];
            float b = ref[idx];

            float abs_err = fabsf(a - b);
            float rel_err = abs_err / (fabsf(b) + 1e-5f);

            if (abs_err > max_abs_err) {
                max_abs_err = abs_err;
            }

            if (rel_err > max_rel_err) {
                max_rel_err = rel_err;
            }

            // numpy allclose 风格：|a-b| <= atol + rtol*|b| 才算通过。
            // 随机 [-1,1] 数据下部分 C 元素接近 0，单独看相对误差会被放大成假 FAIL；
            // 合并判据避免这种误报（kernel 已用 CPU double 参考独立验证为正确）。
            if (abs_err > abs_eps + rel_eps * fabsf(b)) {
                bad_count++;

                if (bad_i == -1) {
                    bad_i = i;
                    bad_j = j;
                }
            }
        }
    }

    printf("max_abs_err = %.6e\n", max_abs_err);
    printf("max_rel_err = %.6e\n", max_rel_err);
    printf("bad_count   = %d / %d\n", bad_count, M * N);

    if (bad_count > 0) {
        int idx = bad_i * N + bad_j;
        printf("first bad at C[%d][%d]: mine = %.6f, ref = %.6f\n",
               bad_i, bad_j, mine[idx], ref[idx]);
    }

    return bad_count == 0;
}



void verify_kernel (const char* name,
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
    float* h_C_ref = (float*)malloc(bytes_C);

    if (!h_A || !h_B || !h_C) {
        printf("Host malloc failed\n");
        exit(1);
    }

    init_matrix(h_A, M * K);
    init_matrix(h_B, K * N);
    init_matrix(h_C, M * N);

    memcpy(h_C_ref, h_C, bytes_C);

    float* d_A = nullptr;
    float* d_B = nullptr;
    float* d_C = nullptr;

    float* d_A_ref = nullptr;
    float* d_B_ref = nullptr;
    float* d_C_ref = nullptr;

    CHECK_CUDA(cudaMalloc(&d_A, bytes_A));
    CHECK_CUDA(cudaMalloc(&d_B, bytes_B));
    CHECK_CUDA(cudaMalloc(&d_C, bytes_C));

    CHECK_CUDA(cudaMalloc(&d_A_ref, bytes_A));
    CHECK_CUDA(cudaMalloc(&d_B_ref, bytes_B));
    CHECK_CUDA(cudaMalloc(&d_C_ref, bytes_C));

    CHECK_CUDA(cudaMemcpy(d_A, h_A, bytes_A, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, bytes_B, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_C, h_C, bytes_C, cudaMemcpyHostToDevice));

    CHECK_CUDA(cudaMemcpy(d_A_ref, h_A, bytes_A, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B_ref, h_B, bytes_B, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_C_ref, h_C, bytes_C, cudaMemcpyHostToDevice));



    kernel(M, N, K, alpha, d_A, d_B, beta, d_C);
    CHECK_CUDA(cudaDeviceSynchronize());

    // 跑 cuBLAS reference
    launch_cublas_ref(M, N, K, alpha, d_A, d_B, beta, d_C_ref);
    CHECK_CUDA(cudaDeviceSynchronize());

    // 拷回
    CHECK_CUDA(cudaMemcpy(h_C,     d_C,     bytes_C, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_C_ref, d_C_ref, bytes_C, cudaMemcpyDeviceToHost));

    // 比较
    int ok = verify_result(h_C, h_C_ref, M, N);
    printf(ok ? "PASS\n" : "FAIL\n");

    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));

    CHECK_CUDA(cudaFree(d_A_ref));
    CHECK_CUDA(cudaFree(d_B_ref));
    CHECK_CUDA(cudaFree(d_C_ref));

    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_ref);
}

KernelFn g_kernels[] = {launch_cublas_ref,launch_naive_kernel,
    launch_smem_kernel,launch_blocktiling_kernel,
    launch_2Dblocktiling_kernel,launch_vectorized_kernel,
    launch_at<64,64,8,8,4>,launch_at<64,64,16,8,4>,
    launch_at<64,64,8,8,8>,launch_warptile_kernel,
    launch_warptile_vec_kernel,launch_bank_conflict_kernel,
    launch_double_buffer_kernel};
const char* g_names[] = {"cublas_ref","naive_kernel",
    "smem_kernel","blocktiling_kernel","Dblocktiling_kernel",
    "vectorized_kernel","autotune_64x64x8_8x4",
    "autotune_64x64x16_8x4","autotune_64x64x8_8x8","warptile_kernel",
    "warptile_vec_kernel","bank_conflict_kernel","double_buffer_kernel"};

int main(int argc,char** argv)
{
    int M = 1024;
    int N = 1024;
    int K = 1024;
    int id=0;
    int num_kernels = sizeof(g_kernels) / sizeof(g_kernels[0]);
    if (argc == 5) {
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

    verify_kernel(g_names[id],
                     g_kernels[id],
                     M, N, K,
                     warmup_iters,
                     repeat_iters);

    return 0;
}
