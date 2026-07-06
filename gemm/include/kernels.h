#pragma once
void launch_naive_kernel(int M, int N, int K, float alpha,
                         const float* A, const float* B,
                         float beta, float* C);

void launch_smem_kernel(int M, int N, int K, float alpha,
                         const float* A, const float* B,
                         float beta, float* C);


void launch_blocktiling_kernel(int M, int N, int K,
                         float alpha,
                         const float* A,
                         const float* B,
                         float beta,
                         float* C);

void launch_2Dblocktiling_kernel(int M, int N, int K,
                         float alpha,
                         const float* A,
                         const float* B,
                         float beta,
                         float* C);

void launch_vectorized_kernel(int M, int N, int K,
                         float alpha,
                         const float* A,
                         const float* B,
                         float beta,
                         float* C);
            
void launch_warptile_kernel(int M, int N, int K,
                         float alpha,
                         const float* A,
                         const float* B,
                         float beta,
                         float* C);
                        
void launch_warptile_vec_kernel(int M, int N, int K,
                         float alpha,
                         const float* A,
                         const float* B,
                         float beta,
                         float* C);

void launch_bank_conflict_kernel(int M, int N, int K,
                         float alpha,
                         const float* A,
                         const float* B,
                         float beta,
                         float* C);

void launch_double_buffer_kernel(int M, int N, int K,
                         float alpha,
                         const float* A,
                         const float* B,
                         float beta,
                         float* C);


                         