#include "../../include/common.h"


__global__ void bank_conflict_kernel(int M, int N, int K,
                             float alpha,
                             const float* A,
                             const float* B,
                             float beta,
                             float* C)
{
    constexpr int BM=64;
    constexpr int BK =8;
    constexpr int BN = 128;
    constexpr int WM = 32;
    constexpr int WN = 64;
    constexpr int TM = 8;
    constexpr int TN = 4;
    __shared__ float As[BK][BM+4];
    __shared__ float Bs[BK][BN];
    constexpr int WSUBM = 4*TM;
    constexpr int WSUBN = 8*TN;
    constexpr int WMITER = WM/WSUBM;
    constexpr int WNITER = WN/WSUBN;
    int tid = threadIdx.x;
    int warpIdx = tid / 32;
    int laneIdx = tid % 32;

    int warpRow = warpIdx / 2;
    int warpCol = warpIdx % 2;
    
    int laneRow = laneIdx / 8;
    int laneCol = laneIdx % 8;

    float acc[WMITER*WNITER*TM*TN] = {0};
    for (int bk = 0; bk < K; bk += BK) 
    {
        for(int load=0; load<4; load++){
            int idx = (tid + load*128);        // 0~511 全覆盖
            int as_r = idx / BK, as_c = idx % BK;   // As: 64×8
            // 算 global 地址、边界判断、填 As[as_r][as_c]
            int a_row = blockIdx.y * BM + as_r;
            int a_col = bk + as_c;
            As[as_c][as_r] = A[a_row*K + a_col];
        }
        for(int load=0; load<2; load++)
        {
            int idx = (tid + load*128)*4;  
            int bs_r = idx / BN, bs_c = idx % BN;
            int b_row = bk + bs_r;
            int b_col = blockIdx.x * BN + bs_c;
            float4 t = reinterpret_cast<const float4*>(&B[b_row*N + b_col])[0];
            reinterpret_cast<float4*>(&Bs[bs_r][bs_c])[0] = t;
        }
        __syncthreads();
        
        for(int k=0;k<BK;k++)
        {
            float regA[WMITER*TM];
            float regB[WNITER*TN];
            for (int wm = 0; wm < WMITER; wm++)
                for (int i = 0; i < TM; i++) {            // step 4 → step 1
                    int row = warpRow*WM + wm*WSUBM + (laneIdx/8)*TM + i;
                    regA[wm*TM+i] = As[k][row];           // 标量读
                }

                    
            for (int wn = 0; wn < WNITER; wn++)     // ③ 取 regA 的各 iter
                for (int j = 0; j < TN; j+=4){
                    int col = warpCol * WN + wn * WSUBN + (laneIdx % 8) * TN + j;
                    float4 tmp = reinterpret_cast<float4*>(&Bs[k][col])[0];
                    regB[wn*TN+j] = tmp.x;
                    regB[wn*TN+j+1] = tmp.y;
                    regB[wn*TN+j+2] = tmp.z;
                    regB[wn*TN+j+3] = tmp.w;

                }
            
            for (int wmiter = 0; wmiter < WMITER; wmiter++) {
                for (int wniter = 0; wniter < WNITER; wniter++) {
                    for (int tm = 0; tm < TM; tm++) {
                        for (int tn = 0; tn < TN; tn++) {
                            acc[((wmiter * WNITER + wniter) * TM + tm) * TN + tn] +=
                                regA[wmiter * TM + tm] *
                                regB[wniter * TN + tn];
                        }
                    }
                }
            }

        }
        __syncthreads();
    }
    int warp_row_base = warpRow * WM;
    int warp_col_base = warpCol * WN;

    for (int wmiter = 0; wmiter < WMITER; wmiter++) {
        for (int wniter = 0; wniter < WNITER; wniter++) {
            for (int tm = 0; tm < TM; tm++) {
                for (int tn = 0; tn < TN; tn++) {

                    int c_row = blockIdx.y * BM
                            + warp_row_base
                            + wmiter * WSUBM
                            + laneRow * TM
                            + tm;

                    int c_col = blockIdx.x * BN
                            + warp_col_base
                            + wniter * WSUBN
                            + laneCol * TN
                            + tn;

                    int acc_idx =
                        ((wmiter * WNITER + wniter) * TM + tm) * TN + tn;

                    if (c_row < M && c_col < N) {
                        C[c_row * N + c_col] =
                            alpha * acc[acc_idx] + beta * C[c_row * N + c_col];
                    }
                }
            }
        }
    }
}

void launch_bank_conflict_kernel(int M, int N, int K,
                         float alpha,
                         const float* A,
                         const float* B,
                         float beta,
                         float* C)
{
    constexpr int BM = 64;
    constexpr int BN = 128;

    dim3 block(128);
    dim3 grid(CEIL_DIV(N, BN), CEIL_DIV(M, BM), 1);

    bank_conflict_kernel<<<grid, block>>>(M, N, K, alpha, A, B, beta, C);

    CHECK_CUDA(cudaGetLastError());
}