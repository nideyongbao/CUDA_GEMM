// BF16 版全套 kernel：与对应 FP32 kernel 结构逐行一致，
// 仅把 A/B 的 global load 换成 bf16->float 转换；smem/寄存器/累加全程 FP32。
#include "../../include/bf16.h"

// ============================= 1. naive =============================
__global__ void naive_bf16_kernel(int M,int N,int K,float alpha,
        const __nv_bfloat16* A,const __nv_bfloat16* B,float beta,float* C){
    int row = threadIdx.y + blockDim.y*blockIdx.y;
    int col = threadIdx.x + blockDim.x*blockIdx.x;
    if(row<M && col<N){
        float temp=0;
        for(int k=0;k<K;k++)
            temp += __bfloat162float(A[row*K+k]) * __bfloat162float(B[N*k+col]);
        C[row*N+col] = alpha*temp + beta*C[row*N+col];
    }
}
void launch_naive_bf16(int M,int N,int K,float alpha,
        const __nv_bfloat16* A,const __nv_bfloat16* B,float beta,float* C){
    dim3 block(32,32,1);
    dim3 grid(CEIL_DIV(N,32),CEIL_DIV(M,32),1);
    naive_bf16_kernel<<<grid,block>>>(M,N,K,alpha,A,B,beta,C);
    CHECK_CUDA(cudaGetLastError());
}

// ============================= 2. smem =============================
__global__ void smem_bf16_kernel(int M,int N,int K,float alpha,
        const __nv_bfloat16* A,const __nv_bfloat16* B,float beta,float* C){
    __shared__ float As[32][32];
    __shared__ float Bs[32][32];
    int row = threadIdx.y + blockDim.y*blockIdx.y;
    int col = threadIdx.x + blockDim.x*blockIdx.x;
    int ty = threadIdx.y, tx = threadIdx.x;
    float temp=0;
    for(int bk=0;bk<K;bk+=32){
        As[ty][tx] = __bfloat162float(A[row*K+bk+tx]);
        Bs[ty][tx] = __bfloat162float(B[(bk+ty)*N+col]);
        __syncthreads();
        for(int k=0;k<32;k++) temp += As[ty][k] * Bs[k][tx];
        __syncthreads();
    }
    C[row*N+col] = alpha*temp + beta*C[row*N+col];
}
void launch_smem_bf16(int M,int N,int K,float alpha,
        const __nv_bfloat16* A,const __nv_bfloat16* B,float beta,float* C){
    dim3 block(32,32,1);
    dim3 grid(CEIL_DIV(N,32),CEIL_DIV(M,32),1);
    smem_bf16_kernel<<<grid,block>>>(M,N,K,alpha,A,B,beta,C);
    CHECK_CUDA(cudaGetLastError());
}

// ============================= 3. blocktiling (1D) =============================
__global__ void blocktiling_bf16_kernel(int M,int N,int K,float alpha,
        const __nv_bfloat16* A,const __nv_bfloat16* B,float beta,float* C){
    constexpr int BM=64,BK=8,BN=64,TM=8;
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];
    int tid = threadIdx.x + threadIdx.y*blockDim.x;
    float acc[TM]={0};
    for(int bk=0;bk<K;bk+=BK){
        int a_row = tid/BK + BM*blockIdx.y;
        int a_col = bk + tid%BK;
        int b_row = bk + tid/BN;
        int b_col = blockIdx.x*BN + tid%BN;
        As[tid/BK][tid%BK] = (a_row<M&&a_col<K)? __bfloat162float(A[a_row*K+a_col]) : 0.0f;
        Bs[tid/BN][tid%BN] = (b_row<K&&b_col<N)? __bfloat162float(B[b_row*N+b_col]) : 0.0f;
        __syncthreads();
        for(int k=0;k<BK;k++){
            float b = Bs[k][tid%BN];
            for(int i=0;i<TM;i++) acc[i]+=As[tid/BN*TM+i][k]*b;
        }
        __syncthreads();
    }
    int c_col=tid%64, c_row_block=tid/64;
    for(int i=0;i<TM;i++){
        int gr=blockIdx.y*BM + c_row_block*TM + i;
        int gc=blockIdx.x*BN + c_col;
        if(gr<M&&gc<N) C[gr*N+gc]=alpha*acc[i]+beta*C[gr*N+gc];
    }
}
void launch_blocktiling_bf16(int M,int N,int K,float alpha,
        const __nv_bfloat16* A,const __nv_bfloat16* B,float beta,float* C){
    dim3 block(64,8,1);
    dim3 grid(CEIL_DIV(N,64),CEIL_DIV(M,64),1);
    blocktiling_bf16_kernel<<<grid,block>>>(M,N,K,alpha,A,B,beta,C);
    CHECK_CUDA(cudaGetLastError());
}

// ============================= 4. 2D blocktiling =============================
__global__ void Dblocktiling_bf16_kernel(int M,int N,int K,float alpha,
        const __nv_bfloat16* A,const __nv_bfloat16* B,float beta,float* C){
    constexpr int BM=64,BK=8,BN=64,TM=8,TN=4;
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];
    int tid = threadIdx.x + threadIdx.y*blockDim.x;
    float acc[TM*TN]={0};
    for(int bk=0;bk<K;bk+=BK){
        for(int load=0;load<4;load++){
            int idx=tid+load*128;
            int as_r=idx/BK, as_c=idx%BK;
            int bs_r=idx/BN, bs_c=idx%BN;
            int a_row=blockIdx.y*BM+as_r, a_col=bk+as_c;
            int b_row=bk+bs_r, b_col=blockIdx.x*BN+bs_c;
            As[as_r][as_c]=(a_row<M&&a_col<K)? __bfloat162float(A[a_row*K+a_col]):0.0f;
            Bs[bs_r][bs_c]=(b_row<K&&b_col<N)? __bfloat162float(B[b_row*N+b_col]):0.0f;
        }
        __syncthreads();
        for(int k=0;k<BK;k++){
            float regA[TM],regB[TN];
            int row_in_tile=(tid/16)*TM, col_in_tile=(tid%16)*TN;
            for(int i=0;i<TM;i++) regA[i]=As[row_in_tile+i][k];
            for(int i=0;i<TN;i++) regB[i]=Bs[k][col_in_tile+i];
            for(int i=0;i<TN;i++) for(int j=0;j<TM;j++) acc[i*TM+j]+=regA[j]*regB[i];
        }
        __syncthreads();
    }
    int c_col_block=tid%16, c_row_block=tid/16;
    for(int i=0;i<TN;i++) for(int j=0;j<TM;j++){
        int gr=blockIdx.y*BM+c_row_block*TM+j;
        int gc=blockIdx.x*BN+c_col_block*TN+i;
        if(gr<M&&gc<N) C[gr*N+gc]=alpha*acc[i*TM+j]+beta*C[gr*N+gc];
    }
}
void launch_2Dblocktiling_bf16(int M,int N,int K,float alpha,
        const __nv_bfloat16* A,const __nv_bfloat16* B,float beta,float* C){
    dim3 block(16,8,1);
    dim3 grid(CEIL_DIV(N,64),CEIL_DIV(M,64),1);
    Dblocktiling_bf16_kernel<<<grid,block>>>(M,N,K,alpha,A,B,beta,C);
    CHECK_CUDA(cudaGetLastError());
}

// ============================= 5. vectorized =============================
__global__ void vectorized_bf16_kernel(int M,int N,int K,float alpha,
        const __nv_bfloat16* A,const __nv_bfloat16* B,float beta,float* C){
    constexpr int BM=64,BK=8,BN=64,TM=8,TN=4;
    __shared__ float As[BK][BM];
    __shared__ float Bs[BK][BN];
    int tid = threadIdx.x + threadIdx.y*blockDim.x;
    float acc[TM*TN]={0};
    for(int bk=0;bk<K;bk+=BK){
        int idx=tid*4;
        int as_r=idx/BK, as_c=idx%BK;
        int bs_r=idx/BN, bs_c=idx%BN;
        int a_row=blockIdx.y*BM+as_r, a_col=bk+as_c;
        int b_row=bk+bs_r, b_col=blockIdx.x*BN+bs_c;
        float a0,a1,a2,a3; ld4_bf16(&A[a_row*K+a_col],a0,a1,a2,a3);
        As[as_c][as_r]=a0; As[as_c+1][as_r]=a1; As[as_c+2][as_r]=a2; As[as_c+3][as_r]=a3;
        float b0,b1,b2,b3; ld4_bf16(&B[b_row*N+b_col],b0,b1,b2,b3);
        reinterpret_cast<float4*>(&Bs[bs_r][bs_c])[0]=make_float4(b0,b1,b2,b3);
        __syncthreads();
        for(int k=0;k<BK;k++){
            float regA[TM],regB[TN];
            int row_in_tile=(tid/16)*TM, col_in_tile=(tid%16)*TN;
            for(int i=0;i<TM;i+=4){
                float4 t=reinterpret_cast<float4*>(&As[k][row_in_tile+i])[0];
                regA[i]=t.x;regA[i+1]=t.y;regA[i+2]=t.z;regA[i+3]=t.w;
            }
            for(int i=0;i<TN;i+=4){
                float4 t=reinterpret_cast<float4*>(&Bs[k][col_in_tile+i])[0];
                regB[i]=t.x;regB[i+1]=t.y;regB[i+2]=t.z;regB[i+3]=t.w;
            }
            for(int i=0;i<TN;i++) for(int j=0;j<TM;j++) acc[i*TM+j]+=regA[j]*regB[i];
        }
        __syncthreads();
    }
    int c_col_block=tid%16, c_row_block=tid/16;
    for(int i=0;i<TN;i++) for(int j=0;j<TM;j++){
        int gr=blockIdx.y*BM+c_row_block*TM+j;
        int gc=blockIdx.x*BN+c_col_block*TN+i;
        if(gr<M&&gc<N) C[gr*N+gc]=alpha*acc[i*TM+j]+beta*C[gr*N+gc];
    }
}
void launch_vectorized_bf16(int M,int N,int K,float alpha,
        const __nv_bfloat16* A,const __nv_bfloat16* B,float beta,float* C){
    dim3 block(16,8,1);
    dim3 grid(CEIL_DIV(N,64),CEIL_DIV(M,64),1);
    vectorized_bf16_kernel<<<grid,block>>>(M,N,K,alpha,A,B,beta,C);
    CHECK_CUDA(cudaGetLastError());
}

// ============================= 9. warptile (标量) =============================
__global__ void warptile_bf16_kernel(int M,int N,int K,float alpha,
        const __nv_bfloat16* A,const __nv_bfloat16* B,float beta,float* C){
    constexpr int BM=128,BK=8,BN=128,WM=64,WN=64,TM=8,TN=4;
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];
    constexpr int WSUBM=4*TM, WSUBN=8*TN, WMITER=WM/WSUBM, WNITER=WN/WSUBN;
    int tid=threadIdx.x, warpIdx=tid/32, laneIdx=tid%32;
    int warpRow=warpIdx/2, warpCol=warpIdx%2, laneRow=laneIdx/8, laneCol=laneIdx%8;
    float acc[WMITER*WNITER*TM*TN]={0};
    for(int bk=0;bk<K;bk+=BK){
        for(int load=0;load<8;load++){
            int idx=tid+load*128;
            int as_r=idx/BK, as_c=idx%BK;
            int bs_r=idx/BN, bs_c=idx%BN;
            int a_row=blockIdx.y*BM+as_r, a_col=bk+as_c;
            int b_row=bk+bs_r, b_col=blockIdx.x*BN+bs_c;
            As[as_r][as_c]=(a_row<M&&a_col<K)? __bfloat162float(A[a_row*K+a_col]):0.0f;
            Bs[bs_r][bs_c]=(b_row<K&&b_col<N)? __bfloat162float(B[b_row*N+b_col]):0.0f;
        }
        __syncthreads();
        for(int k=0;k<BK;k++){
            float regA[WMITER*TM], regB[WNITER*TN];
            for(int wm=0;wm<WMITER;wm++) for(int i=0;i<TM;i++){
                int row=warpRow*WM+wm*WSUBM+(laneIdx/8)*TM+i; regA[wm*TM+i]=As[row][k];
            }
            for(int wn=0;wn<WNITER;wn++) for(int j=0;j<TN;j++){
                int col=warpCol*WN+wn*WSUBN+(laneIdx%8)*TN+j; regB[wn*TN+j]=Bs[k][col];
            }
            for(int wmi=0;wmi<WMITER;wmi++) for(int wni=0;wni<WNITER;wni++)
                for(int tm=0;tm<TM;tm++) for(int tn=0;tn<TN;tn++)
                    acc[((wmi*WNITER+wni)*TM+tm)*TN+tn]+=regA[wmi*TM+tm]*regB[wni*TN+tn];
        }
        __syncthreads();
    }
    int wrb=warpRow*WM, wcb=warpCol*WN;
    for(int wmi=0;wmi<WMITER;wmi++) for(int wni=0;wni<WNITER;wni++)
        for(int tm=0;tm<TM;tm++) for(int tn=0;tn<TN;tn++){
            int cr=blockIdx.y*BM+wrb+wmi*WSUBM+laneRow*TM+tm;
            int cc=blockIdx.x*BN+wcb+wni*WSUBN+laneCol*TN+tn;
            int ai=((wmi*WNITER+wni)*TM+tm)*TN+tn;
            if(cr<M&&cc<N) C[cr*N+cc]=alpha*acc[ai]+beta*C[cr*N+cc];
        }
}
void launch_warptile_bf16(int M,int N,int K,float alpha,
        const __nv_bfloat16* A,const __nv_bfloat16* B,float beta,float* C){
    dim3 block(128);
    dim3 grid(CEIL_DIV(N,128),CEIL_DIV(M,128),1);
    warptile_bf16_kernel<<<grid,block>>>(M,N,K,alpha,A,B,beta,C);
    CHECK_CUDA(cudaGetLastError());
}

// ============================= 10. warptile_vec =============================
__global__ void warptile_vec_bf16_kernel(int M,int N,int K,float alpha,
        const __nv_bfloat16* A,const __nv_bfloat16* B,float beta,float* C){
    constexpr int BM=64,BK=8,BN=128,WM=32,WN=64,TM=8,TN=4;
    __shared__ float As[BK][BM];
    __shared__ float Bs[BK][BN];
    constexpr int WSUBM=4*TM, WSUBN=8*TN, WMITER=WM/WSUBM, WNITER=WN/WSUBN;
    int tid=threadIdx.x, warpIdx=tid/32, laneIdx=tid%32;
    int warpRow=warpIdx/2, warpCol=warpIdx%2, laneRow=laneIdx/8, laneCol=laneIdx%8;
    float acc[WMITER*WNITER*TM*TN]={0};
    for(int bk=0;bk<K;bk+=BK){
        for(int load=0;load<1;load++){
            int idx=(tid+load*128)*4;
            int as_r=idx/BK, as_c=idx%BK;
            int a_row=blockIdx.y*BM+as_r, a_col=bk+as_c;
            float a0,a1,a2,a3; ld4_bf16(&A[a_row*K+a_col],a0,a1,a2,a3);
            As[as_c][as_r]=a0; As[as_c+1][as_r]=a1; As[as_c+2][as_r]=a2; As[as_c+3][as_r]=a3;
        }
        for(int load=0;load<2;load++){
            int idx=(tid+load*128)*4;
            int bs_r=idx/BN, bs_c=idx%BN;
            int b_row=bk+bs_r, b_col=blockIdx.x*BN+bs_c;
            float b0,b1,b2,b3; ld4_bf16(&B[b_row*N+b_col],b0,b1,b2,b3);
            reinterpret_cast<float4*>(&Bs[bs_r][bs_c])[0]=make_float4(b0,b1,b2,b3);
        }
        __syncthreads();
        for(int k=0;k<BK;k++){
            float regA[WMITER*TM], regB[WNITER*TN];
            for(int wm=0;wm<WMITER;wm++) for(int i=0;i<TM;i+=4){
                int row=warpRow*WM+wm*WSUBM+(laneIdx/8)*TM+i;
                float4 t=reinterpret_cast<float4*>(&As[k][row])[0];
                regA[wm*TM+i]=t.x;regA[wm*TM+i+1]=t.y;regA[wm*TM+i+2]=t.z;regA[wm*TM+i+3]=t.w;
            }
            for(int wn=0;wn<WNITER;wn++) for(int j=0;j<TN;j+=4){
                int col=warpCol*WN+wn*WSUBN+(laneIdx%8)*TN+j;
                float4 t=reinterpret_cast<float4*>(&Bs[k][col])[0];
                regB[wn*TN+j]=t.x;regB[wn*TN+j+1]=t.y;regB[wn*TN+j+2]=t.z;regB[wn*TN+j+3]=t.w;
            }
            for(int wmi=0;wmi<WMITER;wmi++) for(int wni=0;wni<WNITER;wni++)
                for(int tm=0;tm<TM;tm++) for(int tn=0;tn<TN;tn++)
                    acc[((wmi*WNITER+wni)*TM+tm)*TN+tn]+=regA[wmi*TM+tm]*regB[wni*TN+tn];
        }
        __syncthreads();
    }
    int wrb=warpRow*WM, wcb=warpCol*WN;
    for(int wmi=0;wmi<WMITER;wmi++) for(int wni=0;wni<WNITER;wni++)
        for(int tm=0;tm<TM;tm++) for(int tn=0;tn<TN;tn++){
            int cr=blockIdx.y*BM+wrb+wmi*WSUBM+laneRow*TM+tm;
            int cc=blockIdx.x*BN+wcb+wni*WSUBN+laneCol*TN+tn;
            int ai=((wmi*WNITER+wni)*TM+tm)*TN+tn;
            if(cr<M&&cc<N) C[cr*N+cc]=alpha*acc[ai]+beta*C[cr*N+cc];
        }
}
void launch_warptile_vec_bf16(int M,int N,int K,float alpha,
        const __nv_bfloat16* A,const __nv_bfloat16* B,float beta,float* C){
    dim3 block(128);
    dim3 grid(CEIL_DIV(N,128),CEIL_DIV(M,64),1);
    warptile_vec_bf16_kernel<<<grid,block>>>(M,N,K,alpha,A,B,beta,C);
    CHECK_CUDA(cudaGetLastError());
}

// ============================= 11. bank_conflict =============================
__global__ void bank_conflict_bf16_kernel(int M,int N,int K,float alpha,
        const __nv_bfloat16* A,const __nv_bfloat16* B,float beta,float* C){
    constexpr int BM=64,BK=8,BN=128,WM=32,WN=64,TM=8,TN=4;
    __shared__ float As[BK][BM+4];
    __shared__ float Bs[BK][BN];
    constexpr int WSUBM=4*TM, WSUBN=8*TN, WMITER=WM/WSUBM, WNITER=WN/WSUBN;
    int tid=threadIdx.x, warpIdx=tid/32, laneIdx=tid%32;
    int warpRow=warpIdx/2, warpCol=warpIdx%2, laneRow=laneIdx/8, laneCol=laneIdx%8;
    float acc[WMITER*WNITER*TM*TN]={0};
    for(int bk=0;bk<K;bk+=BK){
        for(int load=0;load<4;load++){
            int idx=tid+load*128;
            int as_r=idx/BK, as_c=idx%BK;
            int a_row=blockIdx.y*BM+as_r, a_col=bk+as_c;
            As[as_c][as_r]=__bfloat162float(A[a_row*K+a_col]);
        }
        for(int load=0;load<2;load++){
            int idx=(tid+load*128)*4;
            int bs_r=idx/BN, bs_c=idx%BN;
            int b_row=bk+bs_r, b_col=blockIdx.x*BN+bs_c;
            float b0,b1,b2,b3; ld4_bf16(&B[b_row*N+b_col],b0,b1,b2,b3);
            reinterpret_cast<float4*>(&Bs[bs_r][bs_c])[0]=make_float4(b0,b1,b2,b3);
        }
        __syncthreads();
        for(int k=0;k<BK;k++){
            float regA[WMITER*TM], regB[WNITER*TN];
            for(int wm=0;wm<WMITER;wm++) for(int i=0;i<TM;i++){
                int row=warpRow*WM+wm*WSUBM+(laneIdx/8)*TM+i; regA[wm*TM+i]=As[k][row];
            }
            for(int wn=0;wn<WNITER;wn++) for(int j=0;j<TN;j+=4){
                int col=warpCol*WN+wn*WSUBN+(laneIdx%8)*TN+j;
                float4 t=reinterpret_cast<float4*>(&Bs[k][col])[0];
                regB[wn*TN+j]=t.x;regB[wn*TN+j+1]=t.y;regB[wn*TN+j+2]=t.z;regB[wn*TN+j+3]=t.w;
            }
            for(int wmi=0;wmi<WMITER;wmi++) for(int wni=0;wni<WNITER;wni++)
                for(int tm=0;tm<TM;tm++) for(int tn=0;tn<TN;tn++)
                    acc[((wmi*WNITER+wni)*TM+tm)*TN+tn]+=regA[wmi*TM+tm]*regB[wni*TN+tn];
        }
        __syncthreads();
    }
    int wrb=warpRow*WM, wcb=warpCol*WN;
    for(int wmi=0;wmi<WMITER;wmi++) for(int wni=0;wni<WNITER;wni++)
        for(int tm=0;tm<TM;tm++) for(int tn=0;tn<TN;tn++){
            int cr=blockIdx.y*BM+wrb+wmi*WSUBM+laneRow*TM+tm;
            int cc=blockIdx.x*BN+wcb+wni*WSUBN+laneCol*TN+tn;
            int ai=((wmi*WNITER+wni)*TM+tm)*TN+tn;
            if(cr<M&&cc<N) C[cr*N+cc]=alpha*acc[ai]+beta*C[cr*N+cc];
        }
}
void launch_bank_conflict_bf16(int M,int N,int K,float alpha,
        const __nv_bfloat16* A,const __nv_bfloat16* B,float beta,float* C){
    dim3 block(128);
    dim3 grid(CEIL_DIV(N,128),CEIL_DIV(M,64),1);
    bank_conflict_bf16_kernel<<<grid,block>>>(M,N,K,alpha,A,B,beta,C);
    CHECK_CUDA(cudaGetLastError());
}

// ============================= 12. double_buffer =============================
__global__ void double_buffer_bf16_kernel(int M,int N,int K,float alpha,
        const __nv_bfloat16* A,const __nv_bfloat16* B,float beta,float* C){
    constexpr int BM=64,BK=8,BN=128,WM=32,WN=64,TM=8,TN=4;
    int cur=0;
    __shared__ float As[2][BK][BM+4];
    __shared__ float Bs[2][BK][BN];
    constexpr int WSUBM=4*TM, WSUBN=8*TN, WMITER=WM/WSUBM, WNITER=WN/WSUBN;
    int tid=threadIdx.x, warpIdx=tid/32, laneIdx=tid%32;
    int warpRow=warpIdx/2, warpCol=warpIdx%2, laneRow=laneIdx/8, laneCol=laneIdx%8;

    for(int load=0;load<4;load++){
        int idx=tid+load*128; int as_r=idx/BK, as_c=idx%BK;
        int a_row=blockIdx.y*BM+as_r, a_col=0+as_c;
        As[cur][as_c][as_r]=__bfloat162float(A[a_row*K+a_col]);
    }
    for(int load=0;load<2;load++){
        int idx=(tid+load*128)*4; int bs_r=idx/BN, bs_c=idx%BN;
        int b_row=0+bs_r, b_col=blockIdx.x*BN+bs_c;
        float b0,b1,b2,b3; ld4_bf16(&B[b_row*N+b_col],b0,b1,b2,b3);
        reinterpret_cast<float4*>(&Bs[cur][bs_r][bs_c])[0]=make_float4(b0,b1,b2,b3);
    }
    __syncthreads();

    float acc[WMITER*WNITER*TM*TN]={0};
    for(int bk=0;bk<K-BK;bk+=BK){
        int next_bk=bk+BK;
        for(int load=0;load<4;load++){
            int idx=tid+load*128; int as_r=idx/BK, as_c=idx%BK;
            int a_row=blockIdx.y*BM+as_r, a_col=next_bk+as_c;
            As[cur^1][as_c][as_r]=__bfloat162float(A[a_row*K+a_col]);
        }
        for(int load=0;load<2;load++){
            int idx=(tid+load*128)*4; int bs_r=idx/BN, bs_c=idx%BN;
            int b_row=next_bk+bs_r, b_col=blockIdx.x*BN+bs_c;
            float b0,b1,b2,b3; ld4_bf16(&B[b_row*N+b_col],b0,b1,b2,b3);
            reinterpret_cast<float4*>(&Bs[cur^1][bs_r][bs_c])[0]=make_float4(b0,b1,b2,b3);
        }
        for(int k=0;k<BK;k++){
            float regA[WMITER*TM], regB[WNITER*TN];
            for(int wm=0;wm<WMITER;wm++) for(int i=0;i<TM;i++){
                int row=warpRow*WM+wm*WSUBM+(laneIdx/8)*TM+i; regA[wm*TM+i]=As[cur][k][row];
            }
            for(int wn=0;wn<WNITER;wn++) for(int j=0;j<TN;j+=4){
                int col=warpCol*WN+wn*WSUBN+(laneIdx%8)*TN+j;
                float4 t=reinterpret_cast<float4*>(&Bs[cur][k][col])[0];
                regB[wn*TN+j]=t.x;regB[wn*TN+j+1]=t.y;regB[wn*TN+j+2]=t.z;regB[wn*TN+j+3]=t.w;
            }
            for(int wmi=0;wmi<WMITER;wmi++) for(int wni=0;wni<WNITER;wni++)
                for(int tm=0;tm<TM;tm++) for(int tn=0;tn<TN;tn++)
                    acc[((wmi*WNITER+wni)*TM+tm)*TN+tn]+=regA[wmi*TM+tm]*regB[wni*TN+tn];
        }
        __syncthreads();
        cur=cur^1;
    }
    for(int k=0;k<BK;k++){
        float regA[WMITER*TM], regB[WNITER*TN];
        for(int wm=0;wm<WMITER;wm++) for(int i=0;i<TM;i++){
            int row=warpRow*WM+wm*WSUBM+(laneIdx/8)*TM+i; regA[wm*TM+i]=As[cur][k][row];
        }
        for(int wn=0;wn<WNITER;wn++) for(int j=0;j<TN;j+=4){
            int col=warpCol*WN+wn*WSUBN+(laneIdx%8)*TN+j;
            float4 t=reinterpret_cast<float4*>(&Bs[cur][k][col])[0];
            regB[wn*TN+j]=t.x;regB[wn*TN+j+1]=t.y;regB[wn*TN+j+2]=t.z;regB[wn*TN+j+3]=t.w;
        }
        for(int wmi=0;wmi<WMITER;wmi++) for(int wni=0;wni<WNITER;wni++)
            for(int tm=0;tm<TM;tm++) for(int tn=0;tn<TN;tn++)
                acc[((wmi*WNITER+wni)*TM+tm)*TN+tn]+=regA[wmi*TM+tm]*regB[wni*TN+tn];
    }
    int wrb=warpRow*WM, wcb=warpCol*WN;
    for(int wmi=0;wmi<WMITER;wmi++) for(int wni=0;wni<WNITER;wni++)
        for(int tm=0;tm<TM;tm++) for(int tn=0;tn<TN;tn++){
            int cr=blockIdx.y*BM+wrb+wmi*WSUBM+laneRow*TM+tm;
            int cc=blockIdx.x*BN+wcb+wni*WSUBN+laneCol*TN+tn;
            int ai=((wmi*WNITER+wni)*TM+tm)*TN+tn;
            if(cr<M&&cc<N) C[cr*N+cc]=alpha*acc[ai]+beta*C[cr*N+cc];
        }
}
void launch_double_buffer_bf16(int M,int N,int K,float alpha,
        const __nv_bfloat16* A,const __nv_bfloat16* B,float beta,float* C){
    dim3 block(128);
    dim3 grid(CEIL_DIV(N,128),CEIL_DIV(M,64),1);
    double_buffer_bf16_kernel<<<grid,block>>>(M,N,K,alpha,A,B,beta,C);
    CHECK_CUDA(cudaGetLastError());
}
