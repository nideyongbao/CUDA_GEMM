// ============================================================================
// fa_hopper.cuh — hand-written Hopper (sm_90a) flash-attention forward.
//
// The tensor-core end-state for H20: FlashAttention-2 math (two GEMMs fused with
// an online softmax, never materialising the S=QK^T matrix in HBM) implemented on
// Hopper's *native* async tensor-core path:
//   · TMA  (cp.async.bulk.tensor) streams K/V tiles global->smem (128B swizzle).
//   · WGMMA (wgmma.mma_async, warpgroup-wide) does BOTH matmuls:
//        S = Q·Kᵀ  (operand A from smem, operand B from smem)         [n = Bc]
//        O = P·V   (operand A from *registers* = softmax(P), B from smem)[n = D]
//   · online softmax lives entirely in registers; the row reductions ride the
//     WGMMA accumulator layout via 4-lane warp shuffles (no smem round-trip).
//
// The register-resident P->A handoff is the crux that makes FA fast on tensor
// cores: the m64nNk16 WGMMA *accumulator* layout of matmul-1 is bit-for-bit the
// *A-operand* fragment layout of matmul-2, so softmax(S) is fed straight back in
// as the A registers of P·V — no shared-memory bounce for P.
//
// Geometry is pinned to the 128B-swizzle "inner-64" core matrix (same descriptor
// as gemm/tc_04): every smem operand is 64 elements wide in its contiguous dim,
// so head dim D is streamed in 64-wide chunks (D ∈ {64,128}). One warpgroup
// (128 threads) owns one Br=64 query tile; grid = (S/Br, H, B).
//
// Tolerance vs fp32 CPU attention: atol/rtol ~2e-2 (bf16/fp16 tensor core).
// ============================================================================
#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda.h>
#include <cuda/barrier>
#include <cstdint>
#include <cstdio>
#include "fa_common.h"

namespace fah {

#define FAH_WG 128                 // one warpgroup owns a query tile
#define FAH_BR 64                  // query rows per block  (WGMMA m = 64)
#define FAH_BC 64                  // key/value rows per KV tile (WGMMA n1 = 64)
#define DEVI __device__ __forceinline__

using barrier = cuda::barrier<cuda::thread_scope_block>;
namespace cde = cuda::device::experimental;

#define WGMMA_FENCE()        asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory")
#define WGMMA_COMMIT()       asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory")
#define WGMMA_WAIT(n)        asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(n) : "memory")

// ---- 128B-swizzle smem matrix descriptor (leading=16B, stride=1024B, inner-64) ----
// Identical magic to gemm/tc_04: valid for any #rows as long as the contiguous
// dim is 64 bf16 (=128B = one swizzle atom) and the tile came from a 128B-swizzle TMA.
#define SMEM_DESC_ENC(x) ((((uint64_t)(x)) & 0x3FFFF) >> 0x4)
template<class BF>
DEVI uint64_t smem_desc(BF* ptr){
    uint32_t a = (uint32_t)__cvta_generic_to_shared(ptr);
    uint64_t d = 0;
    d |= SMEM_DESC_ENC(a);
    d |= SMEM_DESC_ENC((uint64_t)16)   << 16;
    d |= SMEM_DESC_ENC((uint64_t)1024) << 32;
    d |= 1llu << 62;   // 128B swizzle
    return d;
}

// ---- WGMMA m64n64k16, f32 accumulate, 16-bit inputs. 32 accumulators = d[4][8]. ----
// SS = both operands from smem (used for S=Q·Kᵀ).
// RS = operand A from registers {a0..a3}, B from smem (used for O=P·V).
// One macro per input dtype (bf16 / f16); the register layout is identical.
#define OUT32(d) \
   "+f"((d)[0][0]),"+f"((d)[0][1]),"+f"((d)[0][2]),"+f"((d)[0][3]),\
   "+f"((d)[0][4]),"+f"((d)[0][5]),"+f"((d)[0][6]),"+f"((d)[0][7]),\
   "+f"((d)[1][0]),"+f"((d)[1][1]),"+f"((d)[1][2]),"+f"((d)[1][3]),\
   "+f"((d)[1][4]),"+f"((d)[1][5]),"+f"((d)[1][6]),"+f"((d)[1][7]),\
   "+f"((d)[2][0]),"+f"((d)[2][1]),"+f"((d)[2][2]),"+f"((d)[2][3]),\
   "+f"((d)[2][4]),"+f"((d)[2][5]),"+f"((d)[2][6]),"+f"((d)[2][7]),\
   "+f"((d)[3][0]),"+f"((d)[3][1]),"+f"((d)[3][2]),"+f"((d)[3][3]),\
   "+f"((d)[3][4]),"+f"((d)[3][5]),"+f"((d)[3][6]),"+f"((d)[3][7])
#define OUTIDX \
   "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"\
   "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}"

// S=Q·Kᵀ : both operands from smem, accumulate (scaleD=1), no transpose.
// -- bf16 --
DEVI void wgmma64_ss_bf16(float d[4][8], uint64_t da, uint64_t db){
    asm volatile("{\n wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "
      OUTIDX ", %32, %33, 1, 1, 1, 0, 0;\n}\n" : OUT32(d) : "l"(da),"l"(db));
}
// O=P·V : A from registers {a0..a3}, B from smem, accumulate; TB templated (V transpose).
template<int TB>
DEVI void wgmma64_rs_bf16(float d[4][8], uint32_t a0,uint32_t a1,uint32_t a2,uint32_t a3, uint64_t db){
    asm volatile("{\n wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "
      OUTIDX ", {%32,%33,%34,%35}, %36, 1, 1, 1, %37;\n}\n"
      : OUT32(d) : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"l"(db),"n"(TB));
}
// -- f16 --
DEVI void wgmma64_ss_f16(float d[4][8], uint64_t da, uint64_t db){
    asm volatile("{\n wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16 "
      OUTIDX ", %32, %33, 1, 1, 1, 0, 0;\n}\n" : OUT32(d) : "l"(da),"l"(db));
}
template<int TB>
DEVI void wgmma64_rs_f16(float d[4][8], uint32_t a0,uint32_t a1,uint32_t a2,uint32_t a3, uint64_t db){
    asm volatile("{\n wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16 "
      OUTIDX ", {%32,%33,%34,%35}, %36, 1, 1, 1, %37;\n}\n"
      : OUT32(d) : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"l"(db),"n"(TB));
}

// V-operand transpose immediate for P·V (empirically fixed against CPU reference).
#ifndef FAH_PV_TRANSB
#define FAH_PV_TRANSB 1
#endif

template<class T> struct WG;
template<> struct WG<__nv_bfloat16>{
    static DEVI void ss(float d[4][8],uint64_t a,uint64_t b){ wgmma64_ss_bf16(d,a,b);}
    static DEVI void rs(float d[4][8],uint32_t a0,uint32_t a1,uint32_t a2,uint32_t a3,uint64_t b){ wgmma64_rs_bf16<FAH_PV_TRANSB>(d,a0,a1,a2,a3,b);}
    static DEVI uint32_t pack(float x,float y){ __nv_bfloat162 v=__floats2bfloat162_rn(x,y); uint32_t r; memcpy(&r,&v,4); return r; }
};
template<> struct WG<__half>{
    static DEVI void ss(float d[4][8],uint64_t a,uint64_t b){ wgmma64_ss_f16(d,a,b);}
    static DEVI void rs(float d[4][8],uint32_t a0,uint32_t a1,uint32_t a2,uint32_t a3,uint64_t b){ wgmma64_rs_f16<FAH_PV_TRANSB>(d,a0,a1,a2,a3,b);}
    static DEVI uint32_t pack(float x,float y){ __half2 v=__floats2half2_rn(x,y); uint32_t r; memcpy(&r,&v,4); return r; }
};

// row of an m64 WGMMA accumulator/operand held by this lane (two rows: rr, rr+8)
DEVI int wg_row0(int tid){ int warp=tid/32, lane=tid%32; return warp*16 + lane/4; }

// ============================================================================
// Kernel: one warpgroup, one Br=64 query tile, head dim D (streamed in 64-chunks).
// smem holds Q (persistent) + double-buffered K,V tiles; all 128B-swizzled by TMA.
// ============================================================================
// Single grid over (qtile, h, b) = (blockIdx.x, .y, .z). One warpgroup per query tile.
// tmaQ/K/V are device arrays of B*H maps (one 2D [S,D] map per (b,h)); O is the full
// [B,S,H,D] tensor. This one-launch layout lets causal's lighter blocks finish early.
template<class T, int D>
__global__ __launch_bounds__(FAH_WG) void fa_kernel(
        int S, int H, bool causal, float scale,
        const CUtensorMap* __restrict__ tmaQ,
        const CUtensorMap* __restrict__ tmaK,
        const CUtensorMap* __restrict__ tmaV,
        T* __restrict__ O){
    constexpr int NC = D/64;           // # of 64-wide head-dim chunks (1 or 2)
    constexpr int KP = FAH_BC/16;      // k16 steps for P·V  (contract Bc=64 -> 4)
    const int b = blockIdx.z, h = blockIdx.y, qtile = blockIdx.x;
    const int bh = b*H + h;
    const int q0 = qtile*FAH_BR;       // first query row (within this b,h)
    if (q0 >= S) return;
    const int tid = threadIdx.x;
    const int ostride = H*D;
    T* O_bh = O + ((size_t)b*S*H + h)*D;

    // ---- shared memory: Q[Br,D], K/V double buffered [Bc,D] ----
    extern __shared__ __align__(128) uint8_t smem[];
    T* sQ = reinterpret_cast<T*>(smem);
    T* sK = sQ + FAH_BR*D;
    T* sV = sK + 2*FAH_BC*D;           // 2 = double buffer
    const int TILE_KV = FAH_BC*D;
    #pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ barrier bar[2];
    if (tid==0){ init(&bar[0], FAH_WG); init(&bar[1], FAH_WG); cde::fence_proxy_async_shared_cta(); }
    __syncthreads();

    // ---- load Q tile once (coord: col=chunk*64, row=q0) ----
    {
        barrier::arrival_token t;
        if (tid==0){
            for (int c=0;c<NC;++c)
                cde::cp_async_bulk_tensor_2d_global_to_shared(sQ+c*FAH_BR*64, &tmaQ[bh], c*64, q0, bar[0]);
            t = cuda::device::barrier_arrive_tx(bar[0], 1, FAH_BR*D*sizeof(T));
        } else t = bar[0].arrive();
        bar[0].wait(std::move(t));
    }
    __syncthreads();

    // ---- online-softmax running state (per the 2 rows this lane owns) ----
    float m_i[2] = {-INFINITY,-INFINITY};   // running row max
    float l_i[2] = {0.f,0.f};               // running row sum
    float od[NC][4][8];                      // O accumulator: NC D-chunks, each n64
    #pragma unroll
    for (int hh=0; hh<NC; ++hh)
      #pragma unroll
      for (int g=0; g<4; ++g)
        #pragma unroll
        for (int e=0;e<8;++e) od[hh][g][e]=0.f;

    const int rr = wg_row0(tid);             // this lane's two query rows: rr, rr+8
    const int qrow0 = q0+rr, qrow1 = q0+rr+8;
    const int nkv = (S + FAH_BC - 1)/FAH_BC;
    const int kv_last = causal ? (q0+FAH_BR-1)/FAH_BC : nkv-1;   // causal: skip strictly-future tiles

    for (int kv=0, buf=0; kv<=kv_last; ++kv, buf^=1){
        // ---- TMA K,V tile [Bc,D] at key row kv*Bc ----
        T* pK = sK + buf*TILE_KV;
        T* pV = sV + buf*TILE_KV;
        barrier::arrival_token t;
        if (tid==0){
            for (int c=0;c<NC;++c){
                cde::cp_async_bulk_tensor_2d_global_to_shared(pK+c*FAH_BC*64, &tmaK[bh], c*64, kv*FAH_BC, bar[buf]);
                cde::cp_async_bulk_tensor_2d_global_to_shared(pV+c*FAH_BC*64, &tmaV[bh], c*64, kv*FAH_BC, bar[buf]);
            }
            t = cuda::device::barrier_arrive_tx(bar[buf], 1, 2*TILE_KV*sizeof(T));
        } else t = bar[buf].arrive();
        bar[buf].wait(std::move(t));
        __syncthreads();

        // ---- S = scale · Q·Kᵀ   (WGMMA smem×smem, n=Bc=64) ----
        float sd[4][8];
        #pragma unroll
        for (int g=0;g<4;++g)
            #pragma unroll
            for (int e=0;e<8;++e) sd[g][e]=0.f;
        WGMMA_FENCE();
        #pragma unroll
        for (int c=0;c<NC;++c){
            T* qc = sQ + c*FAH_BR*64;
            T* kc = pK + c*FAH_BC*64;
            #pragma unroll
            for (int k=0;k<4;++k)   // 4 k16 steps within a 64-chunk
                WG<T>::ss(sd, smem_desc(qc + k*16), smem_desc(kc + k*16));
        }
        WGMMA_COMMIT(); WGMMA_WAIT(0);

        // ---- online softmax over the Bc=64 columns of S (rows rr, rr+8) ----
        // this lane holds, per row, columns {g*16 + 2*(lane%4) + {0,1,8,9} : g=0..3}
        const int lane = tid%32;
        const int col_q = 2*(lane%4);
        // apply scale + causal mask, track tile max for the 2 rows
        float mtile[2] = {-INFINITY,-INFINITY};
        #pragma unroll
        for (int g=0; g<4; ++g){
            int cbase = g*16 + col_q;
            int cs[4] = {cbase, cbase+1, cbase+8, cbase+9};
            // sd layout: [0,1,4,5]=row rr ; [2,3,6,7]=row rr+8
            float* r0[4] = {&sd[g][0],&sd[g][1],&sd[g][4],&sd[g][5]};
            float* r1[4] = {&sd[g][2],&sd[g][3],&sd[g][6],&sd[g][7]};
            #pragma unroll
            for (int e=0;e<4;++e){
                int kj = kv*FAH_BC + cs[e];
                *r0[e] *= scale; *r1[e] *= scale;
                if (kj>=S || (causal && kj>qrow0)) *r0[e] = -INFINITY;
                if (kj>=S || (causal && kj>qrow1)) *r1[e] = -INFINITY;
                mtile[0] = fmaxf(mtile[0], *r0[e]);
                mtile[1] = fmaxf(mtile[1], *r1[e]);
            }
        }
        // reduce tile-max across the 4 lanes sharing each row (lane%4 group)
        #pragma unroll
        for (int off=1; off<4; off<<=1){
            mtile[0] = fmaxf(mtile[0], __shfl_xor_sync(0xffffffff, mtile[0], off));
            mtile[1] = fmaxf(mtile[1], __shfl_xor_sync(0xffffffff, mtile[1], off));
        }
        // online update + build P (bf16) directly in WGMMA A-fragment order
        float m_new[2] = { fmaxf(m_i[0],mtile[0]), fmaxf(m_i[1],mtile[1]) };
        float corr[2]  = { __expf(m_i[0]-m_new[0]), __expf(m_i[1]-m_new[1]) };
        if (m_i[0]==-INFINITY) corr[0]=1.f;
        if (m_i[1]==-INFINITY) corr[1]=1.f;
        uint32_t pk[4][4];   // [g][ reg0..3 ] : A-fragment for P·V k16-chunk g
        float lsum[2]={0.f,0.f};
        #pragma unroll
        for (int g=0; g<4; ++g){
            float p0a=__expf(sd[g][0]-m_new[0]), p0b=__expf(sd[g][1]-m_new[0]);
            float p1a=__expf(sd[g][2]-m_new[1]), p1b=__expf(sd[g][3]-m_new[1]);
            float p0c=__expf(sd[g][4]-m_new[0]), p0d=__expf(sd[g][5]-m_new[0]);
            float p1c=__expf(sd[g][6]-m_new[1]), p1d=__expf(sd[g][7]-m_new[1]);
            lsum[0]+= p0a+p0b+p0c+p0d;  lsum[1]+= p1a+p1b+p1c+p1d;
            pk[g][0]=WG<T>::pack(p0a,p0b); pk[g][1]=WG<T>::pack(p1a,p1b);
            pk[g][2]=WG<T>::pack(p0c,p0d); pk[g][3]=WG<T>::pack(p1c,p1d);
        }
        #pragma unroll
        for (int off=1; off<4; off<<=1){
            lsum[0]+=__shfl_xor_sync(0xffffffff, lsum[0], off);
            lsum[1]+=__shfl_xor_sync(0xffffffff, lsum[1], off);
        }
        l_i[0] = l_i[0]*corr[0] + lsum[0];
        l_i[1] = l_i[1]*corr[1] + lsum[1];
        // rescale O accumulator by corr (rows: od[*][*][0,1,4,5]=rr ; [2,3,6,7]=rr+8)
        #pragma unroll
        for (int hh=0; hh<NC; ++hh)
        #pragma unroll
        for (int g=0; g<4; ++g){
            od[hh][g][0]*=corr[0]; od[hh][g][1]*=corr[0]; od[hh][g][4]*=corr[0]; od[hh][g][5]*=corr[0];
            od[hh][g][2]*=corr[1]; od[hh][g][3]*=corr[1]; od[hh][g][6]*=corr[1]; od[hh][g][7]*=corr[1];
        }
        m_i[0]=m_new[0]; m_i[1]=m_new[1];

        // ---- O += P·V  (WGMMA reg×smem, A=P regs, B=V). n=D split into NC n64 blocks:
        //      output cols [hh*64, hh*64+64) come from V's hh-th 64-wide column chunk. ----
        WGMMA_FENCE();
        #pragma unroll
        for (int hh=0; hh<NC; ++hh){
            T* vc = pV + hh*FAH_BC*64;
            #pragma unroll
            for (int g=0; g<KP; ++g)   // k16 chunk over Bc; B advances 16 rows within chunk
                WG<T>::rs(od[hh], pk[g][0],pk[g][1],pk[g][2],pk[g][3],
                          smem_desc(vc + g*16*64));
        }
        WGMMA_COMMIT(); WGMMA_WAIT(0);
        __syncthreads();   // done with this K/V buffer
    }

    // ---- epilogue: O = od / l_i , write [Br,D] tile (bf16/fp16) ----
    const int lane = tid%32;
    const float inv0 = l_i[0]>0.f?1.f/l_i[0]:0.f;
    const float inv1 = l_i[1]>0.f?1.f/l_i[1]:0.f;
    #pragma unroll
    for (int hh=0; hh<NC; ++hh)
    #pragma unroll
    for (int g=0; g<4; ++g){
        int col = hh*64 + g*16 + 2*(lane%4);
        float ov[8] = { od[hh][g][0]*inv0, od[hh][g][1]*inv0, od[hh][g][2]*inv1, od[hh][g][3]*inv1,
                        od[hh][g][4]*inv0, od[hh][g][5]*inv0, od[hh][g][6]*inv1, od[hh][g][7]*inv1 };
        int rows[8]={rr,rr,rr+8,rr+8,rr,rr,rr+8,rr+8};
        int cols[8]={col,col+1,col,col+1,col+8,col+9,col+8,col+9};
        #pragma unroll
        for (int e=0;e<8;++e){
            int qr=q0+rows[e];
            if (qr<S) O_bh[(size_t)qr*ostride + cols[e]] = (T)ov[e];
        }
    }
}

template<class T> static CUtensorMapDataType tma_dtype(){
    return std::is_same<T,__half>::value ? CU_TENSOR_MAP_DATA_TYPE_FLOAT16
                                         : CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
}
// ---- 2D TMA map over the [S,D] slice for one (b,h): rows stride = H*D elements ----
template<class T>
static void make_tma(CUtensorMap* tma, const T* base, int S, int H, int D){
    uint64_t gdim[2]    = {(uint64_t)D, (uint64_t)S};                 // inner=D, outer=S
    uint64_t gstride[1] = {(uint64_t)H*(uint64_t)D*sizeof(T)};        // byte stride of the S dim
    uint32_t bdim[2]    = {64u, (uint32_t)FAH_BC};                    // 64-wide swizzle atom × Bc rows
    uint32_t bstride[2] = {1,1};
    CUresult r = cuTensorMapEncodeTiled(tma, tma_dtype<T>(), 2, (void*)base, gdim, gstride, bdim, bstride,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (r!=CUDA_SUCCESS){ const char* s; cuGetErrorString(r,&s); fprintf(stderr,"TMA encode failed: %s\n",s); }
}

template<class T, int D>
static void launch(const T* Q,const T* K,const T* V,T* O,int B,int S,int H,bool causal,cudaStream_t stream){
    // one 2D [S,D] TMA map per (b,h) per tensor (base ptr carries the offset). The
    // device map arrays are cached and rebuilt only when the buffers/shape change,
    // so a bench's steady-state calls are pure kernel launches (accurate timing).
    int BH=B*H;
    static CUtensorMap *dq=nullptr,*dk=nullptr,*dv=nullptr; static int cap=0;
    static const void *lQ=nullptr,*lK=nullptr,*lV=nullptr; static int lS=0,lBH=0;
    if(BH>cap){ if(dq){cudaFree(dq);cudaFree(dk);cudaFree(dv);}
        cudaMalloc(&dq,BH*sizeof(CUtensorMap)); cudaMalloc(&dk,BH*sizeof(CUtensorMap)); cudaMalloc(&dv,BH*sizeof(CUtensorMap)); cap=BH; }
    if(Q!=lQ||K!=lK||V!=lV||S!=lS||BH!=lBH){
        std::vector<CUtensorMap> hq(BH),hk(BH),hv(BH);
        for(int b=0;b<B;++b) for(int h=0;h<H;++h){ int bh=b*H+h; size_t off=((size_t)b*S*H+h)*D;
            make_tma<T>(&hq[bh],Q+off,S,H,D); make_tma<T>(&hk[bh],K+off,S,H,D); make_tma<T>(&hv[bh],V+off,S,H,D); }
        cudaMemcpyAsync(dq,hq.data(),BH*sizeof(CUtensorMap),cudaMemcpyHostToDevice,stream);
        cudaMemcpyAsync(dk,hk.data(),BH*sizeof(CUtensorMap),cudaMemcpyHostToDevice,stream);
        cudaMemcpyAsync(dv,hv.data(),BH*sizeof(CUtensorMap),cudaMemcpyHostToDevice,stream);
        cudaStreamSynchronize(stream);   // maps must persist; hq/hk/hv are stack-local
        lQ=Q;lK=K;lV=V;lS=S;lBH=BH;
    }
    int smem = (FAH_BR*D + 2*2*FAH_BC*D)*sizeof(T);   // Q + double-buffered K,V
    static bool set=false;
    if(!set){ cudaFuncSetAttribute(fa_kernel<T,D>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem); set=true; }
    dim3 grid((S+FAH_BR-1)/FAH_BR, H, B); float scale = 1.f/sqrtf((float)D);
    fa_kernel<T,D><<<grid, FAH_WG, smem, stream>>>(S, H, causal, scale, dq, dk, dv, O);
}

// ---- launcher: fa_hopper_fwd<T>(Q,K,V,O, B,S,H,D, causal, stream) ----
// Q/K/V/O = [B,S,H,D] row-major (matches fa_common.h). MHA (Hq==Hkv). D ∈ {64,128}.
// Single grid launch over (S/Br, H, B); per-(b,h) 2D TMA maps reuse the proven tc_04
// 128B-swizzle inner-64 descriptor.
template<class T>
void fa_hopper_fwd(const T* Q,const T* K,const T* V,T* O,
                   int B,int S,int H,int D,bool causal,cudaStream_t stream){
    static bool inited=false; if(!inited){ cuInit(0); inited=true; }
    if (D==64)       launch<T,64 >(Q,K,V,O,B,S,H,causal,stream);
    else if (D==128) launch<T,128>(Q,K,V,O,B,S,H,causal,stream);
    else { fprintf(stderr,"fa_hopper: D=%d unsupported (need 64 or 128)\n",D); }
}

} // namespace fah
