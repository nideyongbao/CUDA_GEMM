#pragma once
// ============================================================================
// tensor_core 用例派发表：每个用例(tc_01..tc_06)各自在其 .cu 里暴露 verify()/bench()
// 两个入口（签名统一 void(int M,int N,int K)），由 bench/verify 两个驱动**按 id 查表**调用。
// 这样每个用例各自保留专属 harness（WMMA / TMA / FP8 / mma.sync setup 各不相同），又共享
// 一套命令行入口，镜像 cuda_core 的 ./bench <id> / ./verify <id>。
//
// 派发从"数组下标即 id"改成"显式 id + 线性查表(tc_find)"：H20(默认)全 6 例都注册；但为
// 保留可移植性，老架构(sm_80/sm_89)用 -DNO_HOPPER 会从数组中间摘掉 tc_04/05(Hopper 独占)，
// 若仍用 TC_CASES[id-1] 则 id=6 会越界/错位。显式查表让 id 跨架构稳定：tc_06(sm_80+ 通用
// mma.sync)永远是 id=6，无论 tc_04/05 在不在。
// ============================================================================
#include <cstdio>

void tc01_verify(int,int,int); void tc01_bench(int,int,int);
void tc02_verify(int,int,int); void tc02_bench(int,int,int);
void tc03_verify(int,int,int); void tc03_bench(int,int,int);
#ifndef NO_HOPPER
// tc_04(WGMMA+TMA) / tc_05(FP8) 用 Hopper sm_90 独占指令；A800/Ampere 编译时 -DNO_HOPPER 跳过。
void tc04_verify(int,int,int); void tc04_bench(int,int,int);
void tc05_verify(int,int,int); void tc05_bench(int,int,int);
#endif
// tc_06(mma.sync+ldmatrix+cp.async) 是 sm_80+ 通用指令：A800 与 Hopper 都编都跑，**始终注册**。
void tc06_verify(int,int,int); void tc06_bench(int,int,int);

struct TCCase { int id; const char* name; void(*verify)(int,int,int); void(*bench)(int,int,int); };

static const TCCase TC_CASES[] = {
    { 1, "tc_01 WMMA_naive",   tc01_verify, tc01_bench },
    { 2, "tc_02 WMMA_smem",    tc02_verify, tc02_bench },
    { 3, "tc_03 WMMA_pipe",    tc03_verify, tc03_bench },
#ifndef NO_HOPPER
    { 4, "tc_04 WGMMA_TMA_WS", tc04_verify, tc04_bench },
    { 5, "tc_05 WGMMA_fp8",    tc05_verify, tc05_bench },
#endif
    { 6, "tc_06 MMA_pipe",     tc06_verify, tc06_bench },
};
static const int TC_NCASES = (int)(sizeof(TC_CASES)/sizeof(TC_CASES[0]));

// 按 id 查用例（跨架构稳定）；未注册(如 Ampere 上 id=4/5)返回 nullptr。
static inline const TCCase* tc_find(int id){
    for(int i=0;i<TC_NCASES;i++) if(TC_CASES[i].id==id) return &TC_CASES[i];
    return nullptr;
}
static inline void tc_list(){
    printf("tensor_core 用例 id：\n");
    for(int i=0;i<TC_NCASES;i++) printf("  %d  %s\n", TC_CASES[i].id, TC_CASES[i].name);
}
