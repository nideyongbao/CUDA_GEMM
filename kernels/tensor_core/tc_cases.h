#pragma once
// ============================================================================
// tensor_core 用例派发表：每个用例(tc_01..tc_05)各自在其 .cu 里暴露 verify()/bench()
// 两个入口（签名统一 void(int M,int N,int K)），由 bench/verify 两个驱动按 id(1-5) 调用。
// 这样 5 个用例各自保留专属 harness（WMMA / TMA / FP8 setup 各不相同），又共享一套
// 命令行入口，镜像 cuda_core 的 ./bench <id> / ./verify <id>。
// ============================================================================

void tc01_verify(int M,int N,int K); void tc01_bench(int M,int N,int K);
void tc02_verify(int M,int N,int K); void tc02_bench(int M,int N,int K);
void tc03_verify(int M,int N,int K); void tc03_bench(int M,int N,int K);
void tc04_verify(int M,int N,int K); void tc04_bench(int M,int N,int K);
void tc05_verify(int M,int N,int K); void tc05_bench(int M,int N,int K);

struct TCCase { const char* name; void(*verify)(int,int,int); void(*bench)(int,int,int); };

static const TCCase TC_CASES[] = {
    { "tc_01 WMMA_naive",   tc01_verify, tc01_bench },
    { "tc_02 WMMA_smem",    tc02_verify, tc02_bench },
    { "tc_03 WMMA_pipe",    tc03_verify, tc03_bench },
    { "tc_04 WGMMA_TMA_WS", tc04_verify, tc04_bench },
    { "tc_05 WGMMA_fp8",    tc05_verify, tc05_bench },
};
static const int TC_NCASES = (int)(sizeof(TC_CASES)/sizeof(TC_CASES[0]));
