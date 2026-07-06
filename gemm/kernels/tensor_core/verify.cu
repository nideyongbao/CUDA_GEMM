// tensor_core 统一正确性驱动：./verify <id> [M N K]
// 按 id 查表派发到对应用例的 verify()（对拍参考实现，打印 PASS/FAIL）。镜像 cuda_core/verify。
// 注：tc_05(FP8) 没有简单 cuBLAS 路径，用 CPU double 参考，故其 verify 在 <=512^3 上对拍。
#include "tc_cases.h"
#include <cstdio>
#include <cstdlib>

int main(int argc, char** argv){
    int id = 4, M = 4096, N = 4096, K = 4096;   // 默认 tc_04 @ 4096^3
    if (argc == 2) { id = atoi(argv[1]); }
    else if (argc == 5) { id = atoi(argv[1]); M = atoi(argv[2]); N = atoi(argv[3]); K = atoi(argv[4]); }
    else if (argc != 1) { printf("Usage: %s <id> [M N K]\n", argv[0]); tc_list(); return 1; }

    const TCCase* c = tc_find(id);
    if (!c) { printf("未知/本架构不可用的 id=%d\n", id); tc_list(); return 1; }
    c->verify(M, N, K);
    return 0;
}
