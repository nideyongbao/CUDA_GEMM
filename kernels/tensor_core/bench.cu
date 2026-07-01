// tensor_core 统一性能驱动：./bench <id> [M N K]
// 按 id 查表派发到对应用例的 bench()（纯计时 + 利用率，不做对拍）。镜像 cuda_core/bench。
#include "tc_cases.h"
#include <cstdio>
#include <cstdlib>

int main(int argc, char** argv){
    int id = 4, M = 4096, N = 4096, K = 4096;   // 默认 tc_04（Hopper 手写阶梯标杆）@ 4096^3
    if (argc == 2) { id = atoi(argv[1]); }
    else if (argc == 5) { id = atoi(argv[1]); M = atoi(argv[2]); N = atoi(argv[3]); K = atoi(argv[4]); }
    else if (argc != 1) { printf("Usage: %s <id> [M N K]\n", argv[0]); tc_list(); return 1; }

    const TCCase* c = tc_find(id);
    if (!c) { printf("未知/本架构不可用的 id=%d\n", id); tc_list(); return 1; }
    c->bench(M, N, K);
    return 0;
}
