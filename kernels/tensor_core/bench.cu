// tensor_core 统一性能驱动：./bench <id 1-5> [M N K]
// 按 id 派发到对应用例的 bench()（纯计时 + 利用率，不做对拍）。镜像 cuda_core/bench。
#include "tc_cases.h"
#include <cstdio>
#include <cstdlib>

int main(int argc, char** argv){
    int id = 4, M = 4096, N = 4096, K = 4096;   // 默认 tc_04（手写阶梯标杆）@ 4096^3
    if (argc == 2) { id = atoi(argv[1]); }
    else if (argc == 5) { id = atoi(argv[1]); M = atoi(argv[2]); N = atoi(argv[3]); K = atoi(argv[4]); }
    else if (argc != 1) { printf("Usage: %s <id 1-%d> [M N K]\n", argv[0], TC_NCASES); return 1; }

    if (id < 1 || id > TC_NCASES) {
        printf("tensor_core 用例 id：\n");
        for (int i = 0; i < TC_NCASES; i++) printf("  %d  %s\n", i + 1, TC_CASES[i].name);
        return 1;
    }
    TC_CASES[id - 1].bench(M, N, K);
    return 0;
}
