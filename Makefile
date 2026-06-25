NVCC := nvcc

# 目标硬件：NVIDIA H20 (Hopper, compute capability 9.0)。
# sm_90a 是 Hopper 架构专用目标（WGMMA/TMA 等 Hopper 指令需要它）。
# 换其他卡：Ada -> sm_89，A100 -> sm_80，Turing -> sm_75。
ARCH := -arch=sm_90a
INCLUDES := -I include
OPT := -O3

NVCCFLAGS := $(ARCH) $(INCLUDES) $(OPT)
LDFLAGS := -lcublas
LDFLAGS_TMA := -lcublas -lcuda   # tensor_core 里用 TMA 的用例需要 driver API

COMMON_DEPS := include/common.h include/kernels.h include/06_autotuning.cuh include/bf16.h

# ============ CUDA core ============
# FP32 kernels（kernels/cuda_core 下，排除 bf16_cudacore）
FP32_KERNEL_OBJS := $(patsubst %.cu,%.o,$(filter-out kernels/cuda_core/bf16_cudacore.cu,$(wildcard kernels/cuda_core/*.cu)))
# BF16 on CUDA core（datatype 实验：bf16 输入 + FP32 累加，仍跑 CUDA core）
BF16_KERNEL_OBJS := kernels/cuda_core/bf16_cudacore.o

# ============ Tensor core 最简用例（各自带 main，独立可执行）============
TC_BF16 := tc_01_wmma_naive tc_02_wmma_smem tc_03_wmma_pipe
TC_TMA  := tc_04_wgmma_tma_ws tc_05_wgmma_fp8

.PHONY: all clean run run-verify bf16 tc

# 默认：FP32 sweep + BF16(cuda core) sweep
all: bench verify bench_bf16 verify_bf16

# ---- FP32 (CUDA core) ----
bench: src/benchmark.o $(FP32_KERNEL_OBJS)
	$(NVCC) $^ -o $@ $(NVCCFLAGS) $(LDFLAGS)
verify: src/verify.o $(FP32_KERNEL_OBJS)
	$(NVCC) $^ -o $@ $(NVCCFLAGS) $(LDFLAGS)

# ---- BF16 on CUDA core (全量对照 sweep) ----
bf16: bench_bf16 verify_bf16
bench_bf16: src/bench_bf16.o $(BF16_KERNEL_OBJS)
	$(NVCC) $^ -o $@ $(NVCCFLAGS) $(LDFLAGS)
verify_bf16: src/verify_bf16.o $(BF16_KERNEL_OBJS)
	$(NVCC) $^ -o $@ $(NVCCFLAGS) $(LDFLAGS)

# ---- Tensor core 用例 ----
tc: $(TC_BF16) $(TC_TMA)
$(TC_BF16): %: kernels/tensor_core/%.cu include/tc_common.cuh
	$(NVCC) $< -o $@ $(NVCCFLAGS) $(LDFLAGS)
$(TC_TMA): %: kernels/tensor_core/%.cu
	$(NVCC) $< -o $@ $(NVCCFLAGS) $(LDFLAGS_TMA)

%.o: %.cu $(COMMON_DEPS)
	$(NVCC) -c $< -o $@ $(NVCCFLAGS)

run: bench
	./bench
run-verify: verify
	./verify 1024 1024 1024

clean:
	rm -f kernels/cuda_core/*.o src/*.o bench verify bench_bf16 verify_bf16 $(TC_BF16) $(TC_TMA)
