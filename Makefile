NVCC := nvcc

# 目标硬件：NVIDIA H20 (Hopper, compute capability 9.0)。
# sm_90a 是 Hopper 架构专用目标（WGMMA/TMA 等 Hopper 指令需要它）。
# 换其他卡：Ada -> sm_89，A100 -> sm_80，Turing -> sm_75。
ARCH := -arch=sm_90a
INCLUDES := -I include
OPT := -O3

NVCCFLAGS := $(ARCH) $(INCLUDES) $(OPT)
LDFLAGS := -lcublas
LDFLAGS_TMA := -lcublas -lcuda   # tensor_core 用 TMA，需要 driver API

COMMON_DEPS := include/common.h include/kernels.h include/06_autotuning.cuh include/bf16.h

# ============ CUDA core（kernel + 驱动 + 可执行 全在 kernels/cuda_core/）============
CC_DIR := kernels/cuda_core
# 该目录里的 4 个 host 驱动（各自带 main），是 harness 不是 kernel，链接时单独处理
CC_DRIVERS := $(addprefix $(CC_DIR)/,benchmark.cu verify.cu bench_bf16.cu verify_bf16.cu)
# FP32 kernel 对象 = cuda_core 下所有 .cu，去掉 bf16_cudacore 与 4 个驱动
FP32_KERNEL_OBJS := $(patsubst %.cu,%.o,$(filter-out $(CC_DIR)/bf16_cudacore.cu $(CC_DRIVERS),$(wildcard $(CC_DIR)/*.cu)))
# BF16 on CUDA core（datatype 实验：bf16 输入 + FP32 累加，仍跑 CUDA core）
BF16_KERNEL_OBJS := $(CC_DIR)/bf16_cudacore.o

# ============ Tensor core（kernel + 驱动 + 可执行 全在 kernels/tensor_core/）============
# 5 个用例编成 kernel-only 对象，由统一的 bench/verify 驱动按 id(1-5) 派发（与 cuda_core 对称）。
TC_DIR := kernels/tensor_core
TC_CASE_OBJS := $(addprefix $(TC_DIR)/,tc_01_wmma_naive.o tc_02_wmma_smem.o tc_03_wmma_pipe.o tc_04_wgmma_tma_ws.o tc_05_wgmma_fp8.o)

.PHONY: all clean run run-verify bf16 tc cudacore

# 默认：FP32 sweep + BF16(cuda core) sweep
all: cudacore
cudacore: $(CC_DIR)/bench $(CC_DIR)/verify $(CC_DIR)/bench_bf16 $(CC_DIR)/verify_bf16

# ---- FP32 (CUDA core) ----
$(CC_DIR)/bench: $(CC_DIR)/benchmark.o $(FP32_KERNEL_OBJS)
	$(NVCC) $^ -o $@ $(NVCCFLAGS) $(LDFLAGS)
$(CC_DIR)/verify: $(CC_DIR)/verify.o $(FP32_KERNEL_OBJS)
	$(NVCC) $^ -o $@ $(NVCCFLAGS) $(LDFLAGS)

# ---- BF16 on CUDA core (全量对照 sweep) ----
bf16: $(CC_DIR)/bench_bf16 $(CC_DIR)/verify_bf16
$(CC_DIR)/bench_bf16: $(CC_DIR)/bench_bf16.o $(BF16_KERNEL_OBJS)
	$(NVCC) $^ -o $@ $(NVCCFLAGS) $(LDFLAGS)
$(CC_DIR)/verify_bf16: $(CC_DIR)/verify_bf16.o $(BF16_KERNEL_OBJS)
	$(NVCC) $^ -o $@ $(NVCCFLAGS) $(LDFLAGS)

# ---- Tensor core 统一 bench/verify（链接 5 个用例对象 + 驱动）----
tc: $(TC_DIR)/bench $(TC_DIR)/verify
$(TC_DIR)/bench: $(TC_DIR)/bench.o $(TC_CASE_OBJS)
	$(NVCC) $^ -o $@ $(NVCCFLAGS) $(LDFLAGS_TMA)
$(TC_DIR)/verify: $(TC_DIR)/verify.o $(TC_CASE_OBJS)
	$(NVCC) $^ -o $@ $(NVCCFLAGS) $(LDFLAGS_TMA)

# tensor_core 对象：依赖派发声明头 + 用例公共脚手架
$(TC_DIR)/%.o: $(TC_DIR)/%.cu $(TC_DIR)/tc_cases.h include/tc_common.cuh
	$(NVCC) -c $< -o $@ $(NVCCFLAGS)

# 通用 .o 规则（kernels/cuda_core/ 的 kernel 与驱动都走这条）
%.o: %.cu $(COMMON_DEPS)
	$(NVCC) -c $< -o $@ $(NVCCFLAGS)

run: $(CC_DIR)/bench
	./$(CC_DIR)/bench
run-verify: $(CC_DIR)/verify
	./$(CC_DIR)/verify 1024 1024 1024

clean:
	rm -f $(CC_DIR)/*.o $(TC_DIR)/*.o \
	      $(CC_DIR)/bench $(CC_DIR)/verify $(CC_DIR)/bench_bf16 $(CC_DIR)/verify_bf16 \
	      $(TC_DIR)/bench $(TC_DIR)/verify
