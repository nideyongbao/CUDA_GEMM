NVCC := nvcc

# 目标硬件（可在命令行覆盖）：
#   H20  (Hopper, CC 9.0)  默认: make            / make tc
#   A800 (Ampere,CC 8.0)        : make ARCH=-arch=sm_80
#                                 make tc ARCH=-arch=sm_80 TC_HOPPER=0
# sm_90a 是 Hopper 架构专用目标（WGMMA/TMA 等 Hopper 指令需要它）。
# 换其他卡：Ada -> sm_89，A100/A800 -> sm_80，Turing -> sm_75。
ARCH ?= -arch=sm_90a
INCLUDES := -I include
OPT := -O3

# TC_HOPPER=1（默认，H20）编译全部 5 个 tensor_core 用例（含 WGMMA/TMA/FP8，需 driver API）。
# TC_HOPPER=0（A800/Ampere）只编译 tc_01-03（WMMA / cp.async），并用 -DNO_HOPPER 让派发表
# 跳过 tc_04(WGMMA+TMA) / tc_05(FP8) —— 这两者是 Hopper sm_90 独占指令，Ampere 无法运行。
TC_HOPPER ?= 1

# 编译产物输出目录（out-of-source）：所有 .o 与可执行文件都进 $(BUILD)/，代码目录(kernels/)保持干净。
#   手动 make 默认输出到 build/；run_all.sh 传 BUILD=result/<时间戳>/build 让产物随快照走。
BUILD ?= build

NVCCFLAGS := $(ARCH) $(INCLUDES) $(OPT)
LDFLAGS := -lcublas
LDFLAGS_TMA := -lcublas -lcuda   # tensor_core 用 TMA，需要 driver API

COMMON_DEPS := include/common.h include/kernels.h include/06_autotuning.cuh include/bf16.h

# ============ 源码目录(只读) 与 产物目录(out-of-source) ============
CC_DIR := kernels/cuda_core
TC_DIR := kernels/tensor_core
CC_OUT := $(BUILD)/cuda_core
TC_OUT := $(BUILD)/tensor_core

# CUDA core：4 个 host 驱动(各自带 main)单独处理；FP32 kernel 对象 = cuda_core 下所有 .cu 去掉 bf16 与驱动
CC_DRIVERS := $(addprefix $(CC_DIR)/,benchmark.cu verify.cu bench_bf16.cu verify_bf16.cu)
FP32_SRCS := $(filter-out $(CC_DIR)/bf16_cudacore.cu $(CC_DRIVERS),$(wildcard $(CC_DIR)/*.cu))
FP32_KERNEL_OBJS := $(patsubst $(CC_DIR)/%.cu,$(CC_OUT)/%.o,$(FP32_SRCS))
BF16_KERNEL_OBJS := $(CC_OUT)/bf16_cudacore.o

# Tensor core 用例对象（Hopper 全 5 个 / Ampere 只 tc_01-03）
ifeq ($(TC_HOPPER),1)
  TC_CASE_OBJS := $(addprefix $(TC_OUT)/,tc_01_wmma_naive.o tc_02_wmma_smem.o tc_03_wmma_pipe.o tc_04_wgmma_tma_ws.o tc_05_wgmma_fp8.o)
  TC_LDFLAGS := $(LDFLAGS_TMA)
  TC_DEFS :=
else
  TC_CASE_OBJS := $(addprefix $(TC_OUT)/,tc_01_wmma_naive.o tc_02_wmma_smem.o tc_03_wmma_pipe.o)
  TC_LDFLAGS := $(LDFLAGS)
  TC_DEFS := -DNO_HOPPER
endif

.PHONY: all clean run run-verify bf16 tc cudacore

all: cudacore
cudacore: $(CC_OUT)/bench $(CC_OUT)/verify $(CC_OUT)/bench_bf16 $(CC_OUT)/verify_bf16

# 产物目录（order-only 前置，自动创建）
$(CC_OUT) $(TC_OUT):
	mkdir -p $@

# ---- FP32 (CUDA core) ----
$(CC_OUT)/bench: $(CC_OUT)/benchmark.o $(FP32_KERNEL_OBJS) | $(CC_OUT)
	$(NVCC) $^ -o $@ $(NVCCFLAGS) $(LDFLAGS)
$(CC_OUT)/verify: $(CC_OUT)/verify.o $(FP32_KERNEL_OBJS) | $(CC_OUT)
	$(NVCC) $^ -o $@ $(NVCCFLAGS) $(LDFLAGS)

# ---- BF16 on CUDA core ----
bf16: $(CC_OUT)/bench_bf16 $(CC_OUT)/verify_bf16
$(CC_OUT)/bench_bf16: $(CC_OUT)/bench_bf16.o $(BF16_KERNEL_OBJS) | $(CC_OUT)
	$(NVCC) $^ -o $@ $(NVCCFLAGS) $(LDFLAGS)
$(CC_OUT)/verify_bf16: $(CC_OUT)/verify_bf16.o $(BF16_KERNEL_OBJS) | $(CC_OUT)
	$(NVCC) $^ -o $@ $(NVCCFLAGS) $(LDFLAGS)

# ---- Tensor core 统一 bench/verify ----
tc: $(TC_OUT)/bench $(TC_OUT)/verify
$(TC_OUT)/bench: $(TC_OUT)/bench.o $(TC_CASE_OBJS) | $(TC_OUT)
	$(NVCC) $^ -o $@ $(NVCCFLAGS) $(TC_LDFLAGS)
$(TC_OUT)/verify: $(TC_OUT)/verify.o $(TC_CASE_OBJS) | $(TC_OUT)
	$(NVCC) $^ -o $@ $(NVCCFLAGS) $(TC_LDFLAGS)

# ---- 编译规则：源码在 kernels/，对象输出到 $(BUILD)/ ----
# tensor_core 对象（依赖派发头 + 脚手架；$(TC_DEFS) 让 A800 跳过 Hopper 用例）
$(TC_OUT)/%.o: $(TC_DIR)/%.cu $(TC_DIR)/tc_cases.h include/tc_common.cuh | $(TC_OUT)
	$(NVCC) -c $< -o $@ $(NVCCFLAGS) $(TC_DEFS)
# cuda_core 对象（kernel 与 4 个驱动都走这条）
$(CC_OUT)/%.o: $(CC_DIR)/%.cu $(COMMON_DEPS) | $(CC_OUT)
	$(NVCC) -c $< -o $@ $(NVCCFLAGS)

run: $(CC_OUT)/bench
	./$(CC_OUT)/bench
run-verify: $(CC_OUT)/verify
	./$(CC_OUT)/verify 1024 1024 1024

# 清理：删产物目录 + 顺带清掉历史遗留在代码目录里的产物（保证 kernels/ 干净）
clean:
	rm -rf $(BUILD)
	rm -f $(CC_DIR)/*.o $(TC_DIR)/*.o \
	      $(CC_DIR)/bench $(CC_DIR)/verify $(CC_DIR)/bench_bf16 $(CC_DIR)/verify_bf16 \
	      $(TC_DIR)/bench $(TC_DIR)/verify
