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
# 用例编成 kernel-only 对象，由统一的 bench/verify 驱动按 id 派发（与 cuda_core 对称）。
TC_DIR := kernels/tensor_core
ifeq ($(TC_HOPPER),1)
  # H20：全部 5 个用例（tc_04/05 用 TMA → driver API）
  TC_CASE_OBJS := $(addprefix $(TC_DIR)/,tc_01_wmma_naive.o tc_02_wmma_smem.o tc_03_wmma_pipe.o tc_04_wgmma_tma_ws.o tc_05_wgmma_fp8.o)
  TC_LDFLAGS := $(LDFLAGS_TMA)
  TC_DEFS :=
else
  # A800/Ampere：只编 WMMA 三级（tc_01-03）；tc_04/05 是 Hopper 独占，-DNO_HOPPER 让派发表跳过
  TC_CASE_OBJS := $(addprefix $(TC_DIR)/,tc_01_wmma_naive.o tc_02_wmma_smem.o tc_03_wmma_pipe.o)
  TC_LDFLAGS := $(LDFLAGS)
  TC_DEFS := -DNO_HOPPER
endif

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

# ---- Tensor core 统一 bench/verify（链接用例对象 + 驱动）----
tc: $(TC_DIR)/bench $(TC_DIR)/verify
$(TC_DIR)/bench: $(TC_DIR)/bench.o $(TC_CASE_OBJS)
	$(NVCC) $^ -o $@ $(NVCCFLAGS) $(TC_LDFLAGS)
$(TC_DIR)/verify: $(TC_DIR)/verify.o $(TC_CASE_OBJS)
	$(NVCC) $^ -o $@ $(NVCCFLAGS) $(TC_LDFLAGS)

# tensor_core 对象：依赖派发声明头 + 用例公共脚手架（$(TC_DEFS) 让 A800 跳过 Hopper 用例）
$(TC_DIR)/%.o: $(TC_DIR)/%.cu $(TC_DIR)/tc_cases.h include/tc_common.cuh
	$(NVCC) -c $< -o $@ $(NVCCFLAGS) $(TC_DEFS)

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
