NVCC := nvcc

# 目标硬件（可命令行覆盖 ARCH / TC_HOPPER；run_all.sh 会按 nvidia-smi 的 CC 自动填这两个）：
#   代际        CC     ARCH          TC_HOPPER  张量核用例
#   ---------   -----  ------------  ---------  --------------------------------
#   Ampere-DC   8.0    -arch=sm_80   0          tc_01-03 (WMMA)           A100/A800
#   Ampere-GFX  8.6    -arch=sm_86   0          tc_01-03 (WMMA)           A10/A10G/A40
#   Ada         8.9    -arch=sm_89   0          tc_01-03 (WMMA)*          L4/L40S
#   Hopper      9.0    -arch=sm_90a  1          tc_01-05 (WMMA+WGMMA+FP8) H20/H100/H200  ← 默认
#   Blackwell   10/12  -arch=sm_100  0          tc_01-03 (WMMA)**         B200/RTX50
#   * Ada 有 FP8 硬件，但本仓库 FP8 走 Hopper 独占的 WGMMA 指令，故 Ada 上只跑 WMMA。
#   ** Blackwell 的 WGMMA 换成了 tcgen05/UMMA、并新增 FP4——需另写 kernel，暂只跑通用 WMMA。
# sm_90a 的 'a' = architecture-specific：WGMMA/TMA 等 Hopper 指令**必须**用它(不能用 sm_90)。
# 各卡的峰值/精度支持是 scripts/gpu_specs.py 这一份表说了算(汇总报告算 MFU 用它)。
ARCH ?= -arch=sm_80
INCLUDES := -I include
OPT := -O3

# TC_HOPPER=1（默认，H20）编译全部 5 个 tensor_core 用例（含 WGMMA/TMA/FP8，需 driver API）。
# TC_HOPPER=0（A800/Ampere）只编译 tc_01-03（WMMA / cp.async），并用 -DNO_HOPPER 让派发表
# 跳过 tc_04(WGMMA+TMA) / tc_05(FP8) —— 这两者是 Hopper sm_90 独占指令，Ampere 无法运行。
TC_HOPPER ?= 0

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

# Tensor core 用例对象。tc_06(mma.sync，sm_80+ 通用)两个分支都编；tc_04/05(Hopper 独占)只 Hopper 编。
ifeq ($(TC_HOPPER),1)
  TC_CASE_OBJS := $(addprefix $(TC_OUT)/,tc_01_wmma_naive.o tc_02_wmma_smem.o tc_03_wmma_pipe.o tc_04_wgmma_tma_ws.o tc_05_wgmma_fp8.o tc_06_mma_pipe.o)
  TC_LDFLAGS := $(LDFLAGS_TMA)
  TC_DEFS :=
else
  TC_CASE_OBJS := $(addprefix $(TC_OUT)/,tc_01_wmma_naive.o tc_02_wmma_smem.o tc_03_wmma_pipe.o tc_06_mma_pipe.o)
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
