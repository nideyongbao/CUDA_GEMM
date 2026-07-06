# softmax operator — A800 sm_80. Mirrors the GEMM operator's out-of-source build.
NVCC := nvcc
ARCH ?= -arch=sm_80
INCLUDES := -I include
OPT := -O3 -use_fast_math
NVCCFLAGS := $(ARCH) $(INCLUDES) $(OPT)

BUILD ?= build
CC_DIR := kernels/cuda_core
CC_OUT := $(BUILD)/cuda_core

# kernel objects = every rung .cu except the two drivers
DRIVERS := $(CC_DIR)/bench.cu $(CC_DIR)/verify.cu
KSRCS := $(filter-out $(DRIVERS),$(wildcard $(CC_DIR)/*.cu))
KOBJS := $(patsubst $(CC_DIR)/%.cu,$(CC_OUT)/%.o,$(KSRCS))
DEPS := include/softmax.h

.PHONY: all clean
all: $(CC_OUT)/bench $(CC_OUT)/verify

$(CC_OUT):
	mkdir -p $@

$(CC_OUT)/bench: $(CC_OUT)/bench.o $(KOBJS) | $(CC_OUT)
	$(NVCC) $^ -o $@ $(NVCCFLAGS)
$(CC_OUT)/verify: $(CC_OUT)/verify.o $(KOBJS) | $(CC_OUT)
	$(NVCC) $^ -o $@ $(NVCCFLAGS)

$(CC_OUT)/%.o: $(CC_DIR)/%.cu $(DEPS) | $(CC_OUT)
	$(NVCC) -c $< -o $@ $(NVCCFLAGS)

clean:
	rm -rf $(BUILD)
