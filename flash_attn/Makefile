# flash_attn operator — H20 sm_90a.
#   tensor_core: hand-written Hopper WGMMA+TMA forward (include/fa_hopper.cuh) — the
#                performance path & end-state (register-resident online softmax, TMA
#                K/V loads, WGMMA for both matmuls). Needs the driver API (-lcuda).
#   cuda_core  : from-scratch FP32 FA2 scaffold (online softmax) — teaches the algorithm.
NVCC := nvcc
ARCH ?= -arch=sm_90a
TC_FLAGS := $(ARCH) -O3 -std=c++17 -use_fast_math --expt-relaxed-constexpr -I include
CC_FLAGS := $(ARCH) -O3 -std=c++17 -use_fast_math -I include
LD_TC := -lcuda

BUILD ?= build
TC_OUT := $(BUILD)/tensor_core
CC_OUT := $(BUILD)/cuda_core
TC_DEPS := include/fa_common.h include/fa_hopper.cuh

.PHONY: all tc cc clean
all: tc cc
tc: $(TC_OUT)/bench $(TC_OUT)/verify
cc: $(CC_OUT)/bench $(CC_OUT)/verify

$(TC_OUT) $(CC_OUT):
	mkdir -p $@

# ---- tensor_core (Hopper WGMMA+TMA, header-only kernel) ----
$(TC_OUT)/bench:  kernels/tensor_core/bench.cu  $(TC_DEPS) | $(TC_OUT)
	$(NVCC) $< -o $@ $(TC_FLAGS) $(LD_TC)
$(TC_OUT)/verify: kernels/tensor_core/verify.cu $(TC_DEPS) | $(TC_OUT)
	$(NVCC) $< -o $@ $(TC_FLAGS) $(LD_TC)

# ---- cuda_core (scaffold) ----
CC_KSRCS := $(wildcard kernels/cuda_core/fa_cc_*.cu)
CC_KOBJS := $(patsubst kernels/cuda_core/%.cu,$(CC_OUT)/%.o,$(CC_KSRCS))
$(CC_OUT)/%.o: kernels/cuda_core/%.cu include/fa_common.h include/fa_cc.h | $(CC_OUT)
	$(NVCC) -c $< -o $@ $(CC_FLAGS)
$(CC_OUT)/bench: $(CC_OUT)/bench.o $(CC_KOBJS) | $(CC_OUT)
	$(NVCC) $^ -o $@ $(ARCH)
$(CC_OUT)/verify: $(CC_OUT)/verify.o $(CC_KOBJS) | $(CC_OUT)
	$(NVCC) $^ -o $@ $(ARCH)

clean:
	rm -rf $(BUILD)
