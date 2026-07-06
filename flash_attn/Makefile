# flash_attn operator — A800 sm_80.
#   tensor_core: TinyFA's CuTe forward (vendored) — the performance path & end-state.
#   cuda_core  : from-scratch FA2 scaffold (online softmax) — teaches the algorithm.
NVCC := nvcc
ARCH ?= -arch=sm_80
CUTLASS ?= ../third_party/cutlass/include
TFA := vendor/tfa

# TinyFA build knobs (mirror its setup.py): sm80, headdim 64 & 128, fp16+bf16.
TFA_DEFS := -DTFA_TARGET_SM=80 -DTFA_TARGET_HEADDIM_64=1 -DTFA_TARGET_HEADDIM_128=1
TC_FLAGS := $(ARCH) -O3 -std=c++17 -use_fast_math --expt-relaxed-constexpr \
            -I include -I $(TFA) -I $(CUTLASS) $(TFA_DEFS)
CC_FLAGS := $(ARCH) -O3 -std=c++17 -use_fast_math -I include

BUILD ?= build
TC_OUT := $(BUILD)/tensor_core
CC_OUT := $(BUILD)/cuda_core

# ---- tensor_core (TinyFA vendored) ----
TFA_KOBJS := $(TC_OUT)/tfa_fwd_fp16_sm80.o $(TC_OUT)/tfa_fwd_bf16_sm80.o

.PHONY: all tc cc clean
all: tc cc
tc: $(TC_OUT)/bench $(TC_OUT)/verify
cc: $(CC_OUT)/bench $(CC_OUT)/verify

$(TC_OUT) $(CC_OUT):
	mkdir -p $@

$(TC_OUT)/%.o: $(TFA)/kernels/%.cu | $(TC_OUT)
	$(NVCC) -c $< -o $@ $(TC_FLAGS)
$(TC_OUT)/bench.o: kernels/tensor_core/bench.cu | $(TC_OUT)
	$(NVCC) -c $< -o $@ $(TC_FLAGS)
$(TC_OUT)/verify.o: kernels/tensor_core/verify.cu | $(TC_OUT)
	$(NVCC) -c $< -o $@ $(TC_FLAGS)
$(TC_OUT)/bench: $(TC_OUT)/bench.o $(TFA_KOBJS) | $(TC_OUT)
	$(NVCC) $^ -o $@ $(ARCH)
$(TC_OUT)/verify: $(TC_OUT)/verify.o $(TFA_KOBJS) | $(TC_OUT)
	$(NVCC) $^ -o $@ $(ARCH)

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
