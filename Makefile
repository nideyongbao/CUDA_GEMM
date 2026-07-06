# cuda-ops-a800 — top-level dispatch. Each operator is a self-contained
# CUDA_GEMM-style module (kernels/ include/ docs/ baselines/). A800 sm_80.
.PHONY: all gemm softmax flash_attn clean
all: gemm softmax flash_attn

gemm:
	$(MAKE) -C gemm
	$(MAKE) -C gemm bf16
	$(MAKE) -C gemm tc

softmax:
	$(MAKE) -C softmax

flash_attn:
	$(MAKE) -C flash_attn

clean:
	-$(MAKE) -C gemm clean
	-$(MAKE) -C softmax clean
	-$(MAKE) -C flash_attn clean
