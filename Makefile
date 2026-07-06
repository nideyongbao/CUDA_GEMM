# cuda-ops-h20 — top-level dispatch. Each operator is a self-contained
# CUDA_GEMM-style module (kernels/ include/ docs/ baselines/). H20 Hopper sm_90a.
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
