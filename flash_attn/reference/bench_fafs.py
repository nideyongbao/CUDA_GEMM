#!/usr/bin/env python3
# Benchmark the reference ladder flash_attention_from_scratch (fafs) on A800.
# Gives the authoritative performance curve our hand-written fa_tc ladder is measured
# against. Run after building fafs (see BUILD_fafs.md).
#   LD_LIBRARY_PATH=<torch>/lib CUDA_VISIBLE_DEVICES=<free> python3 bench_fafs.py
import torch, flash_attention
import flash_helpers.kernel_configs as kc

B, H, S, D = 2, 32, 4096, 128
torch.manual_seed(0)
q = torch.randn(B, S, H, D, device="cuda", dtype=torch.float16)  # [B,S,H,D] (fafs & our layout)
k = torch.randn_like(q); v = torch.randn_like(q)
flops = 4.0 * B * H * S * S * D

def bench(cfg):
    f = lambda: flash_attention.forward(cfg, q, k, v)
    for _ in range(10): f()
    torch.cuda.synchronize()
    ts = []
    for _ in range(40):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); f(); e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e))
    ts.sort(); return ts[len(ts)//2]

print(f"fafs @ B{B} H{H} S{S} D{D} fp16 non-causal (A800, lock clock for stable numbers)\n")
print("== curated 7-rung progression (Br=64/Bc=64/4warps, same as our fa_tc) ==")
for i, cfg in enumerate(kc.get_kernel_progression_configs()):
    ms = bench(cfg); print(f"  rung {i+1}: {flops/(ms*1e-3)/1e12:6.1f} TFLOPS  [{cfg.short_form().split(': ')[1]}]")

print("\n== fastest of all built configs (autotune sweep) ==")
res = []
for cfg in kc.get_kernels_to_build():
    try: res.append((bench(cfg), cfg))
    except Exception: pass
res.sort()
for ms, cfg in res[:5]:
    print(f"  {flops/(ms*1e-3)/1e12:6.1f} TFLOPS  [{cfg.short_form()}]")
