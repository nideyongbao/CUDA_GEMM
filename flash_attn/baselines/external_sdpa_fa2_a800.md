# FA 外部基线：torch SDPA 与 flash-attn v2（仅对比，不引入依赖）

本教程刻意**不依赖** torch / flash-attn 三方库。为给 `flash_attn/` 的张量核前向一个"生产库"参照，这里**仅摘取**本机预采归档的数字做对比：

- 来源：`/data/env/workspace/0629/nvidia-gpu-baseline/archives/A800-SXM-torch210-cuda129/03_flash_attn_benchmark.json`
- 机器：NVIDIA A800-SXM4-80GB（sm_80），torch 2.10 / cuda 12.9，采于 2026-04-28
- FA2 = flash-attn **2.8.3**；FA3 该机不可用；correctness `pass=true`（FA2 vs math max_diff 2.4e-3）
- FLOP 口径：`4·B·H·S²·D`，causal ÷2 —— **与本仓库 `fa_common.h::fa_flops` 完全一致**（已逐项核对：FA2 B8 S4096 nh28 hd128 非causal fwd 8.969ms → 214.5 TFLOPS 与本公式吻合）。故 TFLOPS 可直接对比。

## 头对头（B8 H28 S4096 D128，前向）

| backend | 非causal TFLOPS | causal TFLOPS | 出处 |
| --- | ---: | ---: | --- |
| flash-attn v2 (2.8.3) —— 参考上限 | 214.5 | 198.5 | 归档 |
| torch SDPA (flash backend) | 204.6 | 181.6 | 归档 |
| torch SDPA (cuDNN backend) | 188.8 | 179.1 | 归档 |
| **TinyFA（本仓库，fp16）** | **202.9** (bf16 207.5) | **185.0** | 本机实测（`bench fp16 8 28 4096 128 {0,1}`，锁频1410MHz） |
| torch SDPA (mem-efficient) | 112.0 | 106.6 | 归档 |
| custom triton（教学版） | 93.2 | 82.2 | 归档 |

**结论**：TinyFA = FA2 的 **94.6% / 93.2%**（非causal / causal），与 torch SDPA-flash 基本持平，是教学版 triton 的 ~2.2×。印证 "94–96% Dao FA2" 的定位。

## 归档中其它配置（FA2 与 SDPA，A800，前向 TFLOPS）

| config (B,S,nh,hd,causal) | FA2 | SDPA-auto | SDPA-flash | SDPA-cudnn |
| --- | ---: | ---: | ---: | ---: |
| 8,2048,28,128,True | 160.7 | 171.0 | 161.4 | 169.8 |
| 8,2048,28,128,False | 210.7 | 199.2 | 197.0 | 186.0 |
| 8,4096,28,128,True | 198.5 | 182.4 | 181.6 | 179.1 |
| 8,4096,28,128,False | 214.5 | 205.3 | 204.6 | 188.8 |
| 16,4096,28,128,True | 199.1 | 185.2 | 182.8 | 180.2 |
| 32,4096,32,128,True | 199.9 | 184.5 | 184.1 | 180.8 |

> 注：FA2 归档含 backward 数据（fwd+bwd），本教程只做前向，故只取 `fwd_tflops`。SDPA 的 mem-efficient / math 后端与 custom_triton 明显更慢，已在头对头表中给出量级。
