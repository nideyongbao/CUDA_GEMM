# GEMM 测试汇总 — NVIDIA H20

- 计算能力: **CC 9.0**  |  SM 数: 78  |  FP32 核/SM: 128
- 时钟策略: **locked@1980MHz(额定boost)**（额定 max 1980 MHz）
- 运行时实测 SM 时钟(遥测 pclk): 计算期典型 **1830 MHz** / 峰 1830 MHz（额定 1980）
- 理论峰值: FP32 = **39.5 TFLOPS**  |  BF16 张量核 = **148 TFLOPS**  |  FP8 张量核 = **296 TFLOPS**
- headline 尺寸: 4096³

## 正确性（对拍 cuBLAS）
- FP32: 13 PASS / 0 FAIL  |  BF16: 13 PASS / 0 FAIL  |  Tensor(WMMA/WGMMA): 5 PASS / 0 FAIL

## FP32 (CUDA core) @ 4096³
| kernel | GFLOPS | %FP32峰值 | %cuBLAS |
| --- | ---: | ---: | ---: |
| cublas_ref | 28034 |  70.9% | 100.0% |
| naive_kernel | 3353 |   8.5% |  12.0% |
| smem_kernel | 5042 |  12.8% |  18.0% |
| blocktiling_kernel | 8825 |  22.3% |  31.5% |
| Dblocktiling_kernel | 14467 |  36.6% |  51.6% |
| vectorized_kernel | 21072 |  53.3% |  75.2% |
| autotune_64x64x8_8x4 | 21051 |  53.2% |  75.1% |
| autotune_64x64x16_8x4 | 20250 |  51.2% |  72.2% |
| autotune_64x64x8_8x8 | 18327 |  46.4% |  65.4% |
| warptile_kernel | 11882 |  30.1% |  42.4% |
| warptile_vec_kernel | 22221 |  56.2% |  79.3% |
| bank_conflict_kernel | 19952 |  50.5% |  71.2% |
| double_buffer_kernel | 23218 |  58.7% |  82.8% |

## BF16 (CUDA core, bf16输入+fp32累加) @ 4096³
| kernel | GFLOPS | %BF16峰值 | %cuBLAS |
| --- | ---: | ---: | ---: |
| cublas_bf16 | 134014 |  90.5% | 100.0% |
| naive | 3025 |   2.0% |   2.3% |
| smem | 5079 |   3.4% |   3.8% |
| blocktiling | 9615 |   6.5% |   7.2% |
| 2Dblocktiling | 14772 |  10.0% |  11.0% |
| vectorized | 21849 |  14.8% |  16.3% |
| autotune_64x64x8_8x4 | 21848 |  14.8% |  16.3% |
| autotune_64x64x16_8x4 | 20297 |  13.7% |  15.1% |
| autotune_64x64x8_8x8 | 18761 |  12.7% |  14.0% |
| warptile | 11351 |   7.7% |   8.5% |
| warptile_vec | 22797 |  15.4% |  17.0% |
| bank_conflict | 20854 |  14.1% |  15.6% |
| double_buffer | 23989 |  16.2% |  17.9% |

## Tensor Core @ 4096³
> 利用率口径：WMMA/WGMMA(bf16)→BF16 峰值；FP8→FP8 峰值(=2×BF16)；FP4→FP4 峰值(=4×BF16)。
| 用例 | 精度 | GFLOPS | %对应精度峰值 | %cuBLAS_bf16 |
| --- | --- | ---: | ---: | ---: |
| tc_01 WMMA_naive | BF16 | 16946 |  11.5% |  12.6% |
| tc_02 WMMA_smem | BF16 | 30132 |  20.4% |  22.5% |
| tc_03 WMMA_pipe | BF16 | 38678 |  26.1% |  28.9% |
| tc_04 WGMMA | BF16 | 120646 |  81.5% |  90.0% |
| tc_04 cuBLAS BF16 fair | BF16 | 133802 |  90.4% |  99.8% |
| tc_05 WGMMA_fp8 | FP8 | 226051 |  76.4% | 168.7% |

## 尺寸缩放
```
==== size=1024 ====
[FP32 cublas]     GFLOPS=21210.13
[FP32 double_buf] GFLOPS=14813.01
[BF16 cublas]     GFLOPS=91528.73
[WMMA tc_03_pipe] GFLOPS=28342.88
==== size=2048 ====
[FP32 cublas]     GFLOPS=22798.23
[FP32 double_buf] GFLOPS=21569.40
[BF16 cublas]     GFLOPS=119874.72
[WMMA tc_03_pipe] GFLOPS=36153.41
==== size=4096 ====
[FP32 cublas]     GFLOPS=28039.85
[FP32 double_buf] GFLOPS=23215.27
[BF16 cublas]     GFLOPS=134111.70
[WMMA tc_03_pipe] GFLOPS=38659.86
==== size=8192 ====
[FP32 cublas]     GFLOPS=29509.85
[FP32 double_buf] GFLOPS=22993.02
[BF16 cublas]     GFLOPS=139806.26
[WMMA tc_03_pipe] GFLOPS=39823.45
```

## 关键数字
- FP32 cuBLAS: **28.0 TFLOPS**（71% 峰值）
- FP32 手写最佳: **23.2 TFLOPS**（83% cuBLAS）
- BF16 cuBLAS: **134.0 TFLOPS**（91% BF16峰值）
- 手写张量核最佳(BF16 路径): **120.6 TFLOPS**（82% BF16峰值）
- 手写张量核最佳(FP8 路径): **226.1 TFLOPS**（76% FP8峰值；≈153% BF16峰值，因 FP8 吞吐是 BF16 的 2×）

> 时钟提示：GFLOPS 随实际 SM 时钟线性变化。本轮时钟策略=locked@1980MHz(额定boost)，实测计算期约 1830 MHz(峰 1830)。跨机型/跨轮对比 MFU 时务必统一时钟策略（`--lock` 锁额定频最可复现）。