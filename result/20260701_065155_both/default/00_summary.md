# GEMM 测试汇总 — NVIDIA A800-SXM4-80GB

- 计算能力: **CC 8.0**  |  SM 数: 108  |  FP32 核/SM: 64
- 时钟策略: **default(开箱自适应boost)**（额定 max 1410 MHz）
- 运行时实测 SM 时钟(遥测 pclk): 计算期典型 **1410 MHz** / 峰 1410 MHz（额定 1410）
- 理论峰值: FP32 = **19.5 TFLOPS**  |  BF16 张量核 = **312 TFLOPS**
- headline 尺寸: 4096³

## 正确性（对拍 cuBLAS）
- FP32: 13 PASS / 0 FAIL  |  BF16: 13 PASS / 0 FAIL  |  Tensor(WMMA/WGMMA): 3 PASS / 0 FAIL

## FP32 (CUDA core) @ 4096³
| kernel | GFLOPS | %FP32峰值 | %cuBLAS |
| --- | ---: | ---: | ---: |
| cublas_ref | 16045 |  82.3% | 100.0% |
| naive_kernel | 3045 |  15.6% |  19.0% |
| smem_kernel | 5349 |  27.4% |  33.3% |
| blocktiling_kernel | 10309 |  52.9% |  64.3% |
| Dblocktiling_kernel | 14428 |  74.0% |  89.9% |
| vectorized_kernel | 17614 |  90.4% | 109.8% |
| autotune_64x64x8_8x4 | 17605 |  90.3% | 109.7% |
| autotune_64x64x16_8x4 | 16816 |  86.3% | 104.8% |
| autotune_64x64x8_8x8 | 16780 |  86.1% | 104.6% |
| warptile_kernel | 12664 |  65.0% |  78.9% |
| warptile_vec_kernel | 17341 |  89.0% | 108.1% |
| bank_conflict_kernel | 17090 |  87.7% | 106.5% |
| double_buffer_kernel | 17056 |  87.5% | 106.3% |

## BF16 (CUDA core, bf16输入+fp32累加) @ 4096³
| kernel | GFLOPS | %BF16峰值 | %cuBLAS |
| --- | ---: | ---: | ---: |
| cublas_bf16 | 264468 |  84.8% | 100.0% |
| naive | 3103 |   1.0% |   1.2% |
| smem | 5332 |   1.7% |   2.0% |
| blocktiling | 10594 |   3.4% |   4.0% |
| 2Dblocktiling | 14335 |   4.6% |   5.4% |
| vectorized | 17786 |   5.7% |   6.7% |
| autotune_64x64x8_8x4 | 17773 |   5.7% |   6.7% |
| autotune_64x64x16_8x4 | 17009 |   5.5% |   6.4% |
| autotune_64x64x8_8x8 | 16161 |   5.2% |   6.1% |
| warptile | 12142 |   3.9% |   4.6% |
| warptile_vec | 17473 |   5.6% |   6.6% |
| bank_conflict | 17082 |   5.5% |   6.5% |
| double_buffer | 17250 |   5.5% |   6.5% |

## Tensor Core @ 4096³
> 利用率口径：WMMA/WGMMA(bf16)→BF16 峰值；FP8→FP8 峰值(=2×BF16)；FP4→FP4 峰值(=4×BF16)。
| 用例 | 精度 | GFLOPS | %对应精度峰值 | %cuBLAS_bf16 |
| --- | --- | ---: | ---: | ---: |
| tc_01 WMMA_naive | BF16 | 17779 |   5.7% |   6.7% |
| tc_02 WMMA_smem | BF16 | 27111 |   8.7% |  10.3% |
| tc_03 WMMA_pipe | BF16 | 43030 |  13.8% |  16.3% |

## 尺寸缩放
```
==== size=1024 ====
[FP32 cublas]     GFLOPS=16565.18
[FP32 double_buf] GFLOPS=8500.82
[BF16 cublas]     GFLOPS=109798.53
[WMMA tc_03_pipe] GFLOPS=18404.14
==== size=2048 ====
[FP32 cublas]     GFLOPS=17121.36
[FP32 double_buf] GFLOPS=14553.45
[BF16 cublas]     GFLOPS=148734.18
[WMMA tc_03_pipe] GFLOPS=31996.21
==== size=4096 ====
[FP32 cublas]     GFLOPS=15416.87
[FP32 double_buf] GFLOPS=14253.16
[BF16 cublas]     GFLOPS=250593.22
[WMMA tc_03_pipe] GFLOPS=40798.76
==== size=8192 ====
[FP32 cublas]     GFLOPS=19144.87
[FP32 double_buf] GFLOPS=17439.11
[BF16 cublas]     GFLOPS=294482.42
[WMMA tc_03_pipe] GFLOPS=42183.45
```

## 关键数字
- FP32 cuBLAS: **16.0 TFLOPS**（82% 峰值）
- FP32 手写最佳: **17.6 TFLOPS**（110% cuBLAS）
- BF16 cuBLAS: **264.5 TFLOPS**（85% BF16峰值）
- 手写张量核最佳(BF16 路径): **43.0 TFLOPS**（14% BF16峰值）

> 时钟提示：GFLOPS 随实际 SM 时钟线性变化。本轮时钟策略=default(开箱自适应boost)，实测计算期约 1410 MHz(峰 1410)。跨机型/跨轮对比 MFU 时务必统一时钟策略（`--lock` 锁额定频最可复现）。