# GEMM 测试汇总 — NVIDIA H20

- 计算能力: **CC 9.0**  |  SM 数: 78  |  FP32 核/SM: 128
- 时钟策略: **default(开箱自适应boost)**（额定 max 1980 MHz）
- 运行时实测 SM 时钟(遥测 pclk): 计算期典型 **1980 MHz** / 峰 1980 MHz（额定 1980）
- 理论峰值: FP32 = **39.5 TFLOPS**  |  BF16 张量核 = **148 TFLOPS**  |  FP8 张量核 = **296 TFLOPS**
- headline 尺寸: 4096³

## 正确性（对拍 cuBLAS）
- FP32: 13 PASS / 0 FAIL  |  BF16: 13 PASS / 0 FAIL  |  Tensor(WMMA/WGMMA): 5 PASS / 0 FAIL

## FP32 (CUDA core) @ 4096³
| kernel | GFLOPS | %FP32峰值 | %cuBLAS |
| --- | ---: | ---: | ---: |
| cublas_ref | 30266 |  76.6% | 100.0% |
| naive_kernel | 3532 |   8.9% |  11.7% |
| smem_kernel | 5411 |  13.7% |  17.9% |
| blocktiling_kernel | 9410 |  23.8% |  31.1% |
| Dblocktiling_kernel | 15552 |  39.3% |  51.4% |
| vectorized_kernel | 22763 |  57.6% |  75.2% |
| autotune_64x64x8_8x4 | 22779 |  57.6% |  75.3% |
| autotune_64x64x16_8x4 | 21849 |  55.3% |  72.2% |
| autotune_64x64x8_8x8 | 19833 |  50.2% |  65.5% |
| warptile_kernel | 12844 |  32.5% |  42.4% |
| warptile_vec_kernel | 23954 |  60.6% |  79.1% |
| bank_conflict_kernel | 21500 |  54.4% |  71.0% |
| double_buffer_kernel | 25132 |  63.6% |  83.0% |

## BF16 (CUDA core, bf16输入+fp32累加) @ 4096³
| kernel | GFLOPS | %BF16峰值 | %cuBLAS |
| --- | ---: | ---: | ---: |
| cublas_bf16 | 132130 |  89.3% | 100.0% |
| naive | 3230 |   2.2% |   2.4% |
| smem | 5477 |   3.7% |   4.1% |
| blocktiling | 10194 |   6.9% |   7.7% |
| 2Dblocktiling | 15870 |  10.7% |  12.0% |
| vectorized | 23611 |  16.0% |  17.9% |
| autotune_64x64x8_8x4 | 23622 |  16.0% |  17.9% |
| autotune_64x64x16_8x4 | 21936 |  14.8% |  16.6% |
| autotune_64x64x8_8x8 | 20291 |  13.7% |  15.4% |
| warptile | 12225 |   8.3% |   9.3% |
| warptile_vec | 24487 |  16.5% |  18.5% |
| bank_conflict | 22497 |  15.2% |  17.0% |
| double_buffer | 25889 |  17.5% |  19.6% |

## Tensor Core @ 4096³
> 利用率口径：WMMA/WGMMA(bf16)→BF16 峰值；FP8→FP8 峰值(=2×BF16)；FP4→FP4 峰值(=4×BF16)。
| 用例 | 精度 | GFLOPS | %对应精度峰值 | %cuBLAS_bf16 |
| --- | --- | ---: | ---: | ---: |
| tc_01 WMMA_naive | BF16 | 16687 |  11.3% |  12.6% |
| tc_02 WMMA_smem | BF16 | 29738 |  20.1% |  22.5% |
| tc_03 WMMA_pipe | BF16 | 38050 |  25.7% |  28.8% |
| tc_04 WGMMA | BF16 | 119322 |  80.6% |  90.3% |
| tc_04 cuBLAS BF16 fair | BF16 | 131850 |  89.1% |  99.8% |
| tc_05 WGMMA_fp8 | FP8 | 224172 |  75.7% | 169.7% |

## 尺寸缩放
```
==== size=1024 ====
[FP32 cublas]     GFLOPS=22907.96
[FP32 double_buf] GFLOPS=15898.43
[BF16 cublas]     GFLOPS=90370.14
[WMMA tc_03_pipe] GFLOPS=27701.17
==== size=2048 ====
[FP32 cublas]     GFLOPS=24700.64
[FP32 double_buf] GFLOPS=23286.63
[BF16 cublas]     GFLOPS=118449.18
[WMMA tc_03_pipe] GFLOPS=35590.43
==== size=4096 ====
[FP32 cublas]     GFLOPS=30267.46
[FP32 double_buf] GFLOPS=25130.49
[BF16 cublas]     GFLOPS=132041.92
[WMMA tc_03_pipe] GFLOPS=38060.03
==== size=8192 ====
[FP32 cublas]     GFLOPS=31991.15
[FP32 double_buf] GFLOPS=24738.53
[BF16 cublas]     GFLOPS=138085.36
[WMMA tc_03_pipe] GFLOPS=39162.23
```

## 关键数字
- FP32 cuBLAS: **30.3 TFLOPS**（77% 峰值）
- FP32 手写最佳: **25.1 TFLOPS**（83% cuBLAS）
- BF16 cuBLAS: **132.1 TFLOPS**（89% BF16峰值）
- 手写张量核最佳(BF16 路径): **119.3 TFLOPS**（81% BF16峰值）
- 手写张量核最佳(FP8 路径): **224.2 TFLOPS**（76% FP8峰值；≈151% BF16峰值，因 FP8 吞吐是 BF16 的 2×）

> 时钟提示：GFLOPS 随实际 SM 时钟线性变化。本轮时钟策略=default(开箱自适应boost)，实测计算期约 1980 MHz(峰 1980)。跨机型/跨轮对比 MFU 时务必统一时钟策略（`--lock` 锁额定频最可复现）。