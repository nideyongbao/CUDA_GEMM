# GEMM 测试汇总 — NVIDIA A10

- 计算能力: **CC 8.6**  |  SM 数: 72  |  FP32 核/SM: 128
- 时钟策略: **default(开箱自适应boost)**（额定 max 1695 MHz）
- 运行时实测 SM 时钟(遥测 pclk): 计算期典型 **1695 MHz** / 峰 1695 MHz（额定 1695）
- 理论峰值: FP32 = **31.2 TFLOPS**  |  BF16 张量核 = **125 TFLOPS**
- headline 尺寸: 4096³

## 正确性（对拍 cuBLAS）
- FP32: 13 PASS / 0 FAIL  |  BF16: 13 PASS / 0 FAIL  |  Tensor(WMMA/WGMMA): 3 PASS / 0 FAIL

## FP32 (CUDA core) @ 4096³
| kernel | GFLOPS | %FP32峰值 | %cuBLAS |
| --- | ---: | ---: | ---: |
| cublas_ref | 14050 |  45.0% | 100.0% |
| naive_kernel | 1450 |   4.6% |  10.3% |
| smem_kernel | 2110 |   6.8% |  15.0% |
| blocktiling_kernel | 5273 |  16.9% |  37.5% |
| Dblocktiling_kernel | 7967 |  25.5% |  56.7% |
| vectorized_kernel | 9515 |  30.5% |  67.7% |
| autotune_64x64x8_8x4 | 9520 |  30.5% |  67.8% |
| autotune_64x64x16_8x4 | 9575 |  30.6% |  68.2% |
| autotune_64x64x8_8x8 | 10596 |  33.9% |  75.4% |
| warptile_kernel | 8821 |  28.2% |  62.8% |
| warptile_vec_kernel | 11643 |  37.3% |  82.9% |
| bank_conflict_kernel | 11531 |  36.9% |  82.1% |
| double_buffer_kernel | 11710 |  37.5% |  83.3% |

## BF16 (CUDA core, bf16输入+fp32累加) @ 4096³
| kernel | GFLOPS | %BF16峰值 | %cuBLAS |
| --- | ---: | ---: | ---: |
| cublas_bf16 | 92698 |  74.2% | 100.0% |
| naive | 1278 |   1.0% |   1.4% |
| smem | 2170 |   1.7% |   2.3% |
| blocktiling | 5877 |   4.7% |   6.3% |
| 2Dblocktiling | 9437 |   7.5% |  10.2% |
| vectorized | 12438 |  10.0% |  13.4% |
| autotune_64x64x8_8x4 | 12599 |  10.1% |  13.6% |
| autotune_64x64x16_8x4 | 12527 |  10.0% |  13.5% |
| autotune_64x64x8_8x8 | 12209 |   9.8% |  13.2% |
| warptile | 9325 |   7.5% |  10.1% |
| warptile_vec | 14690 |  11.8% |  15.8% |
| bank_conflict | 14650 |  11.7% |  15.8% |
| double_buffer | 14364 |  11.5% |  15.5% |

## Tensor Core @ 4096³
> 利用率口径：WMMA/WGMMA(bf16)→BF16 峰值；FP8→FP8 峰值(=2×BF16)；FP4→FP4 峰值(=4×BF16)。
| 用例 | 精度 | GFLOPS | %对应精度峰值 | %cuBLAS_bf16 |
| --- | --- | ---: | ---: | ---: |
| tc_01 WMMA_naive | BF16 | 10276 |   8.2% |  11.1% |
| tc_02 WMMA_smem | BF16 | 16305 |  13.0% |  17.6% |
| tc_03 WMMA_pipe | BF16 | 25150 |  20.1% |  27.1% |

## 尺寸缩放
```
==== size=1024 ====
[FP32 cublas]     GFLOPS=13264.72
[FP32 double_buf] GFLOPS=11015.54
[BF16 cublas]     GFLOPS=77430.33
[WMMA tc_03_pipe] GFLOPS=24981.43
==== size=2048 ====
[FP32 cublas]     GFLOPS=18295.76
[FP32 double_buf] GFLOPS=14118.67
[BF16 cublas]     GFLOPS=88909.46
[WMMA tc_03_pipe] GFLOPS=27068.76
==== size=4096 ====
[FP32 cublas]     GFLOPS=13747.59
[FP32 double_buf] GFLOPS=11553.26
[BF16 cublas]     GFLOPS=92595.88
[WMMA tc_03_pipe] GFLOPS=25245.54
==== size=8192 ====
[FP32 cublas]     GFLOPS=15301.37
[FP32 double_buf] GFLOPS=9544.65
[BF16 cublas]     GFLOPS=91092.34
[WMMA tc_03_pipe] GFLOPS=23644.84
```

## 关键数字
- FP32 cuBLAS: **14.1 TFLOPS**（45% 峰值）
- FP32 手写最佳: **11.7 TFLOPS**（83% cuBLAS）
- BF16 cuBLAS: **92.7 TFLOPS**（74% BF16峰值）
- 手写张量核最佳(BF16 路径): **25.1 TFLOPS**（20% BF16峰值）

> 时钟提示：GFLOPS 随实际 SM 时钟线性变化。本轮时钟策略=default(开箱自适应boost)，实测计算期约 1695 MHz(峰 1695)。跨机型/跨轮对比 MFU 时务必统一时钟策略（`--lock` 锁额定频最可复现）。