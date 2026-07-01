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
| cublas_ref | 15416 |  79.1% | 100.0% |
| naive_kernel | 3051 |  15.7% |  19.8% |
| smem_kernel | 5349 |  27.4% |  34.7% |
| blocktiling_kernel | 10297 |  52.8% |  66.8% |
| Dblocktiling_kernel | 14434 |  74.1% |  93.6% |
| vectorized_kernel | 17617 |  90.4% | 114.3% |
| autotune_64x64x8_8x4 | 17615 |  90.4% | 114.3% |
| autotune_64x64x16_8x4 | 16823 |  86.3% | 109.1% |
| autotune_64x64x8_8x8 | 16782 |  86.1% | 108.9% |
| warptile_kernel | 12668 |  65.0% |  82.2% |
| warptile_vec_kernel | 17352 |  89.0% | 112.6% |
| bank_conflict_kernel | 17096 |  87.7% | 110.9% |
| double_buffer_kernel | 17063 |  87.5% | 110.7% |

## BF16 (CUDA core, bf16输入+fp32累加) @ 4096³
| kernel | GFLOPS | %BF16峰值 | %cuBLAS |
| --- | ---: | ---: | ---: |
| cublas_bf16 | 264521 |  84.8% | 100.0% |
| naive | 3106 |   1.0% |   1.2% |
| smem | 5334 |   1.7% |   2.0% |
| blocktiling | 10597 |   3.4% |   4.0% |
| 2Dblocktiling | 14342 |   4.6% |   5.4% |
| vectorized | 17782 |   5.7% |   6.7% |
| autotune_64x64x8_8x4 | 17778 |   5.7% |   6.7% |
| autotune_64x64x16_8x4 | 17024 |   5.5% |   6.4% |
| autotune_64x64x8_8x8 | 16165 |   5.2% |   6.1% |
| warptile | 12132 |   3.9% |   4.6% |
| warptile_vec | 17482 |   5.6% |   6.6% |
| bank_conflict | 17088 |   5.5% |   6.5% |
| double_buffer | 17259 |   5.5% |   6.5% |

## Tensor Core @ 4096³
| 用例 | GFLOPS | %BF16峰值 | %cuBLAS_bf16 |
| --- | ---: | ---: | ---: |
| tc_01 WMMA_naive | 17802 |   5.7% |   6.7% |
| tc_02 WMMA_smem | 27075 |   8.7% |  10.2% |
| tc_03 WMMA_pipe | 43039 |  13.8% |  16.3% |

## 尺寸缩放
```
==== size=1024 ====
[FP32 cublas]     GFLOPS=16526.02
[FP32 double_buf] GFLOPS=8514.63
[BF16 cublas]     GFLOPS=110960.42
[WMMA tc_03_pipe] GFLOPS=18396.07
==== size=2048 ====
[FP32 cublas]     GFLOPS=14354.22
[FP32 double_buf] GFLOPS=12238.10
[BF16 cublas]     GFLOPS=124367.79
[WMMA tc_03_pipe] GFLOPS=31990.12
==== size=4096 ====
[FP32 cublas]     GFLOPS=15416.16
[FP32 double_buf] GFLOPS=17061.29
[BF16 cublas]     GFLOPS=264677.03
[WMMA tc_03_pipe] GFLOPS=34851.79
==== size=8192 ====
[FP32 cublas]     GFLOPS=19170.16
[FP32 double_buf] GFLOPS=17436.90
[BF16 cublas]     GFLOPS=294458.20
[WMMA tc_03_pipe] GFLOPS=41102.61
```

## 关键数字
- FP32 cuBLAS: **15.4 TFLOPS**（79% 峰值）
- FP32 手写最佳: **17.6 TFLOPS**（114% cuBLAS）
- BF16 cuBLAS: **264.5 TFLOPS**（85% BF16峰值）
- 手写张量核最佳: **43.0 TFLOPS**（14% BF16峰值）

> 时钟提示：GFLOPS 随实际 SM 时钟线性变化。本轮时钟策略=default(开箱自适应boost)，实测计算期约 1410 MHz(峰 1410)。跨机型/跨轮对比 MFU 时务必统一时钟策略（`--lock` 锁额定频最可复现）。