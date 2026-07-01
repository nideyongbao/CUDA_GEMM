# GEMM 测试汇总 — NVIDIA A800-SXM4-80GB

- 计算能力: **CC 8.0**  |  SM 数: 108  |  FP32 核/SM: 64
- 时钟策略: **locked@1410MHz(额定boost)**（额定 max 1410 MHz）
- 运行时实测 SM 时钟(遥测 pclk): 计算期典型 **1410 MHz** / 峰 1410 MHz（额定 1410）
- 理论峰值: FP32 = **19.5 TFLOPS**  |  BF16 张量核 = **312 TFLOPS**
- headline 尺寸: 4096³

## 正确性（对拍 cuBLAS）
- FP32: 13 PASS / 0 FAIL  |  BF16: 13 PASS / 0 FAIL  |  Tensor(WMMA/WGMMA): 3 PASS / 0 FAIL

## FP32 (CUDA core) @ 4096³
| kernel | GFLOPS | %FP32峰值 | %cuBLAS |
| --- | ---: | ---: | ---: |
| cublas_ref | 19030 |  97.6% | 100.0% |
| naive_kernel | 3050 |  15.6% |  16.0% |
| smem_kernel | 5351 |  27.5% |  28.1% |
| blocktiling_kernel | 10324 |  53.0% |  54.3% |
| Dblocktiling_kernel | 14433 |  74.0% |  75.8% |
| vectorized_kernel | 17628 |  90.4% |  92.6% |
| autotune_64x64x8_8x4 | 17608 |  90.3% |  92.5% |
| autotune_64x64x16_8x4 | 16823 |  86.3% |  88.4% |
| autotune_64x64x8_8x8 | 16782 |  86.1% |  88.2% |
| warptile_kernel | 12666 |  65.0% |  66.6% |
| warptile_vec_kernel | 17350 |  89.0% |  91.2% |
| bank_conflict_kernel | 17100 |  87.7% |  89.9% |
| double_buffer_kernel | 17061 |  87.5% |  89.7% |

## BF16 (CUDA core, bf16输入+fp32累加) @ 4096³
| kernel | GFLOPS | %BF16峰值 | %cuBLAS |
| --- | ---: | ---: | ---: |
| cublas_bf16 | 264729 |  84.8% | 100.0% |
| naive | 3107 |   1.0% |   1.2% |
| smem | 5334 |   1.7% |   2.0% |
| blocktiling | 10596 |   3.4% |   4.0% |
| 2Dblocktiling | 14341 |   4.6% |   5.4% |
| vectorized | 17790 |   5.7% |   6.7% |
| autotune_64x64x8_8x4 | 17795 |   5.7% |   6.7% |
| autotune_64x64x16_8x4 | 17026 |   5.5% |   6.4% |
| autotune_64x64x8_8x8 | 16163 |   5.2% |   6.1% |
| warptile | 12144 |   3.9% |   4.6% |
| warptile_vec | 17476 |   5.6% |   6.6% |
| bank_conflict | 17084 |   5.5% |   6.5% |
| double_buffer | 17257 |   5.5% |   6.5% |

## Tensor Core @ 4096³
| 用例 | GFLOPS | %BF16峰值 | %cuBLAS_bf16 |
| --- | ---: | ---: | ---: |
| tc_01 WMMA_naive | 17819 |   5.7% |   6.7% |
| tc_02 WMMA_smem | 27102 |   8.7% |  10.2% |
| tc_03 WMMA_pipe | 43046 |  13.8% |  16.3% |

## 尺寸缩放
```
==== size=1024 ====
[FP32 cublas]      GFLOPS=16552.11
[FP32 double_buf]  GFLOPS=8511.17
[BF16 cublas]      GFLOPS=110960.42
[WMMA tc_03_pipe]  GFLOPS=22745.68
==== size=2048 ====
[FP32 cublas]      GFLOPS=17662.09
[FP32 double_buf]  GFLOPS=15005.11
[BF16 cublas]      GFLOPS=153356.64
[WMMA tc_03_pipe]  GFLOPS=39512.99
==== size=4096 ====
[FP32 cublas]      GFLOPS=19019.36
[FP32 double_buf]  GFLOPS=17062.16
[BF16 cublas]      GFLOPS=264520.54
[WMMA tc_03_pipe]  GFLOPS=43041.95
==== size=8192 ====
[FP32 cublas]      GFLOPS=19164.76
[FP32 double_buf]  GFLOPS=17437.50
[BF16 cublas]      GFLOPS=294611.70
[WMMA tc_03_pipe]  GFLOPS=43486.29
```

## 关键数字
- FP32 cuBLAS: **19.0 TFLOPS**（98% 峰值）
- FP32 手写最佳: **17.6 TFLOPS**（93% cuBLAS）
- BF16 cuBLAS: **264.7 TFLOPS**（85% BF16峰值）
- 手写张量核最佳: **43.0 TFLOPS**（14% BF16峰值）

> 时钟提示：GFLOPS 随实际 SM 时钟线性变化。本轮时钟策略=locked@1410MHz(额定boost)，实测计算期约 1410 MHz(峰 1410)。跨机型/跨轮对比 MFU 时务必统一时钟策略（`--lock` 锁额定频最可复现）。