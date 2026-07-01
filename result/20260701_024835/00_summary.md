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
| cublas_ref | 15999 |  82.1% | 100.0% |
| naive_kernel | 3050 |  15.6% |  19.1% |
| smem_kernel | 5348 |  27.4% |  33.4% |
| blocktiling_kernel | 10308 |  52.9% |  64.4% |
| Dblocktiling_kernel | 14422 |  74.0% |  90.1% |
| vectorized_kernel | 17593 |  90.3% | 110.0% |
| autotune_64x64x8_8x4 | 17595 |  90.3% | 110.0% |
| autotune_64x64x16_8x4 | 16818 |  86.3% | 105.1% |
| autotune_64x64x8_8x8 | 16777 |  86.1% | 104.9% |
| warptile_kernel | 12667 |  65.0% |  79.2% |
| warptile_vec_kernel | 17344 |  89.0% | 108.4% |
| bank_conflict_kernel | 17093 |  87.7% | 106.8% |
| double_buffer_kernel | 17055 |  87.5% | 106.6% |

## BF16 (CUDA core, bf16输入+fp32累加) @ 4096³
| kernel | GFLOPS | %BF16峰值 | %cuBLAS |
| --- | ---: | ---: | ---: |
| cublas_bf16 | 264416 |  84.7% | 100.0% |
| naive | 3105 |   1.0% |   1.2% |
| smem | 5333 |   1.7% |   2.0% |
| blocktiling | 10601 |   3.4% |   4.0% |
| 2Dblocktiling | 14333 |   4.6% |   5.4% |
| vectorized | 17764 |   5.7% |   6.7% |
| autotune_64x64x8_8x4 | 17777 |   5.7% |   6.7% |
| autotune_64x64x16_8x4 | 17006 |   5.5% |   6.4% |
| autotune_64x64x8_8x8 | 16156 |   5.2% |   6.1% |
| warptile | 12140 |   3.9% |   4.6% |
| warptile_vec | 17474 |   5.6% |   6.6% |
| bank_conflict | 17079 |   5.5% |   6.5% |
| double_buffer | 17255 |   5.5% |   6.5% |

## Tensor Core @ 4096³
| 用例 | GFLOPS | %BF16峰值 | %cuBLAS_bf16 |
| --- | ---: | ---: | ---: |
| tc_01 WMMA_naive | 17798 |   5.7% |   6.7% |
| tc_02 WMMA_smem | 27035 |   8.7% |  10.2% |
| tc_03 WMMA_pipe | 43036 |  13.8% |  16.3% |

## 尺寸缩放
```
==== size=1024 ====
[FP32 cublas]      GFLOPS=16552.11
[FP32 double_buf]  GFLOPS=8507.72
[BF16 cublas]      GFLOPS=110376.42
[WMMA tc_03_pipe]  GFLOPS=18420.31
==== size=2048 ====
[FP32 cublas]      GFLOPS=14355.45
[FP32 double_buf]  GFLOPS=12250.61
[BF16 cublas]      GFLOPS=124367.79
[WMMA tc_03_pipe]  GFLOPS=31999.27
==== size=4096 ====
[FP32 cublas]      GFLOPS=15417.40
[FP32 double_buf]  GFLOPS=17057.38
[BF16 cublas]      GFLOPS=264468.42
[WMMA tc_03_pipe]  GFLOPS=34853.15
==== size=8192 ====
[FP32 cublas]      GFLOPS=19140.91
[FP32 double_buf]  GFLOPS=17439.40
[BF16 cublas]      GFLOPS=294595.52
[WMMA tc_03_pipe]  GFLOPS=42986.04
```

## 关键数字
- FP32 cuBLAS: **16.0 TFLOPS**（82% 峰值）
- FP32 手写最佳: **17.6 TFLOPS**（110% cuBLAS）
- BF16 cuBLAS: **264.4 TFLOPS**（85% BF16峰值）
- 手写张量核最佳: **43.0 TFLOPS**（14% BF16峰值）

> 时钟提示：GFLOPS 随实际 SM 时钟线性变化。本轮时钟策略=default(开箱自适应boost)，实测计算期约 1410 MHz(峰 1410)。跨机型/跨轮对比 MFU 时务必统一时钟策略（`--lock` 锁额定频最可复现）。