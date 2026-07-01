# GEMM 测试汇总 — NVIDIA A800-SXM4-80GB

- 计算能力: **CC 8.0**  |  SM 数: 108  |  FP32 核/SM: 64
- 时钟策略: **default(开箱自适应boost)**（额定 max 1410 MHz）
- 运行时实测 SM 时钟(遥测 pclk): 计算期典型 **1290 MHz** / 峰 1410 MHz（额定 1410）
- 理论峰值: FP32 = **19.5 TFLOPS**  |  BF16 张量核 = **312 TFLOPS**
- headline 尺寸: 4096³

## 正确性（对拍 cuBLAS）
- FP32: 13 PASS / 0 FAIL  |  BF16: 13 PASS / 0 FAIL  |  Tensor(WMMA/WGMMA): 4 PASS / 0 FAIL

## FP32 (CUDA core) @ 4096³
| kernel | GFLOPS | %FP32峰值 | %cuBLAS |
| --- | ---: | ---: | ---: |
| cublas_ref | 15390 |  79.0% | 100.0% |
| naive_kernel | 3054 |  15.7% |  19.8% |
| smem_kernel | 5348 |  27.4% |  34.7% |
| blocktiling_kernel | 9565 |  49.1% |  62.2% |
| Dblocktiling_kernel | 14443 |  74.1% |  93.8% |
| vectorized_kernel | 17624 |  90.4% | 114.5% |
| autotune_64x64x8_8x4 | 17626 |  90.4% | 114.5% |
| autotune_64x64x16_8x4 | 16833 |  86.4% | 109.4% |
| autotune_64x64x8_8x8 | 16790 |  86.1% | 109.1% |
| warptile_kernel | 12678 |  65.0% |  82.4% |
| warptile_vec_kernel | 17362 |  89.1% | 112.8% |
| bank_conflict_kernel | 17105 |  87.8% | 111.1% |
| double_buffer_kernel | 17073 |  87.6% | 110.9% |

## BF16 (CUDA core, bf16输入+fp32累加) @ 4096³
| kernel | GFLOPS | %BF16峰值 | %cuBLAS |
| --- | ---: | ---: | ---: |
| cublas_bf16 | 264573 |  84.8% | 100.0% |
| naive | 3107 |   1.0% |   1.2% |
| smem | 5330 |   1.7% |   2.0% |
| blocktiling | 10606 |   3.4% |   4.0% |
| 2Dblocktiling | 14345 |   4.6% |   5.4% |
| vectorized | 17798 |   5.7% |   6.7% |
| autotune_64x64x8_8x4 | 17799 |   5.7% |   6.7% |
| autotune_64x64x16_8x4 | 17036 |   5.5% |   6.4% |
| autotune_64x64x8_8x8 | 16170 |   5.2% |   6.1% |
| warptile | 12161 |   3.9% |   4.6% |
| warptile_vec | 17481 |   5.6% |   6.6% |
| bank_conflict | 17099 |   5.5% |   6.5% |
| double_buffer | 17260 |   5.5% |   6.5% |

## Tensor Core @ 4096³
> 利用率口径：WMMA/WGMMA(bf16)→BF16 峰值；FP8→FP8 峰值(=2×BF16)；FP4→FP4 峰值(=4×BF16)。
| 用例 | 精度 | GFLOPS | %对应精度峰值 | %cuBLAS_bf16 |
| --- | --- | ---: | ---: | ---: |
| tc_01 WMMA_naive | BF16 | 17824 |   5.7% |   6.7% |
| tc_02 WMMA_smem | BF16 | 27104 |   8.7% |  10.2% |
| tc_03 WMMA_pipe | BF16 | 43084 |  13.8% |  16.3% |
| tc_06 MMA_pipe | BF16 | 120808 |  38.7% |  45.7% |

## 尺寸缩放
```
==== size=1024 ====
[FP32 cublas]     GFLOPS=13366.17
[FP32 double_buf] GFLOPS=6860.16
[BF16 cublas]     GFLOPS=89240.51
[WMMA tc_03_pipe] GFLOPS=18379.95
[mma. tc_06_pipe] GFLOPS=41404.78
==== size=2048 ====
[FP32 cublas]     GFLOPS=14343.18
[FP32 double_buf] GFLOPS=12224.73
[BF16 cublas]     GFLOPS=124183.68
[WMMA tc_03_pipe] GFLOPS=31920.12
[mma. tc_06_pipe] GFLOPS=80005.80
==== size=4096 ====
[FP32 cublas]     GFLOPS=15386.12
[FP32 double_buf] GFLOPS=13868.19
[BF16 cublas]     GFLOPS=242270.28
[WMMA tc_03_pipe] GFLOPS=39403.37
[mma. tc_06_pipe] GFLOPS=150308.22
==== size=8192 ====
[FP32 cublas]     GFLOPS=18704.28
[FP32 double_buf] GFLOPS=17442.37
[BF16 cublas]     GFLOPS=294741.10
[WMMA tc_03_pipe] GFLOPS=42346.41
[mma. tc_06_pipe] GFLOPS=118732.75
```

## 关键数字
- FP32 cuBLAS: **15.4 TFLOPS**（79% 峰值）
- FP32 手写最佳: **17.6 TFLOPS**（115% cuBLAS）
- BF16 cuBLAS: **264.6 TFLOPS**（85% BF16峰值）
- 手写张量核最佳(BF16 路径): **120.8 TFLOPS**（39% BF16峰值）

> 时钟提示：GFLOPS 随实际 SM 时钟线性变化。本轮时钟策略=default(开箱自适应boost)，实测计算期约 1290 MHz(峰 1410)。跨机型/跨轮对比 MFU 时务必统一时钟策略（`--lock` 锁额定频最可复现）。