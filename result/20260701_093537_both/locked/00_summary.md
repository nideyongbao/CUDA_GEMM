# GEMM 测试汇总 — NVIDIA A800-SXM4-80GB

- 计算能力: **CC 8.0**  |  SM 数: 108  |  FP32 核/SM: 64
- 时钟策略: **locked@1410MHz(额定boost)**（额定 max 1410 MHz）
- 运行时实测 SM 时钟(遥测 pclk): 计算期典型 **1410 MHz** / 峰 1410 MHz（额定 1410）
- 理论峰值: FP32 = **19.5 TFLOPS**  |  BF16 张量核 = **312 TFLOPS**
- headline 尺寸: 4096³

## 正确性（对拍 cuBLAS）
- FP32: 13 PASS / 0 FAIL  |  BF16: 13 PASS / 0 FAIL  |  Tensor(WMMA/WGMMA): 4 PASS / 0 FAIL

## FP32 (CUDA core) @ 4096³
| kernel | GFLOPS | %FP32峰值 | %cuBLAS |
| --- | ---: | ---: | ---: |
| cublas_ref | 19034 |  97.7% | 100.0% |
| naive_kernel | 3055 |  15.7% |  16.0% |
| smem_kernel | 5346 |  27.4% |  28.1% |
| blocktiling_kernel | 10304 |  52.9% |  54.1% |
| Dblocktiling_kernel | 14446 |  74.1% |  75.9% |
| vectorized_kernel | 17629 |  90.4% |  92.6% |
| autotune_64x64x8_8x4 | 17631 |  90.5% |  92.6% |
| autotune_64x64x16_8x4 | 16840 |  86.4% |  88.5% |
| autotune_64x64x8_8x8 | 16794 |  86.2% |  88.2% |
| warptile_kernel | 12676 |  65.0% |  66.6% |
| warptile_vec_kernel | 17363 |  89.1% |  91.2% |
| bank_conflict_kernel | 17109 |  87.8% |  89.9% |
| double_buffer_kernel | 17073 |  87.6% |  89.7% |

## BF16 (CUDA core, bf16输入+fp32累加) @ 4096³
| kernel | GFLOPS | %BF16峰值 | %cuBLAS |
| --- | ---: | ---: | ---: |
| cublas_bf16 | 264677 |  84.8% | 100.0% |
| naive | 3104 |   1.0% |   1.2% |
| smem | 5330 |   1.7% |   2.0% |
| blocktiling | 10605 |   3.4% |   4.0% |
| 2Dblocktiling | 14344 |   4.6% |   5.4% |
| vectorized | 17795 |   5.7% |   6.7% |
| autotune_64x64x8_8x4 | 17796 |   5.7% |   6.7% |
| autotune_64x64x16_8x4 | 17042 |   5.5% |   6.4% |
| autotune_64x64x8_8x8 | 16171 |   5.2% |   6.1% |
| warptile | 12145 |   3.9% |   4.6% |
| warptile_vec | 17486 |   5.6% |   6.6% |
| bank_conflict | 17096 |   5.5% |   6.5% |
| double_buffer | 17264 |   5.5% |   6.5% |

## Tensor Core @ 4096³
> 利用率口径：WMMA/WGMMA(bf16)→BF16 峰值；FP8→FP8 峰值(=2×BF16)；FP4→FP4 峰值(=4×BF16)。
| 用例 | 精度 | GFLOPS | %对应精度峰值 | %cuBLAS_bf16 |
| --- | --- | ---: | ---: | ---: |
| tc_01 WMMA_naive | BF16 | 17829 |   5.7% |   6.7% |
| tc_02 WMMA_smem | BF16 | 27139 |   8.7% |  10.3% |
| tc_03 WMMA_pipe | BF16 | 43080 |  13.8% |  16.3% |
| tc_06 MMA_pipe | BF16 | 150333 |  48.2% |  56.8% |

## 尺寸缩放
```
==== size=1024 ====
[FP32 cublas]     GFLOPS=16552.11
[FP32 double_buf] GFLOPS=8500.82
[BF16 cublas]     GFLOPS=111550.63
[WMMA tc_03_pipe] GFLOPS=22733.36
[mma. tc_06_pipe] GFLOPS=51463.86
==== size=2048 ====
[FP32 cublas]     GFLOPS=17665.81
[FP32 double_buf] GFLOPS=15011.83
[BF16 cublas]     GFLOPS=153778.33
[WMMA tc_03_pipe] GFLOPS=39526.96
[mma. tc_06_pipe] GFLOPS=99273.47
==== size=4096 ====
[FP32 cublas]     GFLOPS=19040.14
[FP32 double_buf] GFLOPS=17073.01
[BF16 cublas]     GFLOPS=264781.46
[WMMA tc_03_pipe] GFLOPS=43079.26
[mma. tc_06_pipe] GFLOPS=150367.15
==== size=8192 ====
[FP32 cublas]     GFLOPS=19154.98
[FP32 double_buf] GFLOPS=17438.29
[BF16 cublas]     GFLOPS=294749.20
[WMMA tc_03_pipe] GFLOPS=43495.19
[mma. tc_06_pipe] GFLOPS=122948.00
```

## 关键数字
- FP32 cuBLAS: **19.0 TFLOPS**（98% 峰值）
- FP32 手写最佳: **17.6 TFLOPS**（93% cuBLAS）
- BF16 cuBLAS: **264.7 TFLOPS**（85% BF16峰值）
- 手写张量核最佳(BF16 路径): **150.3 TFLOPS**（48% BF16峰值）

> 时钟提示：GFLOPS 随实际 SM 时钟线性变化。本轮时钟策略=locked@1410MHz(额定boost)，实测计算期约 1410 MHz(峰 1410)。跨机型/跨轮对比 MFU 时务必统一时钟策略（`--lock` 锁额定频最可复现）。