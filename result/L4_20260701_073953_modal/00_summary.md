# GEMM 测试汇总 — NVIDIA L4

- 计算能力: **CC 8.9**  |  SM 数: 58  |  FP32 核/SM: 128
- 时钟策略: **default(开箱自适应boost)**（额定 max 2040 MHz）
- 运行时实测 SM 时钟(遥测 pclk): 计算期典型 **2040 MHz** / 峰 2040 MHz（额定 2040）
- 理论峰值: FP32 = **30.3 TFLOPS**  |  BF16 张量核 = **121 TFLOPS**  |  FP8 张量核 = **242 TFLOPS**
- headline 尺寸: 4096³

## 正确性（对拍 cuBLAS）
- FP32: 13 PASS / 0 FAIL  |  BF16: 13 PASS / 0 FAIL  |  Tensor(WMMA/WGMMA): 3 PASS / 0 FAIL

## FP32 (CUDA core) @ 4096³
| kernel | GFLOPS | %FP32峰值 | %cuBLAS |
| --- | ---: | ---: | ---: |
| cublas_ref | 13280 |  43.8% | 100.0% |
| naive_kernel | 1208 |   4.0% |   9.1% |
| smem_kernel | 1749 |   5.8% |  13.2% |
| blocktiling_kernel | 4149 |  13.7% |  31.2% |
| Dblocktiling_kernel | 6550 |  21.6% |  49.3% |
| vectorized_kernel | 7024 |  23.2% |  52.9% |
| autotune_64x64x8_8x4 | 6755 |  22.3% |  50.9% |
| autotune_64x64x16_8x4 | 6894 |  22.8% |  51.9% |
| autotune_64x64x8_8x8 | 8224 |  27.2% |  61.9% |
| warptile_kernel | 7499 |  24.8% |  56.5% |
| warptile_vec_kernel | 9444 |  31.2% |  71.1% |
| bank_conflict_kernel | 9274 |  30.6% |  69.8% |
| double_buffer_kernel | 9263 |  30.6% |  69.8% |

## BF16 (CUDA core, bf16输入+fp32累加) @ 4096³
| kernel | GFLOPS | %BF16峰值 | %cuBLAS |
| --- | ---: | ---: | ---: |
| cublas_bf16 | 83510 |  69.0% | 100.0% |
| naive | 1415 |   1.2% |   1.7% |
| smem | 2472 |   2.0% |   3.0% |
| blocktiling | 6331 |   5.2% |   7.6% |
| 2Dblocktiling | 9319 |   7.7% |  11.2% |
| vectorized | 12474 |  10.3% |  14.9% |
| autotune_64x64x8_8x4 | 12579 |  10.4% |  15.1% |
| autotune_64x64x16_8x4 | 12154 |  10.0% |  14.6% |
| autotune_64x64x8_8x8 | 12149 |  10.0% |  14.5% |
| warptile | 9325 |   7.7% |  11.2% |
| warptile_vec | 14424 |  11.9% |  17.3% |
| bank_conflict | 13861 |  11.5% |  16.6% |
| double_buffer | 14185 |  11.7% |  17.0% |

## Tensor Core @ 4096³
> 利用率口径：WMMA/WGMMA(bf16)→BF16 峰值；FP8→FP8 峰值(=2×BF16)；FP4→FP4 峰值(=4×BF16)。
| 用例 | 精度 | GFLOPS | %对应精度峰值 | %cuBLAS_bf16 |
| --- | --- | ---: | ---: | ---: |
| tc_01 WMMA_naive | BF16 | 11712 |   9.7% |  14.0% |
| tc_02 WMMA_smem | BF16 | 16535 |  13.7% |  19.8% |
| tc_03 WMMA_pipe | BF16 | 25398 |  21.0% |  30.4% |

## 尺寸缩放
```
==== size=1024 ====
[FP32 cublas]     GFLOPS=13058.23
[FP32 double_buf] GFLOPS=12150.36
[BF16 cublas]     GFLOPS=56527.01
[WMMA tc_03_pipe] GFLOPS=20098.19
==== size=2048 ====
[FP32 cublas]     GFLOPS=16586.47
[FP32 double_buf] GFLOPS=13863.18
[BF16 cublas]     GFLOPS=80504.87
[WMMA tc_03_pipe] GFLOPS=28883.90
==== size=4096 ====
[FP32 cublas]     GFLOPS=13207.94
[FP32 double_buf] GFLOPS=9207.50
[BF16 cublas]     GFLOPS=83396.13
[WMMA tc_03_pipe] GFLOPS=25521.76
==== size=8192 ====
[FP32 cublas]     GFLOPS=11514.35
[FP32 double_buf] GFLOPS=6476.37
[BF16 cublas]     GFLOPS=66736.81
[WMMA tc_03_pipe] GFLOPS=18772.48
```

## 关键数字
- FP32 cuBLAS: **13.3 TFLOPS**（44% 峰值）
- FP32 手写最佳: **9.4 TFLOPS**（71% cuBLAS）
- BF16 cuBLAS: **83.5 TFLOPS**（69% BF16峰值）
- 手写张量核最佳(BF16 路径): **25.4 TFLOPS**（21% BF16峰值）

> 时钟提示：GFLOPS 随实际 SM 时钟线性变化。本轮时钟策略=default(开箱自适应boost)，实测计算期约 2040 MHz(峰 2040)。跨机型/跨轮对比 MFU 时务必统一时钟策略（`--lock` 锁额定频最可复现）。