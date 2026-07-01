# 两轮对比：默认 vs 锁频

- A（默认）: `result/20260701_093537_both/default`  —  default(开箱自适应boost)
- B（锁频）: `result/20260701_093537_both/locked`  —  locked@1410MHz(额定boost)
- 遥测 SM 时钟（计算期典型/峰）: A **1290/1410 MHz** · B **1410/1410 MHz**

## FP32 @ headline（默认 → 锁频，Δ%）
| kernel | 默认 GFLOPS | 锁频 GFLOPS | Δ% |
| --- | ---: | ---: | ---: |
| 00_cublas_ref | 15390 | 19034 | +23.7% ⚠️ |
| 01_naive | 3054 | 3055 | +0.0% |
| 02_smem | 5348 | 5346 | -0.0% |
| 03_blocktiling | 9565 | 10304 | +7.7% ⚠️ |
| 04_2Dblocktiling | 14443 | 14446 | +0.0% |
| 05_vectorized | 17624 | 17629 | +0.0% |
| 06_autotune_64x64x8_8x4 | 17626 | 17631 | +0.0% |
| 07_autotune_64x64x16_8x4 | 16833 | 16840 | +0.0% |
| 08_autotune_64x64x8_8x8 | 16790 | 16794 | +0.0% |
| 09_warptile | 12678 | 12676 | -0.0% |
| 10_warptile_vec | 17362 | 17363 | +0.0% |
| 11_bank_conflict | 17105 | 17109 | +0.0% |
| 12_double_buffer | 17073 | 17073 | -0.0% |

## BF16 @ headline（默认 → 锁频，Δ%）
| kernel | 默认 GFLOPS | 锁频 GFLOPS | Δ% |
| --- | ---: | ---: | ---: |
| 00_cublas_bf16 | 264573 | 264677 | +0.0% |
| 01_naive | 3107 | 3104 | -0.1% |
| 02_smem | 5330 | 5330 | +0.0% |
| 03_blocktiling | 10606 | 10605 | -0.0% |
| 04_2Dblocktiling | 14345 | 14344 | -0.0% |
| 05_vectorized | 17798 | 17795 | -0.0% |
| 06_autotune_64x64x8_8x4 | 17799 | 17796 | -0.0% |
| 07_autotune_64x64x16_8x4 | 17036 | 17042 | +0.0% |
| 08_autotune_64x64x8_8x8 | 16170 | 16171 | +0.0% |
| 09_warptile | 12161 | 12145 | -0.1% |
| 10_warptile_vec | 17481 | 17486 | +0.0% |
| 11_bank_conflict | 17099 | 17096 | -0.0% |
| 12_double_buffer | 17260 | 17264 | +0.0% |

## Tensor Core @ headline（默认 → 锁频，Δ%）
| kernel | 默认 GFLOPS | 锁频 GFLOPS | Δ% |
| --- | ---: | ---: | ---: |
| 01_tc01_wmma_naive | 17824 | 17829 | +0.0% |
| 02_tc02_wmma_smem | 27104 | 27139 | +0.1% |
| 03_tc03_wmma_pipe | 43084 | 43080 | -0.0% |
| 06_tc06_mma_pipe | 120808 | 150333 | +24.4% ⚠️ |

## 正确性（两轮均应全 PASS）
- FP32: A 13P/0F · B 13P/0F
- BF16: A 13P/0F · B 13P/0F
- Tensor Core: A 4P/0F · B 4P/0F

## 结论
- **默认轮有 3 个 kernel 因 boost 抖动被欠采（|Δ|>5%）**，锁频后回正：
  - FP32 `00_cublas_ref`: 15390 → 19034（+23.7%）
  - FP32 `03_blocktiling`: 9565 → 10304（+7.7%）
  - Tensor Core `06_tc06_mma_pipe`: 120808 → 150333（+24.4%）
- 其余 kernel 两轮基本一致（低功耗或恰好撞满频）。**要可复现/公平的逐 kernel 数字用锁频；默认仅适合看整体趋势/真实开箱值。**
- 最大单核差异: +24.4%。
