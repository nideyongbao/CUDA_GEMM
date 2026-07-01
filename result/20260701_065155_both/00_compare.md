# 两轮对比：默认 vs 锁频

- A（默认）: `result/20260701_065155_both/default`  —  default(开箱自适应boost)
- B（锁频）: `result/20260701_065155_both/locked`  —  locked@1410MHz(额定boost)
- 遥测 SM 时钟（计算期典型/峰）: A **1410/1410 MHz** · B **1410/1410 MHz**

## FP32 @ headline（默认 → 锁频，Δ%）
| kernel | 默认 GFLOPS | 锁频 GFLOPS | Δ% |
| --- | ---: | ---: | ---: |
| 00_cublas_ref | 16045 | 19024 | +18.6% ⚠️ |
| 01_naive | 3045 | 3051 | +0.2% |
| 02_smem | 5349 | 5354 | +0.1% |
| 03_blocktiling | 10309 | 10311 | +0.0% |
| 04_2Dblocktiling | 14428 | 14425 | -0.0% |
| 05_vectorized | 17614 | 17615 | +0.0% |
| 06_autotune_64x64x8_8x4 | 17605 | 17613 | +0.0% |
| 07_autotune_64x64x16_8x4 | 16816 | 16825 | +0.1% |
| 08_autotune_64x64x8_8x8 | 16780 | 16780 | +0.0% |
| 09_warptile | 12664 | 12667 | +0.0% |
| 10_warptile_vec | 17341 | 17350 | +0.1% |
| 11_bank_conflict | 17090 | 17101 | +0.1% |
| 12_double_buffer | 17056 | 17063 | +0.0% |

## BF16 @ headline（默认 → 锁频，Δ%）
| kernel | 默认 GFLOPS | 锁频 GFLOPS | Δ% |
| --- | ---: | ---: | ---: |
| 00_cublas_bf16 | 264468 | 264677 | +0.1% |
| 01_naive | 3103 | 3106 | +0.1% |
| 02_smem | 5332 | 5334 | +0.0% |
| 03_blocktiling | 10594 | 10599 | +0.0% |
| 04_2Dblocktiling | 14335 | 14340 | +0.0% |
| 05_vectorized | 17786 | 17787 | +0.0% |
| 06_autotune_64x64x8_8x4 | 17773 | 17779 | +0.0% |
| 07_autotune_64x64x16_8x4 | 17009 | 17017 | +0.0% |
| 08_autotune_64x64x8_8x8 | 16161 | 16163 | +0.0% |
| 09_warptile | 12142 | 12158 | +0.1% |
| 10_warptile_vec | 17473 | 17478 | +0.0% |
| 11_bank_conflict | 17082 | 17087 | +0.0% |
| 12_double_buffer | 17250 | 17257 | +0.0% |

## Tensor Core @ headline（默认 → 锁频，Δ%）
| kernel | 默认 GFLOPS | 锁频 GFLOPS | Δ% |
| --- | ---: | ---: | ---: |
| 01_tc01_wmma_naive | 17779 | 17798 | +0.1% |
| 02_tc02_wmma_smem | 27111 | 27027 | -0.3% |
| 03_tc03_wmma_pipe | 43030 | 43053 | +0.1% |

## 正确性（两轮均应全 PASS）
- FP32: A 13P/0F · B 13P/0F
- BF16: A 13P/0F · B 13P/0F
- Tensor Core: A 3P/0F · B 3P/0F

## 结论
- **默认轮有 1 个 kernel 因 boost 抖动被欠采（|Δ|>5%）**，锁频后回正：
  - FP32 `00_cublas_ref`: 16045 → 19024（+18.6%）
- 其余 kernel 两轮基本一致（低功耗或恰好撞满频）。**要可复现/公平的逐 kernel 数字用锁频；默认仅适合看整体趋势/真实开箱值。**
- 最大单核差异: +18.6%。
