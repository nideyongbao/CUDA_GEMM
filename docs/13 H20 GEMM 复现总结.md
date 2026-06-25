# H20 GEMM 复现总结（最终）

本文是整个项目在 **NVIDIA H20** 上复现 GEMM 的总收口：从 CUDA core 的 FP32 一路打到 Tensor Core 的 WGMMA / FP8。详细的逐步分析见 docs 下各分篇（00–12）与 [H20复现结论](H20%E5%A4%8D%E7%8E%B0%E7%BB%93%E8%AE%BA.md)；本文只做**重点收敛**。

## 1、硬件与三档算力（一切利用率的分母）

| 引擎 | 峰值 | 谁在用 |
| --- | ---: | --- |
| CUDA core (FP32) | ~40–44 TFLOPS | `kernels/cuda_core/` FP32 那条线 |
| **Tensor Core (BF16)** | **148 TFLOPS** | WMMA / WGMMA(bf16) |
| **Tensor Core (FP8 e4m3)** | **296 TFLOPS** | WGMMA(fp8) |

H20：Hopper(sm_90)、78 SM、96GB HBM3、~4 TB/s、L2 60MB、smem 228KB/SM。它是"带宽厚、算力相对薄"的 Hopper 变种，BF16:FP8 = 1:2，FP32 远低于张量核。

## 2、仓库结构（按引擎拆分）

```
kernels/
  cuda_core/      # 跑在 CUDA core 上
    01_naive … 10_doublebuffer   (FP32 阶梯)
    bf16_cudacore.cu             (同一批 kernel 的 BF16 版：datatype 实验)
  tensor_core/    # 跑在 Tensor Core 上，每个是独立最简用例(自带 main)
    tc_01_wmma_naive  tc_02_wmma_smem  tc_03_wmma_pipe
    tc_04_wgmma_tma_ws  tc_05_wgmma_fp8
src/        bench/verify (FP32) + bench_bf16/verify_bf16 (BF16 cuda core)
profiling/  tensor_core 各用例的 ncu --set full 报告 + SUMMARY.md
docs/       00–07 CUDA core 分析；08–12 Tensor core 分析；13 本总结
```
构建：`make`（FP32 + BF16 sweep）、`make tc`（5 个 tensor core 用例）。

## 3、完整性能阶梯（4096³，best-of）

> **重点：利用率分母要选对**——CUDA core kernel 对 148T 天然只有 ~17%（它根本没用张量核），所以 CUDA core 那段同时给"对 FP32 峰值"和"对 cuBLAS SGEMM"。

### CUDA core（FP32，对 FP32 峰值 44T / 对 fair cuBLAS SGEMM 30264）

| kernel | GFLOPS | vs FP32峰值 | vs cuBLAS SGEMM |
| --- | ---: | ---: | ---: |
| naive | 3532 | 8.0% | 11.7% |
| smem | 5411 | 12.3% | 17.9% |
| 2D blocktiling | 15571 | 35.4% | 51.5% |
| vectorized | 22770 | 51.8% | 75.2% |
| warptile_vec | 23962 | 54.5% | 79.2% |
| **double_buffer** | **25116** | **57.1%** | **83.0%** |
| cuBLAS SGEMM (fair) | 30264 | 68.8% | 100% |

### Tensor Core（对 148T；FP8 另对 296T）

| 级 | 用例 | GFLOPS | util | SM Busy(ncu@2048) |
| --- | --- | ---: | ---: | ---: |
| BF16 cuda core(对照) | double_buffer_bf16 | 25896 | 17.5% | — |
| ① | tc_01 WMMA_naive | 16685 | 11.3% | 17.5% |
| ② | tc_02 WMMA_smem | 29733 | 20.1% | 40.9% |
| ③ | tc_03 WMMA_pipe (cp.async) | 38063 | 25.7% | 39.9% |
| ④ | **tc_04 WGMMA(TMA+WS)** | **119311** | **80.6%** | **82.8%** |
| 参考 | cuBLAS BF16 (fair) | 131820 | 89.1% | — |
| ⑤ | **tc_05 WGMMA_fp8** | **224245** | **75.8% /296T** | 73.1% |

规模放大（WGMMA 越大越高）：BF16 8192³ **88.1%**；FP8 8192³ **84.8% /296T（251 TFLOPS）**。

## 4、ncu 关键洞察（详见 profiling/SUMMARY.md 与 docs 08–12）

把 warp 级 WMMA 和 warpgroup 级 WGMMA 摆在一起，差别一目了然：

| | warp 级 WMMA (tc_01–03) | warpgroup 级 WGMMA (tc_04/05) |
| --- | --- | --- |
| L1/TEX Throughput | 82–**99.8%**（搬 fragment 打满）| **20.7%**（TMA 走专用路径）|
| 最高利用管线 | ALU / MIO（访存与索引）| **Tensor (FP)** |
| Achieved Occupancy | 33–71% | **7.6%** |
| SM Busy | 17–41% | **73–83%** |
| 头号 stall | long_scoreboard / mio_throttle | CTA barrier（生产者/消费者交接）|
| 靠什么藏延迟 | **高占用率(TLP)** | **异步多级流水线** |

## 5、五个重点结论（highlighted）

1. **只换数据类型(BF16)对 CUDA core kernel 基本无加速**：它们是 FP32 计算单元 bound，BF16 只省一半 global 访存（不是瓶颈），全量天花板仍 **~17%**（double_buffer_bf16 17.5% ≈ 其 FP32 版）。想用张量核，换类型远远不够。
2. **用 Tensor Core 必要但不充分**：朴素 WMMA(tc_01) 11.3%，甚至低于最好的 CUDA core kernel——L1/TEX 99.8% 打满、张量核饿死。必须 smem 复用→cp.async 流水线，warp 级才爬到 ~26%。
3. **真正的质变来自 Hopper 原生 WGMMA+TMA+warp specialization**：25.6% → **80.6%（×3.1）**，8192³ 88%。其机理是 **用异步流水线代替高占用率隐藏延迟**：占用率仅 7.6%，SM Busy 却 82.8%，张量核成为最高利用管线。这是 WGMMA 和 WMMA 的本质差别。
4. **FP8 是"可插拔"的两倍吞吐**：同一条 TMA+WGMMA 流水线只改类型/指令/BK 三处，吞吐 119→**224 TFLOPS**（FP8 峰值 75.8%，8192³ 84.8%），CPU double 参考验证正确。Hopper GEMM 的范式 = 先把异步骨架搭对，精度只是骨架上的参数。
5. **cuBLAS 没有被反超——别让 handle 假象骗了**：`cublasCreate/Destroy` 约 **0.33ms/次**，放进计时循环会把 cuBLAS 严重低估（4096³ BF16 89%→68%、1024³ FP32 把 22.9 假报成 5.2 TFLOPS）。修正(handle 复用)后 cuBLAS 在所有尺寸都领先，手写 WGMMA 逼到其 **~90%**。固定开销必须移出计时循环。

## 6、正确性

- CUDA core FP32 / BF16：全部 kernel 1024³/2048³ 对拍 cuBLAS PASS（allclose），并用 CPU double 独立交叉验证。
- Tensor core WMMA/WGMMA(bf16)：与 cuBLAS BF16 bit 级一致（同序累加）。
- FP8：与 CPU double 参考对拍 PASS（fp8 粗糙但算得对）。

## 7、定位与后续

这个项目已经从一份"CUDA core SGEMM 教程"成长为**覆盖 H20 全栈 GEMM 的复现教程**：FP32(CUDA core 57% 峰值) → BF16(WMMA 11→26% → WGMMA 80–88%) → FP8(75–85%)，并且每一步都有最简可跑用例 + ncu 实证 + 文档。

✅ 已完成：CUDA core 全套、WMMA 三级、WGMMA+TMA+WS、FP8、全量 ncu 剖析与文档。
⬜ 仍可继续（追平/超过 cuBLAS 的"最后一公里"）：persistent kernel + tile scheduler、cluster/DSMEM(2-CTA)、自己写 smem swizzle（当前靠 TMA 代劳）、瘦长/小 M 形状的 SwapAB 调度、FP8 per-block scaling，以及用 CuTe/CUTLASS 把这套表达得更干净。

参考：`/shard_data/brooksli/workspace/0625/modern-gpu-programming-for-mlsys`（*Tiled to SOTA* 的 TMA pipelining / warp specialization / 2-CTA cluster 章节正好覆盖后续路线）；WGMMA/FP8 kernel 移植并改写自 LeetCUDA `hgemm_wgmma_*`。
