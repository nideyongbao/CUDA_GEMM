# A800 ncu 剖析汇总（与 H20 同口径）

- 硬件：**NVIDIA A800-SXM4-80GB**（Ampere GA100，CC 8.0，sm_80，108 SM）。
- 工具：`ncu --set full`（采计数器需 root → `sudo /usr/local/cuda-12.8/bin/ncu`），脚本 `profiling/a800/run_ncu_a800.sh`。
- 剖析尺寸：**2048³**（`-s 1 -c 1`，跳过第 1 次 launch、剖析第 2 次 warmup），与 H20 的 `profiling/{cuda_core,tensor_core}/SUMMARY.md` 完全同口径，便于逐核对比。
- 完整报告：`profiling/a800/a800_*.ncu-rep`（Nsight UI 打开）+ `a800_*.details.txt`（文本全量）。
- **注意**：ncu 默认把时钟锁到 base clock 采计数器，故下表 Duration 偏长、由它反推的 GFLOPS 是 base-clock 值，**不可与 headline（4096³、锁 boost 1410MHz）的吞吐直接比**。下表只用于看与时钟无关的 **% 类指标 + 占用率 + stall**（这些才是跨卡可比的画像）。headline 吞吐见 `throughput_4096.txt`。

## Speed-of-Light + 占用率（@2048³）

| 用例 | Compute(SM)% | SM Busy% | Issue Slots% | L1/TEX% | L2% | DRAM% | 占用率 | Reg/thr | warp cyc/issue |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **FP32 cuda core** |
| cublas (sm80 gemm) | 91.3 | **96.6** | 56.1 | 38.5 | 13.1 | 2.9 | 18.5% | 122 | **5.26** |
| 01 naive | 75.2 | 44.0 | 44.0 | 75.9 | 8.6 | 0.6 | 98.2% | 32 | 35.68 |
| 02 smem | 77.0 | 40.6 | 38.1 | **92.4** | 10.0 | 1.0 | 98.6% | 32 | 41.45 |
| 04 2D blocktiling | 71.0 | 75.6 | 52.2 | 70.6 | 16.0 | 2.5 | 26.3% | 96 | 8.07 |
| 05 vectorized | 82.1 | 86.7 | 50.2 | 66.1 | 18.9 | 3.3 | 34.6% | 64 | 11.04 |
| 10 doublebuffer | 79.9 | 83.9 | 48.0 | 43.0 | 11.9 | 3.2 | 18.5% | 128 | 6.15 |
| **BF16 cuda core** |
| 10 doublebuffer_bf16 | 82.0 | 86.4 | 50.4 | 43.0 | 6.6 | 1.5 | 18.4% | 120 | 5.83 |
| **Tensor Core (WMMA)** |
| tc_01 WMMA_naive | 16.5 | 10.3 | 10.3 | **99.7** | 16.7 | 1.0 | 68.8% | 40 | **106.8** |
| tc_02 WMMA_smem | 32.6 | 33.9 | 33.9 | 73.3 | 15.8 | 1.6 | 30.9% | 80 | 14.57 |
| tc_03 WMMA_pipe | 24.8 | 18.4 | 18.4 | **93.8** | 13.6 | 2.3 | 31.5% | 66 | 27.36 |

## 头号 warp stall

| 用例 | warp cyc/issue | 头号 stall | 解读 |
| --- | ---: | --- | --- |
| cublas FP32 | 5.26 | — | SM Busy 96.6%、Compute 91.3%，已逼近极限（占用率仅 18.5% 却吃满） |
| 01 naive | 35.68 | L1 指令队列 ~19 cyc | 一线程一元素，访存指令排队，靠 98% 占用率硬扛 |
| 02 smem | 41.45 | MIO throttle ~23 cyc | smem 复用但 load/索引的 MIO/ALU 仍是天花板，L1/TEX 92% |
| 10 doublebuffer | 6.15 | — | ILP 拉满，cyc/issue 砍到 6，SM Busy 84% |
| tc_01 WMMA_naive | 106.8 | L1 指令队列 ~33 cyc | fragment 直接从 global 取，L1/TEX **99.7% 打满**、张量核饿死 |
| tc_02 WMMA_smem | 14.57 | (smem)scoreboard | smem 复用把 cyc/issue 从 107 砍到 15 |
| tc_03 WMMA_pipe | 27.36 | MIO ~10 cyc | cp.async 让搬/算重叠，L1/TEX 仍 94%（搬 fragment） |

## 一句话结论（A800）

- **CUDA core 那条线（FP32/BF16）的 ncu 画像与 H20 几乎同形**：占用率从 naive 98% 一路降到 doublebuffer 18%，SM Busy 反而 44%→84%、cyc/issue 36→6——**ILP 取代 TLP**。cublas（`sm80` gemm）是极致：占用率 18.5% 却 Compute 91% / SM Busy 97%。
- **WMMA 三级（tc_01→03）同样被 L1/TEX 卡死**：tc_01 的 L1/TEX **99.7%**、cyc/issue **106.8**，张量核（Compute 16.5%）完全饿死；smem/cp.async 把 cyc/issue 砍下来，但 L1/TEX 仍 73–94%、Compute(SM) 始终 ≤ 33%。
- **和 H20 最大的不同：A800 没有第 ④ 级（WGMMA+TMA）可走**。H20 上正是这一级把 Compute(SM) 拉到 67%、SM Busy 83%、用异步流水线代替 TLP；**A800（Ampere）硬件无 WGMMA/TMA，WMMA 这条手写路线就停在 Compute(SM) ≤ 33% / 占用率 31% 这一层**。要吃满 A800 的 312T BF16 张量核，得靠 cuBLAS（内部用 Ampere `mma.sync` + `ldmatrix` + 多级 cp.async）或 CUTLASS。
