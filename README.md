# CUDA_MATMUAL — H20 GEMM 优化阶梯

一个学习型矩阵乘法（GEMM）项目：从最朴素的 CUDA SGEMM 出发，一步步优化到 **shared memory → register tiling → float4 → warp tiling → bank conflict → double buffering**（CUDA core），再跨到 **WMMA → cp.async 流水线 → WGMMA + TMA + warp specialization → FP8**（Tensor Core），最终逼近 cuBLAS。

重点不是"给一个最快的 kernel"，而是把**每一步优化为什么有效、什么时候无效、怎么用 Nsight Compute 判断真正的瓶颈**记录下来。代码负责复现实验，`docs/` 负责沉淀分析。所有 ncu 数据均为 **NVIDIA H20（Hopper，CC 9.0，78 SM）实测**。

---

## 结果速览（@4096³）

**CUDA core / FP32**（对 cuBLAS SGEMM 基线 30266 GFLOPS）：

| 阶段 | GFLOPS | 占 cuBLAS |
| --- | ---: | ---: |
| naive（一线程一元素） | 3532 | 11.7% |
| + shared memory | 5411 | 17.9% |
| + 2D register tiling | 15548 | 51.4% |
| + float4 向量化 | 22790 | 75.3% |
| **warp tiling + 向量化（手写最佳）** | **23971** | **79.2%** |
| cuBLAS SGEMM（基线） | 30266 | 100% |

**Tensor Core / BF16·FP8**（手写完整阶梯，对 148 TFLOPS BF16 峰值；FP8 对 296T）：

| id | 用例 | 技术 | 利用率 |
| ---: | --- | --- | ---: |
| 1 | tc_01 WMMA_naive | warp 级，global 直取 | 11.3% |
| 2 | tc_02 WMMA_smem | + shared memory 复用 | 20.1% |
| 3 | tc_03 WMMA_pipe | + cp.async 多级流水线 | 25.7% |
| **4** | **tc_04 WGMMA** | warpgroup 异步 + TMA + warp specialization | **80.6%**（8192³ 88%）|
| 5 | tc_05 WGMMA_fp8 | 同上换 FP8 | 75.8% /296T（224 TFLOPS）|
| — | 参考 cuBLAS BF16（fair） | — | 89.1% |

> 一句话结论：**warp 级 WMMA 靠高占用率（TLP）藏延迟，张量核吃不饱（≤26%）；warpgroup 级 WGMMA 靠异步多级流水线，占用率仅 7.6% 却把 SM Busy 拉到 83%**——这才是 Hopper 上逼近 148T 的路。详见 [docs/13](docs/h20/13%20H20%20GEMM%20复现总结.md)。

---

## 快速开始

**一键全量测试（任意机器，推荐）**——自动识别 GPU 架构、编译、跑正确性+性能+遥测、出汇总报告，结果按时间戳存入 `result/`：

```bash
bash run_all.sh              # 开箱默认时钟，全量测试（新机器从零跑通就这一条）
bash run_all.sh --lock       # 额外锁额定 boost（需 sudo，测可复现的满频上限）
bash run_all.sh --quick      # 快速版（跳过尺寸缩放/autotune）
bash run_all.sh --gpu 1      # 指定 GPU
```

产物：`result/<时间戳>/`（各步 `.log` + 遥测 + `00_summary.md`），并软链 `result/latest`。每台机器跑一遍即得该机型完整 GEMM 结果；跨机型对比见 [docs/README](docs/README.md)。

**手动分步**（原始入口，`make` 默认 H20 sm_90a；A800 见下方"在 A800 上构建"）：

```bash
make            # 编译 CUDA core：bench / verify / bench_bf16 / verify_bf16
make tc         # 编译 Tensor Core：统一 bench / verify 驱动

./kernels/cuda_core/bench  10 4096 4096 4096   # 跑 FP32 warptile_vec @4096³
./kernels/tensor_core/bench 4 4096 4096 4096   # 跑 WGMMA(TMA+WS) @4096³
```

矩阵计算接口统一为 `C = alpha * A * B + beta * C`，A、B、C 均 **row-major**；cuBLAS reference 通过交换 A/B 与维度参数适配 row-major 结果。

---

## 仓库结构

按"**计算引擎**"对称拆分——`cuda_core` 与 `tensor_core` 各自有一套 `bench <id>` / `verify <id>` 驱动，命令、目录、profiling 一一对应：

```
kernels/
  cuda_core/      FP32 阶梯(01_naive…10_doublebuffer) + BF16 对照(bf16_cudacore.cu)
                  驱动源码 benchmark.cu / verify.cu / bench_bf16.cu / verify_bf16.cu
                  ↳ make 后可执行文件 bench / verify / bench_bf16 / verify_bf16 也输出在此
  tensor_core/    5 个用例(tc_01_wmma_naive…tc_05_wgmma_fp8)，各自保留专属 harness、
                  编成 kernel-only 对象，由派发表 tc_cases.h + bench.cu / verify.cu
                  按 id(1–5) 统一驱动
include/          公共宏、FP32/BF16 声明、autotuning 模板、tensor core 脚手架 tc_common.cuh
profiling/        ncu 剖析，按引擎分目录 cuda_core/ + tensor_core/
docs/             GPU 硬件知识、性能方法论、逐 kernel 的 ncu 分析（00–07 CUDA core，
                  08–12 Tensor core，13 总结）
```

| | CUDA core / FP32 | Tensor Core |
| --- | --- | --- |
| 跑性能 | `./kernels/cuda_core/bench <id> M N K` | `./kernels/tensor_core/bench <id> [M N K]` |
| 对拍正确性 | `./kernels/cuda_core/verify <id> M N K` | `./kernels/tensor_core/verify <id> [M N K]` |
| id 范围 | 0–12（见下） | 1–5（见下） |
| profiling | `profiling/cuda_core/` | `profiling/tensor_core/` |

---

## 环境与编译

需要 NVIDIA GPU、CUDA Toolkit（含 `nvcc`）、cuBLAS、GNU Make。目标硬件 **H20（Hopper，CC 9.0）**，Makefile 默认架构：

```makefile
ARCH := -arch=sm_90a
```

`sm_90a` 是 Hopper 架构专用目标（architecture-specific），WGMMA / TMA 等 Hopper 张量指令必须用它。

> 旧版本默认 `sm_75`（Turing）。在 H20 上用 `sm_75` 编出的二进制**也能跑**——fatbin 里带了 `compute_75` 的 PTX，驱动加载时会 JIT 成 sm_90 SASS——但有首次 JIT 开销、且不是 Hopper 原生调优代码，**性能数字不可信**。务必用 `sm_90a`。换卡按设备改 `ARCH`：Ada `sm_89`、A100 `sm_80`、Turing `sm_75`；改完执行 `make clean && make`。

```bash
make        # CUDA core：FP32 bench/verify + BF16 bench_bf16/verify_bf16
make tc     # Tensor Core：统一 bench/verify 驱动（链接 5 个用例对象）
make clean  # 清理 .o 与可执行文件
```

### 在 A800 / A100（Ampere, sm_80）上构建

`ARCH` 与 `TC_HOPPER` 可命令行覆盖。Ampere 无 WGMMA/TMA/FP8，需用 `TC_HOPPER=0` 把 Hopper 独占用例（tc_04/tc_05）从派发表剔除，只编 WMMA 三级：

```bash
make ARCH=-arch=sm_80                  # CUDA core FP32+BF16（源码零改动）
make tc ARCH=-arch=sm_80 TC_HOPPER=0   # Tensor Core 只编 tc_01-03（WMMA）
```

A800 全量复现结果、与 H20/cuBLAS/理论峰值/公开基准的对账见 **[docs/A800 GEMM 复现总结](docs/a800/A800%20GEMM%20复现总结.md)**（profiling 数据在 `profiling/a800/`）。一句话：A800 BF16 张量核峰值 312T（H20 的 2.1×），cuBLAS BF16 实测 214–294T；但手写阶梯在 Ampere 上止步 WMMA（tc_03 43T，13.8% 峰），因 WGMMA/TMA/FP8 是 Hopper 独占。

`make tc` 把 5 个用例编成 kernel-only 对象，链接成两个统一驱动（`tc_04/tc_05` 用 TMA，额外链 `-lcuda`）。产物按引擎分目录，不再污染仓库根目录。

---

## 运行：CUDA core（FP32）

```bash
./kernels/cuda_core/verify              # 默认 id=0(cuBLAS)，1024³
./kernels/cuda_core/verify <id> M N K   # 指定 kernel 与尺寸
./kernels/cuda_core/bench  <id> M N K   # 性能 + GFLOPS
./kernels/cuda_core/bench  autotune M N K   # autotuning 配置扫描
```

`verify` 跑目标 kernel 与 cuBLAS reference 对拍，输出 `max_abs_err / max_rel_err / bad_count / PASS|FAIL`；误差判据为 numpy `allclose` 风格 `|a-b| <= atol + rtol·|b|`（atol=rtol=1e-2，消除随机数据下近零元素的假 FAIL）。传入非法 id 会打印当前注册表。

**Kernel id 注册表**（`bench` / `verify` 共用）：

| id | Kernel | 说明 |
| ---: | --- | --- |
| 0 | cublas_ref | row-major 适配的 cuBLAS SGEMM 基线 |
| 1 | naive | 一线程算 C 的一个元素 |
| 2 | smem | A/B 分块搬入 shared memory |
| 3 | blocktiling | 1D register tiling，一线程算多个结果 |
| 4 | 2D blocktiling | TM/TN 二维 register tiling |
| 5 | vectorized | float4 向量化访存 |
| 6 | autotune 64×64×8_8×4 | `launch_at<64,64,8,8,4>` |
| 7 | autotune 64×64×16_8×4 | `launch_at<64,64,16,8,4>` |
| 8 | autotune 64×64×8_8×8 | `launch_at<64,64,8,8,8>` |
| 9 | warptile | warp-level tiling 基线 |
| 10 | warptile_vec | warp tiling + float4 |
| 11 | bank conflict | 搬运映射与 bank conflict 分析版 |
| 12 | double buffer | double buffering 版 |

### BF16（CUDA core 对照）

`make` 还生成 `bench_bf16` / `verify_bf16`：把上面那批 kernel 改成 **BF16 输入 / FP32 累加**，但**仍跑在 CUDA core 上**——用来证明"光换数据类型、不碰张量核没用"。

```bash
./kernels/cuda_core/bench_bf16  <id> M N K   # GFLOPS 及对 148 TFLOPS BF16 峰值的利用率
./kernels/cuda_core/verify_bf16 <id> M N K   # 与 cuBLAS BF16 对拍
```

id 0=cuBLAS BF16，1–12 与 FP32 同名。结论：全量天花板只有 **~17%**（compute bound 仍卡在 FP32 单元，BF16 只省了一半访存）。

---

## 运行：Tensor Core

```bash
make tc
./kernels/tensor_core/verify <id> [M N K]   # 对拍参考，打印 PASS/FAIL
./kernels/tensor_core/bench  <id> [M N K]   # 纯计时 + 利用率
```

- 不带 id 默认 **id=4（tc_04 WGMMA，手写阶梯标杆）@ 4096³**；非法 id 会打印用例列表。
- `tc_05` FP8 没有简单 cuBLAS 路径，`verify` 用 CPU double 参考（O(n³)），自动在 **≤512³** 上对拍。

```bash
./kernels/tensor_core/bench  4 4096 4096 4096   # tc_04 WGMMA @4096³
./kernels/tensor_core/verify 5 512 512 512      # tc_05 FP8 对拍 CPU double
```

用例阶梯与利用率见文首「结果速览」表。逐例 ncu 分析见 docs 08–12。

> ⚠️ cuBLAS 基线务必让 handle 复用——`cublasCreate/Destroy` 约 0.33ms/次，放进计时循环会把 cuBLAS 严重低估（4096³ 89%→68%），制造"手写反超 cuBLAS"的假象。本项目所有驱动已修复（handle 静态复用、create/destroy 不计时）。详见 docs/13 §5。

---

## Nsight Compute

剖析报告按引擎分目录，**文本产物入库**、二进制本地生成：

- `profiling/cuda_core/`、`profiling/tensor_core/`：各含 `SUMMARY.md`（汇总表）+ 逐核 `*.details.txt`（`ncu --set full` 文本全量）+ `run_ncu.sh`（一键复跑）。
- `*.ncu-rep`（Nsight UI 可打开）与 `*.run.log` 仅本地生成，已 `.gitignore`。
- `profiling/throughput_4096.txt`：FP32 阶梯 @4096³ 的 headline 吞吐表。
- 本机采计数器需 root：`sudo /usr/local/cuda/bin/ncu`（见各 SUMMARY 头注）。

```bash
sudo /usr/local/cuda/bin/ncu --set full ./kernels/cuda_core/bench 10 4096 4096 4096
```

文档主要关注：kernel duration / elapsed cycles、Compute(SM) 与 Memory Throughput、DRAM/L1·TEX/L2 Throughput、achieved occupancy、registers per thread、global coalescing、shared bank conflict、warp stall reason。

> 核心经验：ncu 上某个 throughput 很高 **不等于**它就是 binding bottleneck。真正要看的是优化它之后 **elapsed cycles 是否下降**。

---

## 文档导航

建议顺序阅读：

1. [GPU 前置硬件知识](docs/h20/00%20GPU前置硬件知识.md)
2. [性能分析方法论](docs/h20/01%20性能分析方法论.md)
3. [naive kernel 性能分析](docs/h20/02%20naive%20kernel性能分析.md)
4. [smem kernel 性能分析](docs/h20/03%20smem%20kernel性能分析.md)
5. [register tiling](docs/h20/04%20register%20tiling.md)
6. [vectorizer](docs/h20/05%20vectorizer.md)
7. [warp tile](docs/h20/06%20warp%20tile.md)
8. [bank conflict](docs/h20/07%20blank%20conflict.md)

Tensor Core 系列（每个用例一篇，含 ncu 分析）：

9. [WMMA naive](docs/h20/08%20tensor%20core%20-%20WMMA%20naive.md)
10. [WMMA smem](docs/h20/09%20tensor%20core%20-%20WMMA%20smem.md)
11. [WMMA cp.async pipeline](docs/h20/10%20tensor%20core%20-%20WMMA%20cp.async%20pipeline.md)
12. [WGMMA + TMA + warp specialization](docs/h20/11%20tensor%20core%20-%20WGMMA%20TMA%20warp%20specialization.md)
13. [FP8 WGMMA](docs/h20/12%20tensor%20core%20-%20FP8%20WGMMA.md)

**最终总结**（重点收敛）：[docs/13 H20 GEMM 复现总结.md](docs/h20/13%20H20%20GEMM%20复现总结.md)；**A800 迁移复现**：[docs/A800 GEMM 复现总结.md](docs/a800/A800%20GEMM%20复现总结.md)。早期逐步复现详录见 [docs/H20复现结论.md](docs/h20/H20复现结论.md)；`docs/6.1.md`–`6.5.md` 是阶段性交接文档（实验结论、性能对账、踩坑、下一步），想快速了解项目演进可从最新的 [6.5](docs/h20/6.5.md) 看起。

---

## 当前状态与后续

已覆盖 SGEMM 优化的多条主线：global naive → shared memory → register tiling → float4 coalescing → warp tiling → bank conflict 实证 → double buffering，并跨入 Tensor Core 的 WMMA → cp.async → WGMMA/TMA/warp specialization → FP8 完整阶梯。

后续方向：固化更多尺寸下的 benchmark 表、针对不同 GPU 架构维护独立配置、继续完善 double buffering 与更系统的 autotuning 搜索。
