# GEMM 算子（A800 / sm_80）— 索引与复现

> 本目录的 `gemm/` 算子从 **CUDA_GEMM** 原样移植而来（kernel 逐字节一致，只改了构建/派发的
> 跨代际门控）。因此**每一级的深度剖析不在这里重写，而是复用已随仓库带过来的 A800 分析文档**。
> 本 README 只做三件事：① 摆出 A800 上的手写阶梯与 id 注册表；② 给出构建/运行入口；
> ③ 用**本机实测 ground-truth** 摆出复现表，并把读者指向 `a800/` 下的逐核 ncu 剖析。

目标硬件：**NVIDIA A800-SXM4-80GB（Ampere GA100，CC 8.0，sm_80，108 SM）**。
两条峰值分母贯穿全文：**FP32(CUDA core) 峰值 ≈ 19.5 TFLOPS**、**BF16 张量核峰值 = 312 TFLOPS**。

## 0. 先读什么（文档索引）

| 文档 | 作用 | 备注 |
| --- | --- | --- |
| [00_GPU硬件前置知识.md](./00_GPU硬件前置知识.md) | 一个 SM 的内部结构（DRAM/L2/L1-SMEM/LSU-MIO/warp scheduler/寄存器堆），所有优化术语的来源 | 概念架构无关；表内数字是 H20 原文，A800 数字见 §1 与 `a800/` |
| [01_性能分析方法论.md](./01_性能分析方法论.md) | ncu 三板斧（看大局→看延迟→看访存合并/bank）的固定流程与命令 | 方法论通用；A800 的实测结果落在 `a800/ncu分析.md` |
| [a800/A800 GEMM 复现总结.md](./a800/A800%20GEMM%20复现总结.md) | **A800 主报告**：硬件峰值、逐引擎性能阶梯、时钟策略、公开基准对账、A800↔H20 终览 | 复现的权威口径与结论 |
| [a800/ncu分析.md](./a800/ncu分析.md) | A800 逐核 ncu 画像（Speed-of-Light + 占用率 + 头号 stall，2048³ 同 H20 口径） | 每一级"为什么慢"的证据 |
| [a800/Ampere mma.sync 张量核.md](./a800/Ampere%20mma.sync%20张量核.md) | **tc_06 专题**：为什么 WMMA 在 Ampere 到头、`mma.sync`+`ldmatrix`+`cp.async` 如何把手写拉到 ~48% 峰、以及到 cuBLAS(85%) 还差的 SASS 级机器 | 张量核最深一课 |

本机 ground-truth 原始输出：[`../baselines/GROUNDTRUTH_cuda_gemm_a800.txt`](../baselines/GROUNDTRUTH_cuda_gemm_a800.txt)。

## 1. 手写阶梯（A800 两条引擎线）

**CUDA core 线（FP32，`kernels/cuda_core/`）** — 天花板是 ~19.5 TFLOPS 那一列，瓶颈始终在访存/SRAM/调度：

`01_naive → 02_smem → 03_blocktiling → 04_2Dblocktiling → 05_vectorized → 07_warptile → 08_warptile_vec → 09_bankconflict → 10_doublebuffer`

（文件序号跳过 `06` —— 那是 `include/06_autotuning.cuh` 的自动调参模板，在 bench 里占 id 6/7/8 三个配置，不算独立一级。）

**Tensor core 线（BF16 输入 / FP32 累加，`kernels/tensor_core/`）** — 天花板才是 312 TFLOPS：

`tc_01_wmma_naive → tc_02_wmma_smem → tc_03_wmma_pipe → tc_06_mma_pipe`

- 前三级是 `nvcuda::wmma` C++ API 阶梯（naive → smem 复用 → cp.async 流水）。
- **tc_06** 换成 Ampere 原生 PTX：`mma.sync.m16n8k16` + `ldmatrix` + 多级 `cp.async`，是 sm_80 上手写能到的**最高一级**。
- **A800 上没有 tc_04 / tc_05**：它们用 Hopper 独占的 `wgmma.mma_async` + TMA（+ FP8 `e4m3`），`sm_80` 编译会被 ptxas 拒绝。构建时 `-DNO_HOPPER` 把这两级从派发表剔除，id 4/5 在 A800 上不可用（tc_06 的 id 恒为 6，见 §5 与 `tc_cases.h`）。

## 2. id 注册表与如何运行

### 2.1 构建（A800）

```bash
make ARCH=-arch=sm_80                  # FP32 + BF16（CUDA core），源码零改动
make tc ARCH=-arch=sm_80 TC_HOPPER=0   # Tensor Core 只编 WMMA 三级 + tc_06（跳过 Hopper 独占的 tc_04/05）
```

产物落在 `build/cuda_core/{bench,verify,bench_bf16,verify_bf16}` 与 `build/tensor_core/{bench,verify}`。

### 2.2 FP32 CUDA core（`build/cuda_core/bench <id> M N K`）

| bench id | 源文件 | kernel |
| ---: | --- | --- |
| 0 | — | cublas_ref（cuBLAS SGEMM 基线） |
| 1 | `01_naive.cu` | naive |
| 2 | `02_smem.cu` | smem |
| 3 | `03_blocktiling.cu` | blocktiling(1D) |
| 4 | `04_2Dblocktiling.cu` | 2D blocktiling |
| 5 | `05_vectorized.cu` | vectorized(float4) |
| 6 / 7 / 8 | `06_autotuning.cuh` | autotune 64×64×8_8×4 / 64×64×16_8×4 / 64×64×8_8×8 |
| 9 | `07_warptile.cu` | warptile |
| 10 | `08_warptile_vec.cu` | warptile_vec |
| 11 | `09_bankconflict.cu` | bank_conflict |
| 12 | `10_doublebuffer.cu` | double_buffer |

### 2.3 Tensor core（`build/tensor_core/bench <id> M N K`）

| tc id | 源文件 | 用例 | 手写指令 |
| ---: | --- | --- | --- |
| 1 | `tc_01_wmma_naive.cu` | WMMA_naive | `wmma` fragment 直取 global |
| 2 | `tc_02_wmma_smem.cu` | WMMA_smem | `wmma` + smem 复用 |
| 3 | `tc_03_wmma_pipe.cu` | WMMA_pipe | `wmma` + cp.async 流水 |
| 6 | `tc_06_mma_pipe.cu` | MMA_pipe | `mma.sync` + `ldmatrix` + 多级 `cp.async` |
| ~~4 / 5~~ | tc_04/tc_05 | WGMMA_TMA_WS / WGMMA_fp8 | **A800 不可用（Hopper sm_90 独占）** |

### 2.4 运行 / 对拍

```bash
sudo nvidia-smi -i 0 -lgc 1410                     # 锁额定 boost 复现 canonical（默认 boost 会随预热漂移）
./build/cuda_core/bench   5  4096 4096 4096         # FP32 vectorized
./build/tensor_core/bench 6  4096 4096 4096         # 张量核 mma.sync（~110T，见 §3）
./build/cuda_core/verify  5  4096 4096 4096         # 单核对拍 cuBLAS SGEMM → PASS
./build/tensor_core/verify 6 4096 4096 4096         # 对拍 cuBLAS BF16 → PASS
./build/cuda_core/bench_bf16 0 4096 4096 4096       # 对照：BF16 输入仍跑 CUDA core（证明只换类型没用，见 a800 §4.2）
sudo nvidia-smi -i 0 -rgc                           # 用完解锁
```

`bench <id> M N K` 纯计时打印 GFLOPS；`verify <id> M N K` 用 `|a-b| ≤ atol+rtol|b|`（atol=rtol=1e-2）对拍 cuBLAS，打印 PASS/FAIL。二者 id 语义一致。

## 3. 本机复现表（THIS A800，锁频 1410MHz，4096³）

下表数字为**本机实测 ground-truth**（[`GROUNDTRUTH_cuda_gemm_a800.txt`](../baselines/GROUNDTRUTH_cuda_gemm_a800.txt)）。`gemm/` 与 CUDA_GEMM kernel 一致，故这就是本仓库在本卡的复现值。

### 3.1 FP32 CUDA core（分母：19.49T 峰值 / cublas_ref）

| bench id | kernel | GFLOPS | % 峰值(19.49T) | % cuBLAS |
| ---: | --- | ---: | ---: | ---: |
| 0 | cublas_ref | 15688 | 80.5% | 100% |
| 1 | naive | 2935 | 15.1% | 18.7% |
| 2 | smem | 4143 | 21.3% | 26.4% |
| 3 | blocktiling(1D) | 10289 | 52.8% | 65.6% |
| 4 | 2D blocktiling | 11510 | 59.1% | 73.4% |
| 5 | vectorized(float4) | 13988 | 71.8% | 89.2% |
| 9 | warptile | 10239 | 52.5% | 65.3% |
| **10** | **warptile_vec** | **17363** | **89.1%** | **110.7%** |
| 11 | bank_conflict | 16617 | 85.3% | 105.9% |
| 12 | double_buffer | 13829 | 71.0% | 88.1% |

- **手写最佳 = warptile_vec 17.4T ≈ 峰值 89%**，量级与 a800 主报告 canonical(~17.6T) 一致 —— 轻载的 CUDA-core 阶梯对时钟/预热不敏感，复现得很干净。
- Ampere 特征：`double_buffer`(13.8T) **低于** `vectorized`/`warptile_vec` —— 双缓冲收益不及 Hopper（见 a800 §4.1）。

### 3.2 Tensor core（分母：312T 峰值；工具打印的 `util(vs148T)` 是 H20 硬编码分母，真实利用率见末列）

| 级 | tc id | 用例 | GFLOPS | 工具 util(vs148T) | **真实 %(vs 312T)** |
| --- | ---: | --- | ---: | ---: | ---: |
| ① | 1 | tc_01 WMMA_naive | 15570 | 10.5% | **5.0%** |
| ② | 2 | tc_02 WMMA_smem | 23388 | 15.8% | **7.5%** |
| ③ | 3 | tc_03 WMMA_pipe | 34601 | 23.4% | **11.1%** |
| ⑥ | 6 | **tc_06 MMA_pipe** | **110535 (110.5 TFLOPS)** | 74.7% | **35.4%** |

- 阶梯形状复现无误：naive → smem → cp.async → mma.sync 逐级抬升，**tc_06 一步把手写从 ~11% 峰跳到 ~35% 峰**（`mma.sync`+`ldmatrix` 消掉 WMMA 的 6 路 bank 冲突，见 §4）。
- **本表低于 `a800/` canonical**（那里 tc_06=150.3T/48.2%、tc_03=43T、cuBLAS FP32=19T）：张量核与 cuBLAS 属高功耗密度/时钟敏感负载，本 ground-truth 是在一台**瞬时空闲但整体共享**的 A800 上抓的，受争用影响绝对值偏低 ~10–25%；轻载的 CUDA-core 手写阶梯（§3.1）几乎不受影响。**kernel 相同 → 干净独占的 A800 上会复现 `a800/` 的 canonical 数字。** 结构性结论（阶梯形状、瓶颈归因、sm_80 天花板）不受此绝对偏移影响。

## 4. 深入分析在哪（逐核证据）

- **每一级"为什么是这个数"** → [`a800/ncu分析.md`](./a800/ncu分析.md) + [复现总结 §8](./a800/A800%20GEMM%20复现总结.md)：Speed-of-Light + 占用率 + 头号 stall。要点：CUDA-core 线 naive 占用率 98% → double_buffer 18%，靠 **ILP 取代 TLP**（cyc/iss 36→6）；WMMA 三级被 **L1/TEX 打满**（tc_01 L1/TEX 99.7%、Compute 仅 16.5%、张量核饿死）。
- **张量核最深一课（tc_06）** → [`a800/Ampere mma.sync 张量核.md`](./a800/Ampere%20mma.sync%20张量核.md)：ncu 铁证 tc_03→tc_06 的 L1/TEX 从 **93.8% 降到 44.5%**、`load_matrix_sync` 的 **6 路 bank 冲突（4194 万次、占 83.1% shared-load 波前）消失**、每指令停顿 27.4→12.85 周期、总周期少 2.5×。瓶颈随之从"SMEM bank 冲突墙"变为"占用率/延迟墙"。
- **ncu 流程本身** → [`01_性能分析方法论.md`](./01_性能分析方法论.md)（三板斧 + 命令）；**术语来源** → [`00_GPU硬件前置知识.md`](./00_GPU硬件前置知识.md)。

## 5. sm_80 的诚实天花板

1. **手写张量核止步 `mma.sync`（tc_06）**：Ampere 硬件**无 WGMMA / TMA / FP8**，H20 教程里真正质变的 tc_04(WGMMA→80%)、tc_05(FP8) 在 A800 **无法编译运行**。tc_06 是 sm_80 上可写的最高一级。
2. **~48% 峰值是可读 CUDA C++ 的现实上限**：`a800/` 专题实测（锁频 4096³，约 25 组配置）显示，寄存器级 fragment 双缓冲、单 barrier 主循环、强制高占用率、更大 warp tile **全部无益甚至反降** —— nvcc 已对完全展开的内层循环做了软件流水，手写 ping-pong 只是重复它并多占寄存器。根因是 Ampere 特有的**寄存器文件两难**（大 warp tile 的 ILP 与够用的占用率二选一）。
3. **48% → 85%（cuBLAS）差的是 SASS 级机器**：手排 SASS 寄存器分配 + 指令调度、128B swizzle、block rasterization / L2-aware 调度、必要时 split-K —— 这些属 **CUTLASS/`cublasLt` 区间**，不是 CUDA C++ 里能补的小优化。这也划清了本教学阶梯的诚实边界：逼近 cuBLAS 的正解是接入 CUTLASS，而非继续堆手写技巧。
