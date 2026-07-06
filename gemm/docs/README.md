# GEMM 算子（H20 / sm_90a）— 索引与复现

> 本目录的 `gemm/` 算子是 **CUDA_GEMM** 手写阶梯（kernel 逐字节保持不变），从 A800(sm_80) 移植到
> **H20(sm_90a)**，只改了构建/派发的跨代际门控。移植后 **Hopper 独占的 tc_04(WGMMA+TMA)、
> tc_05(FP8) 两级现在真编真跑**（在旧架构上被 `-DNO_HOPPER` 从派发表剔除，见 `kernels/tensor_core/tc_cases.h`）。
> 本 README 只做四件事：① H20 硬件头 + 两条引擎线；② id 注册表与构建/运行入口；
> ③ 用**本机实测 ground-truth** 摆出复现表；④ 把读者指向 `baselines/**/*.details.txt` 的逐核 ncu 证据。

目标硬件：**NVIDIA H20（Hopper，CC 9.0，sm_90a，78 SM，96 GB HBM3 ≈4 TB/s，60 MB L2，228 KB smem/SM）**，CUDA 12.8 / driver 570。
三条峰值分母贯穿全文：**FP32(CUDA core) ≈ 40 TFLOPS**（78×128×2×1.98 GHz）、**BF16 张量核 = 148 TFLOPS**、
**FP8 张量核 = 296 TFLOPS**（= 2× BF16）。核心结论提前说：**GEMM 在 H20 上是 compute-bound**——4 TB/s 带宽几乎闲置，
撞的是那面被大幅裁剪的 148T 张量核墙，FP8 是把墙翻倍到 296T、夺回算力的唯一杠杆（见 §5）。

## 0. 先读什么（文档索引）

| 文档 | 作用 |
| --- | --- |
| [00_GPU硬件前置知识.md](./00_GPU硬件前置知识.md) | 一个 H20 SM 的内部结构（DRAM/L2/L1-SMEM/LSU-MIO/warp scheduler/寄存器堆），所有优化术语的来源；表内数字为 H20 实测 |
| [01_性能分析方法论.md](./01_性能分析方法论.md) | ncu 三板斧（看大局→看延迟→看访存合并/bank）的固定流程与命令（H20 口径） |
| [`../baselines/GROUNDTRUTH_cuda_gemm_h20.txt`](../baselines/GROUNDTRUTH_cuda_gemm_h20.txt) | **本机 ground-truth 原始输出**：verify 全 PASS + bench @4096³，本 README 复现表的权威口径 |
| [`../baselines/cuda_core/`](../baselines/cuda_core/) · [`../baselines/tensor_core/`](../baselines/tensor_core/) `*.details.txt` | 逐核 ncu 剖析（Speed-of-Light + 占用率），每一级"为什么是这个数"的证据，见 §4 |

## 1. 手写阶梯（H20 两条引擎线）

**CUDA core 线（FP32，`kernels/cuda_core/`）** — 天花板是 ~40 TFLOPS 那一列，瓶颈始终在访存/SRAM/调度，撞不到 FP32 算力墙的下方：

`01_naive → 02_smem → 03_blocktiling → 04_2Dblocktiling → 05_vectorized → 07_warptile → 08_warptile_vec → 09_bankconflict → 10_doublebuffer`

（文件序号跳过 `06` —— 那是 `include/06_autotuning.cuh` 的自动调参模板，在 bench 里占 id 6/7/8 三个配置，不算独立一级。）

**Tensor core 线（BF16 输入 / FP32 累加，tc_05 为 FP8 `e4m3` 输入，`kernels/tensor_core/`）** — 天花板才是 148 TFLOPS（FP8 296T）。
按**手写指令的代际**爬升，性能单调抬升：

`WMMA(C++ API) → mma.sync(warp 级 PTX) → WGMMA(warpgroup 级，Hopper 原生 +TMA) → FP8`

- **tc_01/02/03**：`nvcuda::wmma` C++ API 阶梯（naive → smem 复用 → `cp.async` 流水）。
- **tc_06**：`mma.sync.m16n8k16` + `ldmatrix` + 多级 `cp.async`，warp 级 PTX（`sm_80+` 通用），消掉 WMMA 的 bank 冲突。
- **tc_04**：`wgmma.mma_async` + TMA，warpgroup 级、**异步**执行，Hopper 原生——手写能到 **81.5% 峰**。
- **tc_05**：WGMMA + **FP8**（`e4m3`），把算力天花板从 148T **翻倍到 296T**，打到 226T，是这张算力受限的卡上夺回算力的杠杆。
- 按 bench id 是 `1→6`；按性能是 `tc_01 < tc_02 < tc_03 < tc_06 < tc_04 < tc_05`（见 §3.3）。
- **tc_04 / tc_05 是 Hopper sm_90 独占**（WGMMA / TMA / FP8），旧架构（A800/Ampere）用 `-DNO_HOPPER` 从派发表剔除、编不了；
  **H20 默认 `TC_HOPPER=1` 全 6 例真编真跑**。查表按显式 id，故 tc_06(`sm_80+` 通用)恒为 id=6，与 tc_04/05 在不在无关。

## 2. id 注册表与如何运行

### 2.1 构建（H20，默认 `sm_90a` + `TC_HOPPER=1`）

```bash
make ARCH=-arch=sm_90a all bf16 tc      # all=FP32(CUDA core)，bf16=BF16(CUDA core 对照)，tc=张量核 6 例
```

`ARCH` 默认已是 `-arch=sm_90a`、`TC_HOPPER` 默认已是 `1`，所以裸 `make all bf16 tc` 等价。
产物落在 `build/cuda_core/{bench,verify,bench_bf16,verify_bf16}` 与 `build/tensor_core/{bench,verify}`（`nvcc -arch=sm_90a -O3`，张量核额外 `-lcuda` 走 TMA driver API）。

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

### 2.3 Tensor core（`build/tensor_core/bench <id> M N K`，H20 全 6 例注册）

| tc id | 源文件 | 用例 | 手写指令 | 代际 |
| ---: | --- | --- | --- | --- |
| 1 | `tc_01_wmma_naive.cu` | WMMA_naive | `wmma` fragment 直取 global | sm_70+ |
| 2 | `tc_02_wmma_smem.cu` | WMMA_smem | `wmma` + smem 复用 | sm_70+ |
| 3 | `tc_03_wmma_pipe.cu` | WMMA_pipe | `wmma` + `cp.async` 流水 | sm_80+ |
| 4 | `tc_04_wgmma_tma_ws.cu` | WGMMA_TMA_WS | `wgmma.mma_async` + TMA | **Hopper sm_90a 独占** |
| 5 | `tc_05_wgmma_fp8.cu` | WGMMA_fp8 | WGMMA + FP8 `e4m3` | **Hopper sm_90a 独占** |
| 6 | `tc_06_mma_pipe.cu` | MMA_pipe | `mma.sync` + `ldmatrix` + 多级 `cp.async` | sm_80+ |

### 2.4 运行 / 对拍

```bash
sudo nvidia-smi -i 0 -lgc 1980                      # 锁 1980MHz 复现 ground-truth（默认 boost 会随预热漂移）
./build/cuda_core/bench    12 4096 4096 4096         # FP32 double_buffer（手写最佳，见 §3.1）
./build/tensor_core/bench   4 4096 4096 4096         # 张量核 WGMMA（~121T，见 §3.3）
./build/tensor_core/bench   5 4096 4096 4096         # 张量核 FP8（~226T，越过 148T BF16 天花板）
./build/cuda_core/verify   12 4096 4096 4096         # 单核对拍 cuBLAS SGEMM → PASS
./build/tensor_core/verify  5 4096 4096 4096         # 对拍（FP8 与 CPU double 对，误差符合 e4m3 预期）→ PASS
./build/cuda_core/bench_bf16 0 4096 4096 4096        # 对照：BF16 输入仍跑 CUDA core（证明只换数据类型没用，见 §3.2）
sudo nvidia-smi -i 0 -rgc                            # 用完解锁
```

`bench <id> M N K` 纯计时打印 GFLOPS；`verify <id> M N K` 用 `|a-b| ≤ atol+rtol|b|`（atol=rtol=1e-2）对拍 cuBLAS，打印 PASS/FAIL。二者 id 语义一致。

## 3. 本机复现表（THIS H20，锁频 1980MHz，4096³，best-of warmup2/repeat10）

下表数字为**本机实测 ground-truth**（[`GROUNDTRUTH_cuda_gemm_h20.txt`](../baselines/GROUNDTRUTH_cuda_gemm_h20.txt)）。

### 3.1 FP32 CUDA core（分母：~40T 峰值 / cublas_ref）

| bench id | kernel | GFLOPS | % 峰值(~40T) | % cuBLAS |
| ---: | --- | ---: | ---: | ---: |
| 0 | cublas_ref | 28039 | 70.1% | 100% |
| 1 | naive | 3352 | 8.4% | 12.0% |
| 2 | smem | 5042 | 12.6% | 18.0% |
| 3 | blocktiling(1D) | 8826 | 22.1% | 31.5% |
| 4 | 2D blocktiling | 14489 | 36.2% | 51.7% |
| 5 | vectorized(float4) | 21064 | 52.7% | 75.1% |
| 9 | warptile（标量） | 11902 | 29.8% | 42.4% |
| 10 | warptile_vec | 22233 | 55.6% | 79.3% |
| 11 | bank_conflict | 19926 | 49.8% | 71.1% |
| **12** | **double_buffer** | **23221** | **58.1%** | **82.8%** |

- **手写最佳 = double_buffer 23.2T ≈ 82.8% cuBLAS ≈ ~58% FP32 峰**：在 H20 上双缓冲（load/compute 重叠）终于压过 vectorized/warptile_vec，成为 FP32 阶梯顶点。
- **教学点：`warptile`（标量，11.9T）是一次负优化**——它**低于** `vectorized`(21.1T) 与 `warptile_vec`(22.2T)。warp tile 只有配上 `float4` 向量化（warptile_vec）才有意义；裸标量 warp tile 因寄存器压力反而倒退。

### 3.2 BF16 on CUDA core（分母：148T 张量核峰值）——证明"只换数据类型不够"

| id | kernel | GFLOPS | util(vs 148T) |
| ---: | --- | ---: | ---: |
| 0 | **cublas_bf16**（走张量核） | **134032** | **90.6%** |
| 10 | warptile_vec（手写最佳，仍走 CUDA core） | 22782 | 15.4% |

BF16 数据喂进 **CUDA core** kernel，手写最快也只有 15.4%——因为 CUDA core 不做张量核数学。把同一 BF16 数据交给 cuBLAS（自动走张量核）立刻到 90.6%。**结论：BF16 的价值全在张量核，换类型而不换计算单元等于白换**，这正是 §3.3 张量核线存在的理由。

### 3.3 Tensor core（分母：148T 张量核峰值；FP8 另附 296T 分母）——按性能爬升排序

| 级 | tc id | 用例 | 手写指令 | GFLOPS | util(vs 148T) |
| --- | ---: | --- | --- | ---: | ---: |
| ① | 1 | tc_01 WMMA_naive | `wmma` 直取 global | 16946 | 11.5% |
| ② | 2 | tc_02 WMMA_smem | `wmma` + smem 复用 | 30123 | 20.4% |
| ③ | 3 | tc_03 WMMA_pipe | `wmma` + `cp.async` 流水 | 38659 | 26.1% |
| ④ | 6 | tc_06 MMA_pipe | `mma.sync` + `ldmatrix` + `cp.async` | 75332 | 50.9% |
| ⑤ | 4 | **tc_04 WGMMA** | `wgmma.mma_async` + TMA（Hopper） | **120646** | **81.5%** |
| ⑥ | 5 | **tc_05 WGMMA_fp8** | WGMMA + FP8 `e4m3`（Hopper） | **226026** | **152.7%** |

- 参考天花板：**cuBLAS BF16（同一 bench 内跑）= 133802 GFLOPS = 90.4%**——手写 WGMMA(81.5%) 距 cuBLAS 只差不到 9 个百分点。
- 阶梯单调抬升：WMMA 三级(11.5%→26.1%) → **tc_06 一步跳到 50.9%**（`mma.sync`+`ldmatrix` 消掉 WMMA 的 shared bank 冲突）→ **tc_04 WGMMA 再跳到 81.5%**（warpgroup 异步指令 + TMA 直接从 global 搬进 smem，绕开寄存器绕道）。
- **tc_05 FP8 = 226026 GFLOPS**：`util(vs 296T)=76.4%`、`util(vs 148T)=152.7%`。**越过 148T BF16 天花板本身就是 FP8 的意义**——FP8 把算力上限翻倍，在这张算力受限的卡上重新夺回吞吐。

### 3.4 正确性（全部 PASS）

| 精度 | 对拍对象 | max_abs_err | 说明 |
| --- | --- | --- | --- |
| FP32 CUDA core | cuBLAS SGEMM @2048³ | 1.9e-4 | 累加顺序差异 |
| BF16 CUDA core | cuBLAS BF16 @2048³ | 2.5e-4 | 累加顺序差异 |
| tensor core tc_01–04 / tc_06 | cuBLAS @4096³ | 0（bit-exact） | 与 cuBLAS 逐位一致 |
| tensor core tc_05（FP8） | CPU double @512³ | 2.9e-2 | `e4m3` 低精度预期内 |

## 4. 逐核 ncu 证据（Speed-of-Light）

下表 4 个代表核的 ncu Speed-of-Light 摘要（原始 section 见对应 `.details.txt`）：

| 核（bench id） | `.details.txt` | DRAM Throughput | Compute(SM) Throughput | Achieved Occupancy |
| --- | --- | ---: | ---: | ---: |
| cc_10 double_buffer（cuda_core id 12） | [`cuda_core/gemm_cc_10_doublebuffer.details.txt`](../baselines/cuda_core/gemm_cc_10_doublebuffer.details.txt) | 3.5% | 69.5% | 24% |
| tc_06 mma.sync（tensor_core id 6） | [`tensor_core/gemm_tc_06_mma_pipe.details.txt`](../baselines/tensor_core/gemm_tc_06_mma_pipe.details.txt) | 5.9% | 77.2% | 24% |
| tc_04 WGMMA（tensor_core id 4） | [`tensor_core/gemm_tc_04_wgmma.details.txt`](../baselines/tensor_core/gemm_tc_04_wgmma.details.txt) | 10.3% | 82.8% | 7.7% |
| tc_05 FP8（tensor_core id 5） | [`tensor_core/gemm_tc_05_wgmma_fp8.details.txt`](../baselines/tensor_core/gemm_tc_05_wgmma_fp8.details.txt) | 3.4% | 77.9% | 7.6% |

判读要点：

- **全部 compute-bound**：DRAM Throughput 全程 <11%——**4 TB/s 的 HBM3 基本闲置**，GEMM 用不上它；真正的墙是 Compute(SM) 那一列（69%–83%），即被裁剪到只剩 148T 的张量核/FP32 管线。
- **低占用率 ≠ 低 MFU**：tc_04/tc_05 占用率只有 7–8%，却仍能打到 81.5%/76.4% MFU。原因是 **WGMMA 是异步的 warpgroup 级指令**——它不靠"挂很多 warp（TLP）"隐藏延迟，而是靠单个 warpgroup 的 ILP 持续喂满张量核流水线。这与 CUDA core 线的经验（`double_buffer` 占用率仅 24% 却最快，靠 ILP 取代 TLP）是同一条道理。

方法论本身见 [`01_性能分析方法论.md`](./01_性能分析方法论.md)（三板斧 + 命令）；术语来源见 [`00_GPU硬件前置知识.md`](./00_GPU硬件前置知识.md)。

## 5. H20 的故事：GEMM 是 compute-bound，FP8 是唯一杠杆

1. **H20 是被大幅裁剪的 Hopper**：张量核天花板只有 **148 TFLOPS**（H100/H200 的零头），连 cuBLAS BF16 也只到 ~134T(90.6%)。这张卡上 GEMM 的瓶颈**不在 4 TB/s 带宽**（DRAM<11%，全程闲置），而在这面矮墙本身。
2. **手写阶梯诚实地爬到 81.5% 峰**：`WMMA(C++ API) → mma.sync(warp 级 PTX) → WGMMA(warpgroup 级，Hopper 原生 +TMA)`，tc_04 到 120.6T，距 cuBLAS(90.4%) 只差不到 9 点——剩下的是 SASS 级机器（手排寄存器/调度、128B swizzle、L2-aware rasterization、split-K），属 CUTLASS/`cublasLt` 区间。
3. **FP8(tc_05) 是唯一能翻墙的杠杆**：`e4m3` 把算力上限从 148T **翻倍到 296T**，tc_05 打到 **226T = 296T 的 76.4% = 148T BF16 天花板的 152.7%**。在一张算力被砍、带宽过剩的卡上，**降精度（FP8）而非堆带宽，才是夺回吞吐的正解**。这也是移植到 H20 后 tc_04/tc_05 真编真跑带来的最大新增结论——它们在旧架构（A800/Ampere）上根本无法存在。
