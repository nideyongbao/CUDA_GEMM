# 学习阶梯（CURRICULUM）— H20 / sm_90a

本文把 `cuda-ops-h20` 的三个算子拉成**一条有序的学习路径**：每一级（rung）教一个技术、
依赖前面某一级、并在某处**定型（reach finished form）**成为后续复用的基元。总线索是
**GEMM → softmax → FlashAttention**，FA 只新增"融合（FUSION）"。

> 约定：`R##` 是全局阶梯序号；括号里是**源码文件**与**运行 id**。所有级都遵循**先
> `verify` 再 `bench`**。硬件为 **NVIDIA H20（Hopper, sm_90a, 78 SM）**：算力弱
> （BF16 148 TFLOPS / FP8 296 TFLOPS / FP32 ~40 TFLOPS）、带宽大（HBM3 ~4000 GB/s）。

---

## 0. 前置（读文档，不写 kernel）

| 前置 | 文档 | 教什么 |
| --- | --- | --- |
| P1 | `gemm/docs/00_GPU硬件前置知识.md` | SM / warp / shared memory / 带宽 / 算术强度 —— 利用率的分母从哪来 |
| P2 | `gemm/docs/01_性能分析方法论.md` | **ncu 三板斧**：看大局(Compute vs Memory) → 看延迟(stall/cyc-per-issue) → 看访存(合并/bank) |

---

## 1. 算子① GEMM · cuda_core —— 快矩阵乘（FP32，天花板 ~40 TFLOPS）

一条从 naive 到接近 FP32 峰值的经典 SGEMM 阶梯。每一级用 ncu 证明"上一级卡在哪、这一级
凭什么更快"。地面真值见 `gemm/baselines/GROUNDTRUTH_cuda_gemm_h20.txt`。

| Rung | 源码 (运行 id) | 教什么（新技术） | 依赖 | 上一级瓶颈 → 本级解法 |
| --- | --- | --- | --- | --- |
| **R01** | `01_naive` (id 1) | GEMM baseline：每线程算一个 C 元素，直读 global | P1 | 算术强度太低、global 重复读 |
| **R02** | `02_smem` (id 2) | **shared-memory tiling**：分块进 smem 复用 | R01 | 复用率 → 把 A/B 块缓存进 smem |
| **R03** | `03_blocktiling` (id 3) | **1D register tiling**：每线程算一列微块 | R02 | smem 带宽 → 寄存器复用抬算术强度 |
| **R04** | `04_2Dblocktiling` (id 4) | **2D register tiling**：每线程算 TM×TN 微块 | R03 | 进一步抬算术强度 |
| **R05** | `05_vectorized` (id 5) | **float4 向量化** load/store + 转置 A 入 smem | R04 | 访存事务 → 128-bit 合并访存 |
| **R06** | `07_warptile` (id 9) | **warp-level tiling**（block/warp/thread 三层，对齐 CUTLASS 层级） | R04 | 调度 → 显式 warp tile（标量版是**负优化**，须配 float4） |
| **R07** | `08_warptile_vec` (id 10) | warp tile **叠加向量化**（22.2T） | R05,R06 | warp tile + float4 一起上 |
| **R08** | `09_bankconflict` (id 11) | **smem bank-conflict 消除**（swizzle/padding） | R07 | smem bank 冲突 |
| **R09** | `10_doublebuffer` (id 12) | ★**double buffering / software pipelining** —— **CUDA-core 快矩阵乘定型（23.2T = 82.8% cuBLAS SGEMM）** | R08 | 访存-计算串行 → 双缓冲重叠 |

- **旁支（自动调参）**：`include/06_autotuning.cuh` 的 `launch_at<...>` = 运行 id 6/7/8。
- **交付基元**：一把**分块 + 向量化 + 双缓冲的 CUDA-core 快矩阵乘**。FA 的 cuda_core 脚手架
  （R21/R22）里 QKᵀ/PV 的手写 matmul 就是这套思路的缩小版。

---

## 2. 算子① GEMM · tensor_core —— 张量核 matmul（天花板 148 / FP8 296 TFLOPS）

把 matmul 搬上张量核。**H20 是 Hopper**：WMMA、Ampere 原生 `mma.sync`、以及 Hopper 独占的
**WGMMA + TMA**、**FP8** 全都能跑。这一段终点 **tc_04（WGMMA）是 FA tensor_core 的硬前置**，
**tc_05（FP8）是在算力弱的 H20 上收回算力的杠杆**。

| Rung | 源码 (运行 id) | 教什么（新技术） | 依赖 | 瓶颈 → 解法（GFLOPS / 占 148T） |
| --- | --- | --- | --- | --- |
| **R10** | `tc_01_wmma_naive` (id 1) | **WMMA API** 入门：16×16×16 fragment | R05 | fragment 直取 global，张量核饿死（16946 / 11.5%） |
| **R11** | `tc_02_wmma_smem` (id 2) | 给张量核**从 smem 喂数** | R10,R02 | smem 复用（30123 / 20.4%） |
| **R12** | `tc_03_wmma_pipe` (id 3) | **`cp.async` 多级流水**喂 WMMA —— **wmma API 天花板** | R11,R09 | 仍被 L1/TEX 喂数卡死（38659 / 26.1%） |
| **R13** | `tc_06_mma_pipe` (id 6) | **warp 级原生 `mma.sync.m16n8k16` + `ldmatrix` + `cp.async`**（绕开 WMMA API） | R12 | `ldmatrix` 直喂 fragment（75332 / 50.9%） |
| **R14** | `tc_04_wgmma_tma_ws` (id 4) | ★**Hopper 原生 WGMMA + TMA**（warpgroup 级异步张量指令 + 128B-swizzle 异步搬运） —— **手写张量核 matmul 定型（120646 / 81.5%）** | R13 | warp→warpgroup 异步 + TMA（cuBLAS BF16 90.4%） |
| **R15** | `tc_05_wgmma_fp8` (id 5) | ★**FP8（e4m3）WGMMA**：把算力天花板从 148T 抬到 296T | R14 | FP8 输入（226026 GFLOPS = 76.4% of 296T = **152.7% of 148T**） |

- ncu：R14/R15 的 **DRAM 仅 3–10%**（4 TB/s 闲置），占用率 7–8% 却拿 81% MFU——
  **WGMMA 异步，低占用不碍事**。深剖见 `gemm/docs/`。
- **交付基元 = R14 tc_04（WGMMA+TMA）**：Hopper 原生张量核 matmul。**这是 FA tensor_core
  （R23）的硬前置**——FA 的 `fa_hopper.cuh` 复用它的 128B-swizzle 描述符几何与 WGMMA/TMA。

---

## 3. 算子② softmax · cuda_core —— 快归约 + online（天花板 = HBM 4 TB/s）

softmax 是**访存受限的行归约**（read x + write y = 2·M·N·4 bytes），无需张量核；`bench` 报
**有效 HBM 带宽**（÷4000 GB/s 得屋顶线占比）。这一段终点 **sc_05 online** 是 FA 的桥。

| Rung | 源码 (运行 id) | 教什么（新技术） | 依赖 | 瓶颈 → 解法（占 4000 GB/s） |
| --- | --- | --- | --- | --- |
| **R16** | `sc_01_naive` (id 1) | **safe softmax** 定义 + 归约 baseline：max/sum/normalize 各扫一遍 | P1 | 多趟全局访存（2.6%） |
| **R17** | `sc_02_block_reduce` (id 2) | **block reduction**：smem 树形块内归约 | R16,R02 | 串行→smem 树形并行（62.3%） |
| **R18** | `sc_03_warp_shuffle` (id 3) | ★**warp shuffle 归约**（`__shfl_xor`），去掉 smem —— **快归约定型** | R17 | smem 往返 → 寄存器内 warp 归约 |
| **R19** | `sc_04_vectorized` (id 4) | ★**float4 向量化 + 合并访存**，逼近 HBM 屋顶 —— **带宽定型（79.1%）** | R18,R05 | 访存事务 → 128-bit 合并 |
| **R20** | `sc_05_online` (id 5) | ★**online softmax**：一遍流式，边扫边更新 running (max, sum) + rescale —— **online 定型（FA 的桥，77%）** | R19 | 多趟 → 单趟流式 |

- 旁支 `sc_06_resident`（id 6）：寄存器常驻整行、单次 DRAM 读，长行（16384）达 **88.9% 带宽**。
- ncu：sc_05 DRAM 83.1% / 占用 90%——**访存受限算子在 H20 的 4 TB/s 上如鱼得水**。
- **交付基元 = R20 sc_05**：online-softmax（running max/sum + rescale）。**这是 FA 内层
  softmax 的算法前置**——FA 里不能物化整行分数，只能像 sc_05 一样边算 QKᵀ 边在线更新。

---

## 4. 算子③ FlashAttention —— 融合（= 两次 GEMM + online softmax 融于 SRAM）

FA **不引入新的计算基元**，只把前两课的基元**融合**：两次 matmul 上张量核（复用 R14 的
WGMMA），中间 softmax 用 online（复用 R20），三者放进一个 kernel、**全程留在 SRAM**，既不
物化 S=QKᵀ、也不把中间量落回 HBM。`bench` 报 TFLOPS，`verify` 对拍 fp32 CPU 参考注意力。

先用 cuda_core fp32 脚手架把**算法**讲清（慢，但看得见每一步），再交给**手写 Hopper
WGMMA+TMA 前向**拿**性能**。

| Rung | 源码 (运行 id) | 教什么（新技术） | 依赖 | 要点 |
| --- | --- | --- | --- | --- |
| **R21** | `fa_cc_01_stream` (id 1) | cuda_core fp32：**每 query 流式扫 K/V**，用 online softmax 累加 O，不物化分数 | R20 | online softmax 落地到注意力（~0.44T） |
| **R22** | `fa_cc_02_tiled` (id 2) | cuda_core fp32：**Q/K/V 分块进 smem，block 处理一个 query tile，跨 K-tile 在线更新** | R21,R02 | FA2 的 **tiling + delayed normalization**（~0.65T）。教训：纯 CUDA core 慢 ~180× —— **注意力的 matmul 必须上张量核** |
| **R23** | `tensor_core`（手写 `fa_hopper.cuh`；`bench/verify <fp16\|bf16> B H S D causal`） | ★**融合定型**：QKᵀ 与 PV 两次 **WGMMA** + 中间**寄存器常驻 online softmax**，K/V 走 **TMA**，全程 SRAM（**122 TFLOPS = 82.8% 峰**） | **R14**(WGMMA+TMA) + **R20**(online softmax) | 性能路径 & 终态。**新增的唯一东西 = FUSION**。关键：P 直接当作 PV 的 A-fragment（无 smem 往返） |

- **依赖收束**：R23 = R14（GEMM 的 WGMMA+TMA 张量核 matmul）⊕ R20（softmax 的 online）⊕
  融合。R21/R22 是同一算法的 fp32 教学版，解释"为什么要融合、融合了什么"。深剖见
  `flash_attn/docs/02_tensor_core_hopper.md`。
- **为什么保留慢的 cuda_core 脚手架**：R21/R22 与 R23 用**同一个 FLOP 模型**测速，直接对照
  就看到 cuda_core 比张量核慢 ~180×——这正是 FA 必须上张量核的实证。

---

## 全局：每个技术在哪一级定型

| 技术 / 基元 | 定型于 | 被谁复用 |
| --- | --- | --- |
| CUDA-core 快矩阵乘（分块+向量化+双缓冲） | **R09** `10_doublebuffer` | FA cuda_core 脚手架 R21/R22 的手写 matmul |
| Hopper 张量核 matmul（WGMMA + TMA） | **R14** `tc_04_wgmma_tma_ws` | **FA tensor_core R23（硬前置）** |
| FP8 张量核（把算力天花板翻倍） | **R15** `tc_05_wgmma_fp8` | 算力受限场景收回算力 |
| 快归约（warp shuffle） | **R18** `sc_03_warp_shuffle` | softmax / FA 内层归约 |
| 带宽屋顶线（向量化+合并） | **R19** `sc_04_vectorized` | 所有 memory-bound 算子 |
| online softmax（running max/sum + rescale） | **R20** `sc_05_online` | **FA online-softmax R21–R23** |
| 融合（FUSION：两 matmul + online softmax 于 SRAM） | **R23** FA tensor_core | 终点 |

## 建议学习顺序

1. **P1→P2**（硬件 + ncu 方法论）——建立"利用率分母"和"读 profile"的能力。
2. **R01→R09**（GEMM cuda_core）——把 SGEMM 优化直觉打满；每级 `verify` 再 `bench`，配 `--ncu` 看瓶颈迁移。
3. **R10→R15**（GEMM tensor_core）——从 WMMA 爬到 Ampere `mma.sync`，再到 **Hopper WGMMA+TMA（R14）** 与 **FP8（R15）**；**R14 是后面 FA 的地基，务必吃透**。
4. **R16→R20**（softmax）——快归约 + 带宽屋顶，最后 **R20 online softmax** 是 FA 的桥。
5. **R21→R23**（FlashAttention）——先用 fp32 脚手架理解融合算法（R21/R22），再看手写 Hopper WGMMA+TMA 前向（R23）把 R14 与 R20 融合到 SRAM，只新增 FUSION。

## H20 一条主线

三个算子把「**算力弱、带宽大**」讲透：GEMM 与 FA **计算受限**（DRAM<11%，被 148T 张量核卡住
→ FP8 是杠杆）；softmax **访存受限**（DRAM 60–88%，被 4 TB/s 托起）。异步 WGMMA 让低占用率
也高 MFU——这条 Hopper 经验在带宽富余的 H20 上被放大：矩阵乘从不挨饿。
