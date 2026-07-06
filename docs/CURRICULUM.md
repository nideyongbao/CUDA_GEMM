# 学习阶梯（CURRICULUM）

本文把 `cuda-ops-a800` 的三个算子拉成**一条有序的学习路径**：每一级（rung）教一个技术、依赖前面某一级、并在某处**定型（reach finished form）**成为后续复用的基元。总线索是 **GEMM → softmax → FlashAttention**，FA 只新增"融合（FUSION）"。

> 约定：`R##` 是全局阶梯序号；括号里是**源码文件**与**运行 id**。所有级都遵循**先 `verify` 再 `bench`**（见 README 的验证方法论）。

---

## 0. 前置（读文档，不写 kernel）

| 前置 | 文档 | 教什么 |
| --- | --- | --- |
| P1 | `gemm/docs/00_GPU硬件前置知识.md` | SM / warp / shared memory / 带宽 / 算术强度 —— 利用率的分母从哪来 |
| P2 | `gemm/docs/01_性能分析方法论.md` | **ncu 三板斧**：看大局(Compute vs Memory) → 看延迟(stall/cyc-per-issue) → 看访存(合并/bank) |

---

## 1. 算子① GEMM · cuda_core —— 快矩阵乘（FP32）

一条从 naive 到接近 FP32 峰值的经典 SGEMM 阶梯。每一级用 ncu 证明"上一级卡在哪、这一级凭什么更快"。地面真值见 README 复现表。

| Rung | 源码 (运行 id) | 教什么（新技术） | 依赖 | 上一级瓶颈 → 本级解法 |
| --- | --- | --- | --- | --- |
| **R01** | `01_naive` (id 1) | GEMM baseline：每线程算一个 C 元素，直读 global | P1 | 算术强度太低、global 重复读 |
| **R02** | `02_smem` (id 2) | **shared-memory tiling**：分块进 smem 复用 | R01 | 复用率 → 把 A/B 块缓存进 smem |
| **R03** | `03_blocktiling` (id 3) | **1D register tiling**：每线程算一列微块 | R02 | smem 带宽 → 寄存器复用抬算术强度 |
| **R04** | `04_2Dblocktiling` (id 4) | **2D register tiling**：每线程算 TM×TN 微块 | R03 | 进一步抬算术强度 |
| **R05** | `05_vectorized` (id 5) | **float4 向量化** load/store + 转置 A 入 smem | R04 | 访存事务 → 128-bit 合并访存 |
| **R06** | `07_warptile` (id 9) | **warp-level tiling**（block/warp/thread 三层，对齐 CUTLASS 层级） | R04 | 调度 → 显式 warp tile |
| **R07** | `08_warptile_vec` (id 10) | warp tile **叠加向量化** —— ★**CUDA-core 快矩阵乘定型（17.4T ≈ 89% 峰）** | R05,R06 | warp tile + float4 一起上 |
| **R08** | `09_bankconflict` (id 11) | **smem bank-conflict 消除**（swizzle/padding） | R07 | smem bank 冲突 |
| **R09** | `10_doublebuffer` (id 12) | **double buffering / software pipelining**：预取重叠计算与访存 | R08 | 访存-计算串行 → 双缓冲重叠 |

- **旁支（自动调参）**：`include/06_autotuning.cuh` 的 `launch_at<...>` 模板 = 运行 id 6/7/8，用不同 tile 配置做 autotune，不在主阶梯上但同源。
- **交付基元**：一把**分块 + 向量化 + 双缓冲的 CUDA-core 快矩阵乘**。FA 的 cuda_core 脚手架（R19/R20）里，QKᵀ/PV 的手写 matmul 就是这套思路的缩小版。

---

## 2. 算子① GEMM · tensor_core —— 张量核 matmul

把 matmul 搬上张量核。A800 只有 WMMA + Ampere 原生 `mma.sync`（无 Hopper 的 WGMMA/TMA/FP8）。这一段的终点 **tc_06** 是整个仓库最重要的一把基元——**FA tensor_core 直接复用它**。

| Rung | 源码 (运行 id) | 教什么（新技术） | 依赖 | 瓶颈 → 解法（ncu 佐证） |
| --- | --- | --- | --- | --- |
| **R10** | `tc_01_wmma_naive` (id 1) | **WMMA API** 入门：16×16×16 fragment、`load_matrix_sync`/`mma_sync` | R05 | fragment 直取 global，L1/TEX 99.7% 打满、张量核饿死 |
| **R11** | `tc_02_wmma_smem` (id 2) | 给张量核**从 smem 喂数** | R10,R02 | 走 smem 复用，但 `load_matrix_sync` 仍压 L1/TEX |
| **R12** | `tc_03_wmma_pipe` (id 3) | **`cp.async` 多级流水线**喂 WMMA | R11,R09 | cp.async 重叠，但仍被 L1/TEX 喂数卡死（~11% 峰即到顶） |
| **R13** | `tc_06_mma_pipe` (id 6) | ★**Ampere 原生 `mma.sync.m16n8k16` + `ldmatrix` + 多级 `cp.async`** —— **张量核 matmul 定型（110.5T = 35.4% 峰）** | R12 | `ldmatrix` 绕开 `load_matrix_sync`，L1/TEX 93.8%→44.5%、消 6 路 bank 冲突、cyc/iss 27→13 |

- **Hopper 独占（A800 不编译）**：`tc_04`(WGMMA+TMA+warp-specialization)、`tc_05`(FP8) 用 sm_90 独占指令，A800 编译时 `-DNO_HOPPER` 从派发表剔除。
- **交付基元 = R13 tc_06**：Ampere 原生张量核 matmul。**这是 FA tensor_core（R21）的硬前置**——FA 的两次 matmul（QKᵀ、PV）跑在同一族 `mma.sync` + `ldmatrix` + `cp.async` 之上。深剖见 `gemm/docs/a800/Ampere mma.sync 张量核.md`。

---

## 3. 算子② softmax · cuda_core —— 快归约 + online

softmax 是**访存受限的行归约**（read x + write y = 2·M·N·4 bytes），无需张量核；`bench` 报**有效 HBM 带宽**（÷2039 GB/s 得屋顶线占比）。这一段的终点 **sc_05 online** 是 FA 的桥。

| Rung | 源码 (运行 id) | 教什么（新技术） | 依赖 | 瓶颈 → 解法 |
| --- | --- | --- | --- | --- |
| **R14** | `sc_01_naive` (id 1) | **safe softmax** 定义 + 归约 baseline：每行一 block，max/sum/normalize 各扫一遍 | P1 | 多趟全局访存、memory-bound |
| **R15** | `sc_02_block_reduce` (id 2) | **block reduction**：shared-memory 树形块内归约 | R14,R02 | 串行归约 → smem 树形并行 |
| **R16** | `sc_03_warp_shuffle` (id 3) | ★**warp shuffle 归约**（`__shfl_xor`），去掉 smem —— **快归约定型** | R15 | smem 往返 → 寄存器内 warp 归约 |
| **R17** | `sc_04_vectorized` (id 4) | **float4 向量化 + 合并访存**，逼近 HBM 屋顶线 —— **带宽定型** | R16,R05 | 访存事务 → 128-bit 合并，吃满带宽 |
| **R18** | `sc_05_online` (id 5) | ★**online softmax**：一遍流式，边扫边更新 running (max, sum) + 校正因子 rescale —— **online 技巧定型（FA 的桥）** | R17 | 多趟 → 单趟流式（为融合进 attention 铺路） |

- **交付基元 = R18 sc_05**：online-softmax（running max/sum + rescale）。**这是 FA 内层 softmax 的算法前置**——FA 里不能物化整行分数，只能像 sc_05 一样边算 QKᵀ 边在线更新。

---

## 4. 算子③ FlashAttention —— 融合（= 两次 GEMM + online softmax 融于 SRAM）

FA **不引入新的计算基元**，只把前两课的基元**融合**：两次 matmul 用张量核（复用 R13），中间 softmax 用 online（复用 R18），三者放进一个 kernel、**全程留在 SRAM**，既不物化 S=QKᵀ、也不把中间量落回 HBM。`bench` 报 TFLOPS，`verify` 对拍 fp32 CPU 参考注意力（= SDPA 数学）。

先用 cuda_core fp32 脚手架把**算法**讲清（慢，但看得见每一步），再交给 vendored TinyFA 的张量核前向拿**性能**。

| Rung | 源码 (运行 id) | 教什么（新技术） | 依赖 | 要点 |
| --- | --- | --- | --- | --- |
| **R19** | `fa_cc_01_stream` (id 1) | cuda_core fp32：**每 query 流式扫 K/V**，用 online softmax 累加 O，不物化分数矩阵 | R18 | 把 QKᵀ→softmax→PV 在寄存器里"流"起来（online softmax 落地到注意力） |
| **R20** | `fa_cc_02_tiled` (id 2) | cuda_core fp32：**Q/K/V 分块进 smem，block 处理一个 query tile，跨 K-tile 在线更新** | R19,R02 | FA2 的 **tiling + delayed normalization**（不用张量核）。教训：纯 CUDA core 慢 —— **注意力的 matmul 属于张量核** |
| **R21** | `tensor_core`（vendored TinyFA CuTe 前向；`bench/verify <fp16\|bf16> B H S D causal`） | ★**融合定型**：QKᵀ 与 PV 两次 `mma.sync` + 中间 online softmax，全程 SRAM | **R13**(mma.sync 张量核) + **R18**(online softmax) | 性能路径 & 终态，~94–96% Dao FA2。**新增的唯一东西 = FUSION** |

- **依赖收束**：R21 = R13（GEMM 的张量核 matmul）⊕ R18（softmax 的 online）⊕ 融合。R19/R20 是同一算法的 fp32 教学版，用来解释"为什么要融合、融合了什么"。
- **为什么保留慢的 cuda_core 脚手架**：R19/R20 与 R21 用**同一个 FLOP 模型**测速，直接对照就能看到"cuda_core 比张量核慢很多"——这正是 FA 必须上张量核的实证。

---

## 全局：每个技术在哪一级定型

| 技术 / 基元 | 定型于 | 被谁复用 |
| --- | --- | --- |
| CUDA-core 快矩阵乘（分块+向量化+双缓冲） | **R07** `08_warptile_vec`（+R09 双缓冲） | FA cuda_core 脚手架 R19/R20 的手写 matmul |
| Ampere 张量核 matmul（mma.sync+ldmatrix+cp.async） | **R13** `tc_06_mma_pipe` | **FA tensor_core R21（硬前置）** |
| 快归约（warp shuffle） | **R16** `sc_03_warp_shuffle` | softmax / FA 内层归约 |
| 带宽屋顶线（向量化+合并） | **R17** `sc_04_vectorized` | 所有 memory-bound 算子 |
| online softmax（running max/sum + rescale） | **R18** `sc_05_online` | **FA online-softmax R19–R21** |
| 融合（FUSION：两 matmul + online softmax 于 SRAM） | **R21** FA tensor_core | 终点 |

## 建议学习顺序

1. **P1→P2**（硬件 + ncu 方法论）——建立"利用率分母"和"读 profile"的能力。
2. **R01→R09**（GEMM cuda_core）——把 SGEMM 优化直觉打满；每级都 `verify` 再 `bench`，配 `--ncu` 看瓶颈迁移。
3. **R10→R13**（GEMM tensor_core）——从 WMMA 爬到 Ampere 原生 `mma.sync`；**R13 是后面 FA 的地基，务必吃透**。
4. **R14→R18**（softmax）——快归约 + 带宽屋顶，最后 **R18 online softmax** 是 FA 的桥。
5. **R19→R21**（FlashAttention）——先用 fp32 脚手架理解融合算法（R19/R20），再看 TinyFA 张量核前向（R21）把 R13 与 R18 融合到 SRAM，只新增 FUSION。
