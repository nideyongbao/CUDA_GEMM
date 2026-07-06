# 03 · 综合：FA = 两个 GEMM + Online Softmax，融合在 SRAM 里

> 前置：`00`（算法）、`01`（fp32 语义）、`02`（张量核终态）。
> 本文是这条学习线的**收官**：证明 FlashAttention **几乎没有引入任何新的 CUDA 原语**。
> 它把此前两个算子（gemm、softmax）已经练熟的原语拼起来，**唯一真正新增的东西是「融合」**。

## 0. 一句话论点

$$\boxed{\text{FlashAttention} = \underbrace{QK^\top}_{\text{GEMM-1}} \;\oplus\; \underbrace{\text{online softmax}}_{\text{sc\_05}} \;\oplus\; \underbrace{PV}_{\text{GEMM-2}} \;\big|\; \text{全程驻留 SRAM，中间矩阵永不落地}}$$

- **两个 GEMM**：`S=QKᵀ`、`O=PV`，用的原语 = gemm 算子 tensor_core 那一级（`tc_06`）里学的 `mma.sync + ldmatrix + cp.async`，一模一样。
- **Online Softmax**：夹在两个 GEMM 中间的重标定/延迟归一化，用的递推 = softmax 算子 `sc_05_online` 里学的 `(m,l)` 融合扫描，一模一样。
- **融合（fusion）**：这才是 FA 的原创贡献——把 GEMM-1 的输出**留在寄存器/SRAM**直接喂 softmax，softmax 的输出**再留在寄存器**直接喂 GEMM-2，`S/P` 从不写回 HBM。加上 causal 掩码与 K/V 流式分块，就是完整的 FA。

换句话说：如果你已经能写高性能 GEMM、能写 online softmax，那么学 FA 的**增量知识量非常小**——你要学的不是新指令，而是**怎么把两个 kernel 焊成一个、让中间结果不落地**。

---

## 1. 原语交接表（Primitive Handoff）

FA 的每个零件，都能追溯到之前某个算子的某一级。这张表是本文的核心：

| FA 组成部分 | 具体做的事 | 来自哪个算子 / 哪一级 | 在 FA 里的落点 |
|---|---|---|---|
| **GEMM-1 `S=QKᵀ`** | warp 级张量核矩阵乘 | **gemm · tensor_core `tc_06_mma_pipe`**：`mma.sync.m16n8k16` + `ldmatrix` + `cp.async` 多级流水 | `mma/gemm.cuh::computeScore` |
| **GEMM-2 `O=PV`** | 同一套 MMA，第二个矩阵乘 | 同上（同一 `TiledMma`，复用 `mma.sync`） | `mma/gemm.cuh::computeOutput` |
| **smem→reg 喂料** | `ldmatrix` 直接把 smem 灌进 fragment | gemm · `tc_06`（`ldmatrix.x4/.x2`） | `mma/gemm.cuh` 的 `SmemCopyAtom` |
| **异步搬运 overlap** | `cp.async` 把 DRAM 延迟藏到计算后 | gemm · `tc_06`（`cp.async.cg` + K_STAGE ring buffer） | `mma/kernel.cuh` 主循环 + `memory.cuh` |
| **swizzle 消 bank conflict** | smem 列地址异或打乱 | gemm · tensor_core smem 布局（对应 gemm cuda_core `09_bankconflict` 的动机延续） | `mma/layout.cuh::Swizzle` |
| **Online Softmax `(m,l)` 递推** | running max/sum + 重标定 | **softmax · `sc_05_online`**：`m_new=max; l=l·exp(m-m_new)+exp(x-m_new)` | `mma/softmax.cuh::update` |
| **延迟归一化（除一次）** | epilogue 才 `/l` | softmax · `sc_05` 的延迟除法思想 + FA2 Step 4 | `mma/softmax.cuh::finalize` |
| **warp shuffle 归约** | 跨线程求行 max/sum | **softmax · `sc_03_warp_shuffle`** 的 `__shfl_xor` 归约（FA 里收窄成 warp-4） | `utils.cuh::warpReduce<4>` |
| **向量化访存** | 16B/线程 `uint128` 搬运 | gemm/softmax 的 `float4`/`uint4` 向量化（如 softmax `sc_04_vectorized`、gemm `05_vectorized`） | `memory.cuh` 的 `uint128_t` copy |
| **causal 掩码 + KV 流式** | 反向循环 + masked/unmasked 分段 | FA **原创**（`00_principle.md` FA2 循环顺序） | `mma/kernel.cuh` 双 for + `softmax.cuh::applyMask` |
| **🔴 融合（fusion）** | S、P 留寄存器，永不落 HBM | **FA 原创，本课唯一全新概念** | 整个 `flashAttnMma` 的数据流 |

读法：表里**只有最后两行是 FA 真正新增的**，而其中真正的“新原语级创新”只有**融合**这一件事（causal/KV-streaming 是调度技巧，不是新指令）。上面九行全是搬运已有肌肉记忆。

---

## 2. 为什么“融合”是唯一的新东西

把三个 kernel 摆一起看数据流就清楚了：

**如果不融合（标准做法，三个独立 kernel）：**
```
kernel A: S = Q @ Kᵀ            → S 写 HBM        (N×N)
kernel B: P = softmax(S)        → 读 S, 写 P HBM  (N×N ×2)
kernel C: O = P @ V             → 读 P, 写 O HBM
```
`S/P` 这两个 `N×N` 矩阵在 HBM 上写→读→写→读，正是 `00_principle.md` 里算出的 **192 B** 灾难（`O(N²)` 访存）。

**FA（融合成一个 kernel）：**
```
一个 block 内，对每个 KV 块循环：
  accS = Q@Kᵀ         (mma.sync)      ← accS 是寄存器 fragment
  softmax.update(accS, accO)          ← 直接吃寄存器里的 accS，重标定寄存器里的 accO
  accO += P@V         (mma.sync)      ← P(=accS) 仍在寄存器，直接当 A 操作数
最后 finalize + 写 O 一次
```
`S`（`accS`）和 `P` 从头到尾是**寄存器 fragment**，连 smem 都很少落，更别说 HBM。这就是 **96 B**。

融合之所以过去做不了、FA 才做成，卡点就在 softmax：**标准 softmax 要先看到整行 `S` 才能归一化**，逼你把 `S` 全物化。`00_principle.md` 的 Online Softmax 打破了这个约束（边扫边维护 `m,l`，最后统一归一化），**融合才成为可能**。所以：

> **Online Softmax 是“钥匙”，融合是“门”**。前者（你在 `sc_05` 已学）解锁了后者（FA 的唯一新增）。

---

## 3. 三个算子如何在这条线上叠起来

```
        softmax 算子                     gemm 算子
   sc_01 naive                      tc_01→03 WMMA
   sc_02 block_reduce               tc_06 mma.sync + ldmatrix + cp.async   ← 张量核 GEMM 肌肉
   sc_03 warp_shuffle  ← 归约         tc_04/05 WGMMA/FP8（Hopper）
   sc_04 vectorized                        │
   sc_05 online  ← (m,l) 递推、延迟归一化    │
         │                                  │
         └───────────────┬──────────────────┘
                         ▼
                 flash_attn 算子
      01: fa_cc_01 / fa_cc_02   （fp32，FA2 语义，无张量核 → 学“融合的数据流”）
      02: TinyFA CuTe           （把 fa_cc_02 的语义搬上 mma.sync，~94–96% Dao FA2）
                         ▲
              融合 = 本条线唯一的新概念
```

- 从 **softmax 算子**继承：`sc_03` 的 `__shfl_xor` 归约（→ FA 的 warp-4 行归约）、`sc_05` 的 online `(m,l)` 递推与延迟归一化（→ FA 的 `softmax.cuh`）。
- 从 **gemm 算子**继承：`tc_06` 的 `mma.sync m16n8k16 + ldmatrix + cp.async` 三件套（→ FA 的两个 GEMM 与搬运流水）、swizzle/向量化访存的动机。
- flash_attn 算子**只教一件新事**：把上面这些焊在一个 kernel 里、让 `S/P` 不落地——**融合**。

---

## 4. 学习验收：你应该能回答

1. **FA 里有几个矩阵乘？分别是什么？用什么指令？** → 两个：`S=QKᵀ`、`O=PV`，都用 `mma.sync m16n8k16`（`gemm.cuh`），与 gemm `tc_06` 同一原语。
2. **softmax 那部分和独立 softmax kernel 差在哪？** → 递推完全相同（`sc_05` 的 `(m,l)` 融合扫描），差别只是它作用在 MMA 的 **fragment** 上、归约收窄成 **warp-4**、且**不写回**（延迟到 `finalize`）。
3. **FA 到底新在哪？** → **融合**：GEMM-1 的输出留寄存器直接喂 softmax，softmax 输出留寄存器直接喂 GEMM-2，中间矩阵 `S/P` 永不落 HBM。这把 HBM 访存从 `O(N²)` 压到 `O(N·d)`（`00` 的 192→96 B）。
4. **融合为什么以前做不到、FA 才做到？** → 因为标准 softmax 要先物化整行 `S`；Online Softmax（`00` Step 3/4）解除了这个前置依赖。
5. **causal 和 KV streaming 算新原语吗？** → 不算，是调度/循环技巧（`02` §5 的反向循环 + masked/unmasked 分段），没有引入新指令。

**一句话结论**：掌握了高性能 GEMM 和 online softmax 之后，FlashAttention 的增量学习成本极低——它不是一堆新指令，而是一种**把已学原语在 SRAM 里焊成单 kernel、让中间结果不落地**的编排艺术。这就是本课的 payoff。
