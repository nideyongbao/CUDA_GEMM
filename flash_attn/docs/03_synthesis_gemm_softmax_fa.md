# 03 · 综合：FA = 两个 GEMM + Online Softmax，融合在 SRAM 里

> 前置：`00`（算法）、`01`（fp32 语义）、`02`（张量核终态）。
> 本文是这条学习线的**收官**：证明 FlashAttention **几乎没有引入任何新的 CUDA 原语**。
> 它把此前两个算子（gemm、softmax）已经练熟的原语拼起来，**唯一真正新增的东西是「融合」**。

## 0. 一句话论点

$$\boxed{\text{FlashAttention} = \underbrace{QK^\top}_{\text{GEMM-1}} \;\oplus\; \underbrace{\text{online softmax}}_{\text{sc\_05}} \;\oplus\; \underbrace{PV}_{\text{GEMM-2}} \;\big|\; \text{全程驻留 SRAM，中间矩阵永不落地}}$$

- **两个 GEMM**：`S=QKᵀ`、`O=PV`，用的原语 = gemm 算子 tensor_core 的 Hopper 那一级（`tc_04_wgmma_tma_ws`）里学的 `WGMMA + TMA`，一模一样。
- **Online Softmax**：夹在两个 GEMM 中间的重标定/延迟归一化，用的递推 = softmax 算子 `sc_05_online` 里学的 `(m,l)` 融合扫描，一模一样。
- **融合（fusion）**：这才是 FA 的原创贡献——把 GEMM-1 的输出**留在寄存器/SRAM**直接喂 softmax，softmax 的输出**再留在寄存器**直接喂 GEMM-2，`S/P` 从不写回 HBM。加上 causal 掩码与 K/V 流式分块，就是完整的 FA。

换句话说：如果你已经能写高性能 GEMM、能写 online softmax，那么学 FA 的**增量知识量非常小**——你要学的不是新指令，而是**怎么把两个 kernel 焊成一个、让中间结果不落地**。

---

## 1. 原语交接表（Primitive Handoff）

FA 的每个零件，都能追溯到之前某个算子的某一级。这张表是本文的核心：

| FA 组成部分 | 具体做的事 | 来自哪个算子 / 哪一级 | 在 FA 里的落点 |
|---|---|---|---|
| **GEMM-1 `S=QKᵀ`** | warpgroup 级张量核矩阵乘（smem×smem） | **gemm · tensor_core `tc_04_wgmma_tma_ws`**：`wgmma.mma_async.m64n64k16` | `fa_hopper.cuh::wgmma64_ss_*`（主循环 S 段） |
| **GEMM-2 `O=PV`** | 同一族 WGMMA，第二个矩阵乘（A 来自寄存器） | 同上（WGMMA 的 reg×smem 变体） | `fa_hopper.cuh::wgmma64_rs_*`（A=P 寄存器、B=V smem） |
| **smem→张量核 喂料** | WGMMA 用 smem 矩阵描述符直接寻址 smem 操作数（免 `ldmatrix`）；P 则以寄存器直喂 | gemm · `tc_04`（128B-swizzle smem 描述符） | `fa_hopper.cuh::smem_desc` |
| **异步搬运 overlap** | `TMA` 把 K/V 块 global→smem，双缓冲 + mbarrier 把 DRAM 延迟藏到 WGMMA 后 | gemm · `tc_04`（`cp.async.bulk.tensor` + 双 buffer + `cuda::barrier`） | `fa_hopper.cuh` 主循环 `cp_async_bulk_tensor_2d_*` + `bar[2]` |
| **swizzle 消 bank conflict** | 128B swizzle 打乱 smem 列地址 | gemm · `tc_04` 的 TMA 128B swizzle（承接 gemm cuda_core `09_bankconflict` 的动机） | `fa_hopper.cuh::make_tma`（`SWIZZLE_128B`）+ `smem_desc` 的 128B swizzle 位 |
| **Online Softmax `(m,l)` 递推** | running max/sum + 重标定 | **softmax · `sc_05_online`**：`m_new=max; l=l·exp(m-m_new)+exp(x-m_new)` | `fa_hopper.cuh` 主循环 `m_i/l_i` 在线更新 |
| **延迟归一化（除一次）** | epilogue 才 `/l` | softmax · `sc_05` 的延迟除法思想 + FA2 Step 4 | `fa_hopper.cuh` epilogue（`inv=1/l_i`） |
| **warp shuffle 归约** | 跨线程求行 max/sum | **softmax · `sc_03_warp_shuffle`** 的 `__shfl_xor` 归约（FA 里收窄成 warp-4：4 lane 共享一行） | `fa_hopper.cuh` 的 `__shfl_xor_sync`（lane%4 组归约 `mtile/lsum`） |
| **块级张量搬运** | TMA 一次搬整块 `[Bc,D]`（取代 per-thread 向量 load） | gemm/softmax 向量化访存的动机（`sc_04_vectorized`）在 Hopper 上由 TMA 承担 | `fa_hopper.cuh` 的 `cp.async.bulk.tensor` |
| **causal 掩码 + KV 流式** | 块级跳过未来 tile + 对角块逐元素掩码 | FA **原创**（`00_principle.md` FA2 循环顺序） | `fa_hopper.cuh` 的 `kv_last` + softmax 段 `-INF` 掩码 |
| **🔴 融合（fusion）** | S、P 留寄存器，永不落 HBM；且 matmul-1 累加器布局 = matmul-2 的 A 操作数布局 | **FA 原创，本课唯一全新概念** | 整个 `fa_hopper.cuh::fa_kernel` 的数据流 |

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
一个 warpgroup 内，对每个 KV 块循环：
  sd = Q@Kᵀ           (WGMMA ss)      ← sd 是寄存器累加器 (m64n64)
  online-softmax(sd) → P              ← 直接吃寄存器里的 sd，重标定寄存器里的 od
  od += P@V           (WGMMA rs)      ← P 仍在寄存器，直接当 A 操作数
最后 epilogue（/l）+ 写 O 一次
```
`S`（`sd`）和 `P` 从头到尾是**寄存器**里的量，连 smem 都很少落，更别说 HBM。这就是 **96 B**。

**张量核上跑快的命门**：matmul-1 的 `m64n64k16` WGMMA **累加器**布局，恰好逐位等于 matmul-2 的 **A 操作数** fragment 布局——所以 `softmax(S)=P` 算完就地留在寄存器、直接当 `P·V` 的 A 喂回去，P 连 smem 往返都省了。这正是"融合"能真正落到张量核上的物理前提。

融合之所以过去做不了、FA 才做成，卡点就在 softmax：**标准 softmax 要先看到整行 `S` 才能归一化**，逼你把 `S` 全物化。`00_principle.md` 的 Online Softmax 打破了这个约束（边扫边维护 `m,l`，最后统一归一化），**融合才成为可能**。所以：

> **Online Softmax 是“钥匙”，融合是“门”**。前者（你在 `sc_05` 已学）解锁了后者（FA 的唯一新增）。

---

## 3. 三个算子如何在这条线上叠起来

```
        softmax 算子                     gemm 算子
   sc_01 naive                      tc_01→03 WMMA
   sc_02 block_reduce               tc_06 mma.sync（Ampere 原生）
   sc_03 warp_shuffle  ← 归约         tc_04 WGMMA+TMA / tc_05 FP8（Hopper）← 张量核 GEMM 肌肉
   sc_04 vectorized                        │
   sc_05 online  ← (m,l) 递推、延迟归一化    │
         │                                  │
         └───────────────┬──────────────────┘
                         ▼
                 flash_attn 算子
      01: fa_cc_01 / fa_cc_02   （fp32，FA2 语义，无张量核 → 学“融合的数据流”）
      02: fa_hopper WGMMA+TMA    （把 fa_cc_02 的语义搬上 Hopper 张量核，122T = 148T 峰的 82.8%）
                         ▲
              融合 = 本条线唯一的新概念
```

- 从 **softmax 算子**继承：`sc_03` 的 `__shfl_xor` 归约（→ FA 的 warp-4 行归约）、`sc_05` 的 online `(m,l)` 递推与延迟归一化（→ FA 的在线更新，`fa_hopper.cuh`）。
- 从 **gemm 算子**继承：`tc_04` 的 `WGMMA m64n64k16 + TMA`（Hopper 原生异步张量核路径，→ FA 的两个 GEMM 与 TMA 搬运）、128B swizzle 的动机。
- flash_attn 算子**只教一件新事**：把上面这些焊在一个 kernel 里、让 `S/P` 不落地——**融合**。

---

## 4. 学习验收：你应该能回答

1. **FA 里有几个矩阵乘？分别是什么？用什么指令？** → 两个：`S=QKᵀ`、`O=PV`，都用 `WGMMA m64n64k16`（`fa_hopper.cuh`），与 gemm `tc_04`（WGMMA+TMA）同一原语。
2. **softmax 那部分和独立 softmax kernel 差在哪？** → 递推完全相同（`sc_05` 的 `(m,l)` 融合扫描），差别只是它作用在 WGMMA 的 **累加器**上、归约收窄成 **warp-4**（4 lane 共享一行）、且**不写回**（延迟到 epilogue）。
3. **FA 到底新在哪？** → **融合**：GEMM-1 的输出留寄存器直接喂 softmax，softmax 输出（=P）留寄存器直接喂 GEMM-2（matmul-1 的累加器布局 = matmul-2 的 A 操作数布局，P 无需过 smem），中间矩阵 `S/P` 永不落 HBM。这把 HBM 访存从 `O(N²)` 压到 `O(N·d)`（`00` 的 192→96 B）。
4. **融合为什么以前做不到、FA 才做到？** → 因为标准 softmax 要先物化整行 `S`；Online Softmax（`00` Step 3/4）解除了这个前置依赖。
5. **causal 和 KV streaming 算新原语吗？** → 不算，是调度/循环技巧（`fa_hopper.cuh` 里 causal 把 `kv_last` 卡在本 tile 需要的最后一个 K 块、再对对角块逐元素补 `-INF`），没有引入新指令。

---

## 5. H20 视角：算力穷、带宽富，FA 是把两端焊起来的综合体

同一条学习线跑在 **H20（Hopper, sm_90a, CC 9.0）** 上，三个算子会落到 roofline 的两端——因为 H20 是一块**算力穷、带宽富**的卡：BF16 张量峰只有 **148 TFLOPS**，HBM3 带宽却高达 **4000 GB/s（4 TB/s）**。

| 算子 | 屋顶线 | H20 实测 | 受限于 |
|---|---|---|---|
| **GEMM**（大矩阵乘） | 148T 张量峰 | cuBLAS BF16 **90.6%**、手写 WGMMA **81.5%**；**FP8 226 TFLOPS = BF16 峰的 152%** | **算力**（compute-bound）——上 FP8 才能把算力“要”回来 |
| **softmax**（行归约） | 4 TB/s HBM | **77–89%** 有效带宽 | **带宽**（bandwidth-bound）——只读一遍写一遍，在带宽富的 H20 上“飞起来” |
| **FlashAttention** | 148T 张量峰 | **80–86%**（`fa_hopper` 122 TFLOPS = **82.8%**） | **算力**——但前提是别把 `S/P` 砸进 HBM |

FA 正是这条线的**综合体**：它 = 两个 GEMM（吃算力）+ 一个 online softmax（那个归约）。若按标准 attention 拆成三个独立 kernel，中间的 `S/P` 会砸进 HBM，把一个本该 compute-bound 的算子拖成 bandwidth-bound（`00` 的 192B 灾难）。**融合**把 `S/P` 摁在寄存器/SRAM 里，让 FA 稳稳停在**算力受限**这一端——在 H20 上跑到 148T 峰的 80–86%，而不是反被 4 TB/s 的 HBM 卡住。

一句话：**softmax 单独看是带宽算子、在 H20 上轻松吃满带宽；两个 GEMM 单独看是算力算子；FA 把它们焊在一起、让归约藏进张量核的间隙、`S/P` 不落地，于是整体呈现 GEMM 那一端的 compute-bound 特征。** 这就是“融合”在 H20 这块具体硬件上的收益。

---

**一句话结论**：掌握了高性能 GEMM 和 online softmax 之后，FlashAttention 的增量学习成本极低——它不是一堆新指令，而是一种**把已学原语在 SRAM 里焊成单 kernel、让中间结果不落地**的编排艺术。这就是本课的 payoff。
