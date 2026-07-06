# 超越 online：手写 softmax 还能怎么再快？（含 sc_06 实测 + 参考 TinyFA / flash-attention）

> 起点：`sc_05 online` @8192² = **3075 GB/s（77.0% 屋顶）**。问题：能不能更贴近 H20 的 **4 TB/s** 屋顶？答案是 `sc_06` 寄存器驻留、**单次 DRAM 读**——它在**长行**（16384²）上冲到 **3557 GB/s（88.9% 屋顶）**、贴住单次读理想；但在默认的 8192² 上它只有 **2575 GB/s（64.4%）**，**反而低于 sc_05**。这一档的核心教训不是“sc_06 永远最快”，而是：**独立算子要贴屋顶线，先想“能不能只读一次”——而在 H20 的大 L2 下，这份收益要等行足够长才兑现**。

## 1. 诊断：softmax 是 DRAM-bound，唯一杠杆是“少落到 DRAM 的读”——但 H20 的大 L2 改写了账本

ncu 铁证：`sc_05` 的 **DRAM Throughput = 83.1%**、`sc_06` = **88.0%**。softmax 是**访存受限**——它早已把 DRAM 用满，SFU/exp 不是瓶颈（`sc_06` 占用仅 **24%** 仍能把 DRAM 打到 88%，说明连占用率都不是限制）。所以“更快”只有一个杠杆：**减少真正落到 DRAM 的 x 读**。

一行 softmax 的 DRAM 流量账（M·N 个元素、每个 4B）：

| kernel | 名义 x 读 | 名义总流量 | 实测 @8192² | 实测 @16384² |
| --- | :-: | :-: | ---: | ---: |
| sc_02 block | 3 | 4·MN | 2491 (62.3%) | 1667 (41.7%) |
| sc_03 warp | 3 | 4·MN | 1663 (41.6%) | — |
| sc_04 vectorized | 3 | 4·MN | 3163 (79.1%) | 2442 (61.1%) |
| sc_05 online | 2 | 3·MN | 3075 (77.0%) | 2451 (61.3%) |
| **sc_06 resident** | **1** | **2·MN=理想** | 2575 (64.4%) | **3557 (88.9%)** |

**H20 的账本被 L2 改写**：60 MB L2 把 sc_02/04/05 的“重复读”大半**就地命中**，所以在 8192 上它们的实际 DRAM 流量都已≈2·MN——这就是为什么 sc_04 ≈ sc_05、且都逼近甚至盖过 sc_06。sc_06 显式“单次读”的优势在 8192 被 L2 抹平（还倒贴 24% 低占用）。**只有把行拉到 16384**（一行 64 KB，L2 藏不住多趟重复读）时，账本才回到“读次数”主导，sc_06 的单次读一举夺冠（88.9%），而多趟 kernel 全部回落到 ~41–61%。

## 2. 可提升方向（按收益排序）

### ★ (A) 整行缓存、单次 DRAM 读 —— 已实现 = `sc_06`
第一次读 x 就把该行**留在寄存器**（每线程 `float4 reg[]`），之后 max→sum→写全在片上算，**x 只读一次**（3·MN→2·MN）。
- **长行（16384²）实测 3557 GB/s = 88.9% 屋顶 —— 夺冠**，贴住单次读理想。ncu：DRAM **88.0%**、占用 **24%**（访存受限，低占用无妨：24% 占用仍把 DRAM 打满）。
- **默认（8192²）实测仅 2575 GB/s = 64.4%，反而低于 sc_05（77.0%）**。原因：H20 的 60 MB L2 已把 sc_04/sc_05 的重复读就地命中，给了它们一个**事实上的单次 DRAM 读**；此时 sc_06 显式缓存到寄存器没有额外好处，反被 24% 的低占用小拖一把。

**这是最大的一条——但收益只在“行足够长、L2 藏不住重复读”时兑现。** 它也把本课的中心教训点明：贴屋顶线的关键是“**只读一次**”，只是要在 L2 遮不住的规模上才看得见。

### (B) 按 N 分派（dispatch-by-N）—— sc_06 的短板与生产库的做法
`sc_06`（block-per-row，256 线程）只在**足够长的行**才赢。在 H20 上这一点更极端：连默认的 **N=8192 它都已落后**（64.4% < sc_05 77.0%），要到 **N=16384** 才反超（88.9%）。行越短，整块的 block-reduce（2× `__syncthreads` + smem）与低占用越盖过“单次读”的收益；而 H20 的大 L2 又把多趟 kernel 的重复读**免费藏住**，进一步推高了“寄存器驻留单次读开始划算”的 N 门槛。生产库（cuDNN / PyTorch / **OneFlow** 的经典三分派）按 N 选实现：
- **小 N（≲1024）**：**warp-per-row**——一个 warp 一行，元素进寄存器，纯 `__shfl` 归约、无 block sync；
- **中 N（能进寄存器/smem）**：本 `sc_06` register-resident 单读；
- **超大 N（放不下片上）**：退回 **online 流式**（`sc_05`）——这才是 online 的真正用武之地（见 §4）。
> 下一个 rung `sc_07` 就是 warp-per-row，补上小 N 段（也正好修 sc_03 在长 N 上暴露的“每行并行度不足”另一面）。

### (C) fp16/bf16 I/O + fp32 归约 —— 直接砍一半流量
本教程 softmax 走 fp32。真实场景（attention 的 mask/bias、logits）多为 fp16/bf16——H20 的张量核路径本就跑 BF16/FP8：**读写用半精度、归约上 f4 fp32** → DRAM 流量减半 → 再 ~2×。这正是 flash-attention `layer_norm` 的写法（`.to(tl.float32)` 归约、half 存回）。

### (D) exp2f + 折叠 log2e —— 已在 sc_06 采用（借鉴 TinyFA）
把任何外部 scale 与 log2(e) **预折成一个常数**，用硬件 `exp2f` 代替 `expf`：`exp2f((x-m)·log2e)`。DRAM 受限时收益小，但是零成本好习惯、且与 FA 内层口径一致。

### (E) 占用率/尾效应微调
sc_06 占用 **24%**（寄存器所限）；对访存受限核，只要够喂满 DRAM 即可（它 24% 占用仍把 DRAM 打到 88.0%，**占用率不是限制**）。可试 `blockDim=128`、或每线程更少 float4 提占用；M 不是 SM（H20 为 78 个）整数倍时的尾块可用 grid-stride 缓解。属最后 5% 的调参。

## 3. 能参考 TinyFA / flash-attention 的写法吗？——能，但要分清“融合 vs 独立”

**两者的 softmax 都是 FUSED（融合在别的 kernel 里），不是独立算子**——这是关键区别，决定了哪些能抄、哪些不能。

### TinyFA（`mma/softmax.cuh` / `fma/softmax.cuh`）
- **本质**：softmax 作用在 **QK^T 刚算出、已驻留寄存器**的 score 分片上，**全程不碰 DRAM**。它的 online (m,l) 递推、`accO*=scale` 输出重标定，都是**为了跨 KV 分块流式 + 融合 P·V** 才存在的。
- **可借鉴（sc_06 已采用）**：① `warpReduce{Max,Sum}` 的 `__shfl_xor` 蝶形归约（宽度按你的“线程→元素”布局选，TinyFA 的宽度-4 是其 MMA 分片布局的产物，别照搬 4）；② 先在寄存器里累局部 partial、再一次 warp 归约的**两段式**；③ **exp2f + 折叠 log2e**；④ `-INFINITY` 空行哨兵 + `sum>0?1/sum:0` 保护。
- **不可借鉴进独立 softmax**：online (m,l) **重标定递推**、`accO` 输出校正、`applyMask`/causal、`convertAccRowCol`、KV-tile 循环——全是**注意力融合机制**，独立算子整行驻留后只需“一趟 max + 一趟 exp/sum”，不需要这些。

### flash-attention（`ops/triton/layer_norm.py`、`cross_entropy.py`）
- FA 的注意力 softmax 也是融合的；**独立行归约的更好参照是它的 layer_norm 与 cross_entropy**（同为“逐行 reduce”，结构 = softmax）。
- **核心技法 = 单次 DRAM 读**（`_layer_norm_fwd_1pass_kernel`）：`program_id` 一行，`tl.load` 把**整行一次读进 SRAM/寄存器**，mean/var（≈ softmax 的 max/sum）两趟归约**都复用这份驻留副本、不回读 DRAM**，最后 `tl.store` 一次。→ 正是 sc_06 的思路。
- **行是否放得下的门控**：`MAX_FUSED_SIZE = 65536 // element_size; BLOCK_N = min(MAX_FUSED_SIZE, next_pow2(N))`——放不下就换路。sc_06 用 `N4 > kBlock*kVptMax 则 fallback` 表达同一门控（本仓库 `kVptMax=16` → N ≤ 256×4×16 = 16384，正好覆盖到长行夺冠的那一档）。
- **cross_entropy 的流式技巧**：vocab（N）超大放不下时，分块流式 online 归约——即“放不下就退回 online”，与 §2(B) 的超大-N 分支一致。
- **CUDA 需手写、Triton 自动的部分**：Triton 隐藏了 smem 落位、向量化、`num_warps`/`BLOCK_N` autotune；CUDA 版要自己写 `float4` 向量化、显式 block-reduce、（可选）`cp.async` 预取。收益换的是对布局/占用的完全掌控。

## 4. 结论 / 教学点：online 的真正归宿是 FA，不是独立 softmax

一个反直觉但重要的收敛：**对独立 softmax，online 递推在 H20 上更是“没必要”**——8192 上 60 MB L2 让它与 sc_04 打平（都事实上单次读 DRAM），长行上 sc_06 的显式单次读又把它甩开。online 的 4·MN→3·MN 收益，要么被 L2 提前吃掉、要么被 caching 的 3·MN→2·MN 直接超越。

那 online 什么时候**不可或缺**？当行**放不下片上、且必须边流式边出结果**时——这正是 **FlashAttention**：S=QK^T 是 [Sq,Skv]，长序列下永远物化不下，只能一块块 KV 流式，用 (m,l) 递推 + O 累加器重标定“带着未归一化的值裸奔、最后除一次”。所以：

> **softmax 阶梯里 sc_05 online 的价值不在“独立 softmax 更快”（H20 上它与 sc_04 打平、并被 sc_06 长行反超），而在于它就是 FA 内层那把钥匙**——把它放在 softmax 这一课学会，到 FA 就水到渠成。sc_06 则告诉你：**独立算子要贴屋顶线，先想“能不能只读一次”——并且要在 L2 遮不住的规模上，这份优势才看得见。**

## 附：本机实测（H20 sm_90a，锁频 1980 MHz，fp32）

**有效带宽（8192² 与 16384²，% 为对 4000 GB/s 屋顶的达成率）：**

| kernel | @8192² GB/s | %屋顶 | @16384² GB/s | %屋顶 |
| --- | ---: | ---: | ---: | ---: |
| sc_01 naive | 102.9 | 2.6% | — | — |
| sc_02 block_reduce | 2491 | 62.3% | 1667 | 41.7% |
| sc_03 warp_shuffle | 1663 | 41.6% | — | — |
| sc_04 vectorized | 3163 | 79.1% | 2442 | 61.1% |
| sc_05 online | 3075 | 77.0% | 2451 | 61.3% |
| **sc_06 resident** | 2575 | 64.4% | **3557** | **88.9%（长行夺冠）** |

**ncu `--set full` Speed-of-Light（默认 8192² 口径）：**

| kernel | DRAM Throughput | Achieved Occupancy |
| --- | ---: | ---: |
| sc_02 block_reduce | 61.6% | 96% |
| sc_04 vectorized | 77.4% | 71% |
| sc_05 online | 83.1% | 90% |
| sc_06 resident | 88.0% | 24% |

**正确性**：sc_01…sc_06 全部对 CPU double 参考 **PASS**（max_rel ≈ 2e-6）。

> 读表要点：(1) 好 kernel 在 4 TB/s 上到 **77–89% 屋顶**，naive 因非合并/延迟只有 2.6%。(2) 8192 上 sc_04≈sc_05≈sc_06——L2 把重复读藏住，谁都事实上“单次读 DRAM”。(3) 拉到 16384，L2 藏不住，只有 sc_06 的**显式**单次读稳在 88.9%，其余全部回落。(4) sc_06 以 24% 占用打出 88.0% DRAM——**占用率不是这些访存受限核的限制**。
