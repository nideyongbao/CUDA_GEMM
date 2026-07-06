# 超越 online：手写 softmax 还能怎么再快？（含 sc_06 实测 + 参考 TinyFA / flash-attention）

> 起点：`sc_05 online` = 1211 GB/s（59.4% 屋顶），PyTorch 库基线 = 1628 GB/s（79.9%）。手写差 26%。本文诊断差距、给出可提升的方向、并实现验证最关键的一条（`sc_06`，反超库基线）。

## 1. 诊断：差距 100% 来自 DRAM 访问遍数，不是算力

ncu 铁证：`sc_05` 的 **DRAM Throughput 已达 87.2%**、Compute(SM) 仅 26%。softmax 是**访存受限**——它早已把 DRAM 用满，SFU/exp 根本不是瓶颈。所以"更快"只有一个杠杆：**减少对 x 的 DRAM 遍数**。

一行 softmax 的 DRAM 流量账（M·N 个元素、每个 4B）：

| kernel | max 趟 | sum 趟 | 写 趟 | x 读次数 | 总流量 | 实测 GB/s |
| --- | :-: | :-: | :-: | :-: | :-: | ---: |
| sc_02/03/04 | 读 x | 读 x | 读 x + 写 y | **3** | 4·MN | 908 / 834 / 1034 |
| sc_05 online | ——（max+sum 融合一趟读）—— | 读 x + 写 y | **2** | 3·MN | 1211 |
| **sc_06 resident** | ——（整行缓存进寄存器，只读一次）—— | 写 y | **1** | **2·MN=理想** | **1708** |

理想下限 = 读 x 一次 + 写 y 一次 = 2·MN。sc_06 达到它 → 屋顶线级。

## 2. 可提升方向（按收益排序）

### ★ (A) 整行缓存、单次 DRAM 读 —— 已实现 = `sc_06`
第一次读 x 就把该行**留在寄存器**（每线程 `float4 reg[]`），之后 max→sum→写全在片上算，**x 只读一次**。3·MN→2·MN。**实测 1708 GB/s = 83.8% 屋顶 = 105% of PyTorch**（反超库基线）。ncu：DRAM 85.2% / 1.74 TB/s、96 寄存器/线程、占用 24%（访存受限，低占用无妨）。**这是最大的一条。**

### (B) 按 N 分派（dispatch-by-N）—— sc_06 的短板与生产库的做法
`sc_06`（block-per-row，256 线程）只在**大 N** 赢。实测 **N=2048 时 sc_06 反而慢**（1106 vs sc_05 1618 GB/s）：行太短时，整块的 block-reduce（2× `__syncthreads` + smem）与启动开销盖过收益。生产库（cuDNN / PyTorch / **OneFlow** 的经典三分派）按 N 选实现：
- **小 N（≲1024）**：**warp-per-row**——一个 warp 一行，元素进寄存器，纯 `__shfl` 归约、无 block sync；
- **中 N（能进寄存器/smem）**：本 `sc_06` register-resident 单读；
- **超大 N（放不下片上）**：退回 **online 流式**（`sc_05`）——这才是 online 的真正用武之地（见 §4）。
> 下一个 rung `sc_07` 就是 warp-per-row，补上小 N 段。

### (C) fp16/bf16 I/O + fp32 归约 —— 直接砍一半流量
本教程 softmax 走 fp32。真实场景（attention 的 mask/bias、logits）多为 fp16/bf16：**读写用半精度、归约上 f4 fp32** → DRAM 流量减半 → 再 ~2×。这正是 flash-attention `layer_norm` 的写法（`.to(tl.float32)` 归约、half 存回）。

### (D) exp2f + 折叠 log2e —— 已在 sc_06 采用（借鉴 TinyFA）
把任何外部 scale 与 log2(e) **预折成一个常数**，用硬件 `exp2f` 代替 `expf`：`exp2f((x-m)·log2e)`。DRAM 受限时收益小，但是零成本好习惯、且与 FA 内层口径一致。

### (E) 占用率/尾效应微调
sc_06 占用 24%（寄存器 96/线程所限）；对访存受限核，只要够喂满 DRAM 即可。可试 `blockDim=128`、或每线程更少 float4 提占用；M 不是 SM 整数倍时的尾块可用 grid-stride 缓解。属最后 5% 的调参。

## 3. 能参考 TinyFA / flash-attention 的写法吗？——能，但要分清"融合 vs 独立"

**两者的 softmax 都是 FUSED（融合在别的 kernel 里），不是独立算子**——这是关键区别，决定了哪些能抄、哪些不能。

### TinyFA（`mma/softmax.cuh` / `fma/softmax.cuh`）
- **本质**：softmax 作用在 **QK^T 刚算出、已驻留寄存器**的 score 分片上，**全程不碰 DRAM**。它的 online (m,l) 递推、`accO*=scale` 输出重标定，都是**为了跨 KV 分块流式 + 融合 P·V** 才存在的。
- **可借鉴（sc_06 已采用）**：① `warpReduce{Max,Sum}` 的 `__shfl_xor` 蝶形归约（宽度按你的"线程→元素"布局选，TinyFA 的宽度-4 是其 MMA 分片布局的产物，别照搬 4）；② 先在寄存器里累局部 partial、再一次 warp 归约的**两段式**；③ **exp2f + 折叠 log2e**；④ `-INFINITY` 空行哨兵 + `sum>0?1/sum:0` 保护。
- **不可借鉴进独立 softmax**：online (m,l) **重标定递推**、`accO` 输出校正、`applyMask`/causal、`convertAccRowCol`、KV-tile 循环——全是**注意力融合机制**，独立算子整行驻留后只需"一趟 max + 一趟 exp/sum"，不需要这些。

### flash-attention（`ops/triton/layer_norm.py`、`cross_entropy.py`）
- FA 的注意力 softmax 也是融合的；**独立行归约的更好参照是它的 layer_norm 与 cross_entropy**（同为"逐行 reduce"，结构 = softmax）。
- **核心技法 = 单次 DRAM 读**（`_layer_norm_fwd_1pass_kernel`）：`program_id` 一行，`tl.load` 把**整行一次读进 SRAM/寄存器**，mean/var（≈ softmax 的 max/sum）两趟归约**都复用这份驻留副本、不回读 DRAM**，最后 `tl.store` 一次。→ 正是 sc_06 的思路。
- **行是否放得下的门控**：`MAX_FUSED_SIZE = 65536 // element_size; BLOCK_N = min(MAX_FUSED_SIZE, next_pow2(N))`——放不下就换路。sc_06 用 `N4 > kBlock*kVptMax 则 fallback` 表达同一门控。
- **cross_entropy 的流式技巧**：vocab（N）超大放不下时，分块流式 online 归约——即"放不下就退回 online"，与 §2(B) 的超大-N 分支一致。
- **CUDA 需手写、Triton 自动的部分**：Triton 隐藏了 smem 落位、向量化、`num_warps`/`BLOCK_N` autotune；CUDA 版要自己写 `float4` 向量化、显式 block-reduce、（可选）`cp.async` 预取。收益换的是对布局/占用的完全掌控。

## 4. 结论 / 教学点：online 的真正归宿是 FA，不是独立 softmax

一个反直觉但重要的收敛：**对独立 softmax，一旦整行能驻留片上（sc_06），online 递推就"没必要"了**——你手里有整行，普通"一趟 max + 一趟 exp/sum"即可，DRAM 都是 1 读。online 的 4·MN→3·MN 收益，被 caching 的 3·MN→2·MN 直接超越。

那 online 什么时候**不可或缺**？当行**放不下片上、且必须边流式边出结果**时——这正是 **FlashAttention**：S=QK^T 是 [Sq,Skv]，长序列下永远物化不下，只能一块块 KV 流式，用 (m,l) 递推 + O 累加器重标定"带着未归一化的值裸奔、最后除一次"。所以：

> **softmax 阶梯里 sc_05 online 的价值不在"独立 softmax 更快"（sc_06 已反超），而在于它就是 FA 内层那把钥匙**——把它放在 softmax 这一课学会，到 FA 就水到渠成。sc_06 则告诉你：**独立算子要贴屋顶线，先想"能不能只读一次"**。

## 附：本机实测（A800 sm_80，锁频 1410MHz，8192² fp32）

| kernel | 有效带宽 GB/s | %屋顶 | %torch |
| --- | ---: | ---: | ---: |
| sc_01 naive | 77 | 3.8% | 4.7% |
| sc_02 block_reduce | 908 | 44.5% | 55.8% |
| sc_03 warp_shuffle | 834 | 40.9% | 51.2% |
| sc_04 vectorized | 1034 | 50.7% | 63.5% |
| sc_05 online | 1211 | 59.4% | 74.4% |
| **sc_06 resident** | **1708** | **83.8%** | **105%** |
| PyTorch `torch.softmax`（库基线） | 1628 | 79.9% | 100% |
