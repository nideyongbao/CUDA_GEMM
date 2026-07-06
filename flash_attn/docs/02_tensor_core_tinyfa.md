# 02 · Tensor-Core 终态：TinyFA CuTe 前向内核

> 前置：`00_principle.md`（算法）、`01_cuda_core_scaffold.md`（fp32 语义参照）。
> 本文对象：`flash_attn/vendor/tfa/` 下 vendor 进来的 **TinyFA**（作者 keith@robot9.me）CuTe 前向内核。
> 它是本仓库 FA 算子的**性能终态**：把 `fa_cc_02` 的 FA2 语义原封不动搬到 Tensor Core，
> 在 A100 上打到 Dao FlashAttention-2 的 **~94–96%**。
>
> 关键文件（本文引用行号均指这些文件）：
> - `vendor/tfa/mma/kernel.cuh` —— 主内核 `flashAttnMma`，FA2 主循环骨架
> - `vendor/tfa/mma/gemm.cuh` —— 两个 GEMM（QKᵀ、PV）+ MMA/LDSM 配置
> - `vendor/tfa/mma/softmax.cuh` —— fragment 级 online softmax
> - `vendor/tfa/mma/layout.cuh` / `memory.cuh` —— swizzle smem 布局、cp.async 搬运
> - `vendor/tfa/config.cuh` / `utils.cuh` —— `KernelConfig`（kBr/kBc/kHeadDim/kNumWarps）、warp 归约、AttentionScale

## 定位：算法没变，换了执行引擎

把 `01` 的对照表再压一句话：**FA2 的数据流一个字都没改，改的全是“矩阵乘和搬运用什么硬件做”。**

| `fa_cc_02`（CUDA-core） | TinyFA（Tensor-core） |
|---|---|
| `for d: s += qreg[d]*sK[...]` | `S = Q@Kᵀ` 走 `mma.sync m16n8k16`（`gemm.cuh::computeScore`） |
| `for d: acc += p*sV[...]` | `O += P@V` 走同一条 `mma.sync`（`gemm.cuh::computeOutput`） |
| `sK[idx]=K[...]` 协作搬 smem | `cp.async` 异步搬 + `ldmatrix` 喂 fragment（`memory.cuh` / `layout.cuh`） |
| 标量 `m/l` + `__expf` 递推 | fragment 级 `rowMax/rowSum` + **warp-4 shuffle 归约** + `exp2f`（`softmax.cuh`） |
| `loopEnd` 块级 causal 剪枝 | KV 反向循环 + **masked / unmasked 两段**（`kernel.cuh`） |

下面按“主循环骨架 → 两个 GEMM → fragment softmax → cp.async/swizzle → causal 反向+分段 → epilogue”拆。

---

## 1. 主循环骨架（`kernel.cuh::flashAttnMma`）

模板参数 `Config`（`config.cuh::KernelConfig`）固化 `kBr / kBc / kHeadDim / kNumWarps`；`kIsCausal` 编译期分支。**一个 block 负责一个 `[b, h, Q-tile]`**（`BlockInfo`），这与 `fa_cc_02` 的 grid 语义一致——Q 在外。

**SRAM 常驻状态**（对应 `fa_cc_02` 的 `acc/m/l`）：
```cpp
auto accO = partition_fragment_C(tiledMma, Shape<Int<kBr>, Int<kHeadDim>>{});  // 未归一化输出 Õ，累加器 fragment
clear(accO);                                                                    // kernel.cuh:100-101
...
constexpr int kNRows = 2 * size<1>(accO);
Softmax<Config, kNRows> softmax;  softmax.init();                               // running (m,l) 在寄存器  kernel.cuh:123-125
```
`accO` 是 Tensor Core 的 C 累加器 fragment，**全程驻留寄存器**，等价于 `fa_cc_02` 里那个钉死不写回的 `acc[]`。`kNRows` 是本 warp 每个线程负责的行数（见 §3 布局）。

**主循环形态**（`kernel.cuh:145-194`）：一个 `tileIteration` lambda，对每个 KV 块依次做
`load V(async) → S=QKᵀ → (mask) → softmax.update → O+=PV → load 下一个 K(async)`，
外面由两个 for 驱动（先 masked 段再 unmasked 段，见 §5）。循环体末尾 `softmax.finalize(accO)` 做 epilogue。

---

## 2. 两个 GEMM：QKᵀ 与 PV（`gemm.cuh`）

FA 的算力全在这两个矩阵乘上，二者共用同一个 `TiledMma`：

```cpp
// gemm.cuh:21-31  —— SM80 原生 MMA atom + LDSM copy atom
using MmaAtomArch = MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>;   // fp16 输入 / f32 累加；bf16 走 F32BF16BF16F32
using SmemCopyAtom          = Copy_Atom<SM75_U32x4_LDSM_N, ElemType>;  // ldmatrix.x4，非转置
using SmemCopyAtomTransposed= Copy_Atom<SM75_U16x8_LDSM_T, ElemType>;  // ldmatrix.x4，转置（喂 Vᵀ）
using TiledMma = TiledMMA<MmaAtomArch, Layout<Shape<Int<kNumWarps>,_1,_1>>, Tile<Int<16*kNumWarps>,_16,_16>>;
```

- `SM80_16x8x16_F32F16F16F32_TN` = Ampere 的 `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`，一条 warp 级指令算 16×8×16。
- **两个 GEMM 都靠 `ldmatrix` 把 smem 直接喂进 fragment**（`SM75_U32x4_LDSM_N`），省掉手工 smem→reg 的 shuffle。

### GEMM-1：`S = Q @ Kᵀ`（`computeScore`, gemm.cuh:38-55）
```cpp
copy(smemTiledCopyQ, tSsQ(_,_,_0{}), tSrQCopy(_,_,_0{}));   // ldmatrix 预取第 0 个 K-slice
copy(smemTiledCopyK, tSsK(_,_,_0{}), tSrKCopy(_,_,_0{}));
for (int ki = 0; ki < size<2>(tSrQ); ki++) {
  if (ki < ...-1) { copy(...ki+1...); }                     // 沿 headDim 软件流水：算 ki 时预取 ki+1
  gemm(tiledMma, tSrQ(_,_,ki), tSrK(_,_,ki), accS);         // mma.sync 累加到 accS
}
```
输出 `accS = Q·Kᵀ`（`[kBr, kBc]` fragment），就是 `fa_cc_02` 里那个标量 `s`，只是现在整块一次算出来。

### GEMM-2：`O += P @ V`（`computeOutput`, gemm.cuh:60-76）
```cpp
auto tOrP = convertAccToP<Config>(accS);   // 把 f32 的 P（softmax 结果）降精到 fp16/bf16，重排成 A-fragment 布局
copy(smemTiledCopyV, tOsVt(_,_,_0{}), tOrVtCopy(_,_,_0{}));    // 用转置 ldmatrix 取 Vᵀ
for (int ki = 0; ki < size<2>(tOrP); ki++) {
  if (ki < ...-1) { copy(...ki+1...); }
  gemm(tiledMma, tOrP(_,_,ki), tOrVt(_,_,ki), accO);           // 累加进常驻的 accO（= Õ）
}
```
`accS`（softmax 后即 `P̃`）直接作为 GEMM-2 的 A 操作数**留在寄存器**送进第二个 MMA——`S` 和 `P` **从不落 HBM，甚至不落 smem**。这就是 `00_principle.md` 讲的“中间矩阵消失”在张量核上的终极形态。V 需要转置（`sVt` / `SmemCopyAtomTransposed`），因为 `P@V` 里 V 当 B 操作数要按列取。

---

## 3. Fragment 级 Online Softmax（`softmax.cuh`）

这是 `sc_05_online` 的“上张量核”版。难点：`fa_cc_02` 里一整行 `s` 在一个线程手上，`rowmax` 是普通 for 循环；而在 MMA 的 C-fragment 布局里，**一行的若干列被摊到多个线程上**，`rowmax/rowsum` 必须跨线程归约。

TinyFA 的关键设计：`SM80_16x8x16` 的输出布局下，**一行的列只分布在 4 个线程（一个 quad）里**，所以归约宽度是 4：
```cpp
// softmax.cuh:141-148  —— warpReduce<4>：只在 4 个 lane 间做 butterfly shuffle
static T warpAllReduceMax(T v){ return warpReduceMax<4>(v); }   // delta 2→1
static T warpAllReduceSum(T v){ return warpReduceSum<4>(v); }
```
`utils.cuh:75-91` 的 `warpReduce<kReduceWidth>` 就是 `__shfl_xor_sync` 蝶形归约，`kReduceWidth=4` 意味着只跨 4 lane——比全 warp 归约便宜得多。这就是任务里说的 **“warp-4 归约”**。

`update(accS, accO, attnScale)`（softmax.cuh:67-111）逐字对应 FA2 递推：
```cpp
// 1) 当前块行内 max（先线程内，再 warp-4 归约）
scoresMaxCur[mi] = ... ; scoresMaxCur[mi] = warpAllReduceMax(scoresMaxCur[mi]);   // = fa_cc_02 的 row_m
// 2) 用新旧 max 差重标定历史（rowSum 和 accO 一起缩）—— 这就是 acc[d]*=c
float scale = (oldMax==-INFINITY)?0.f:exp2f((oldMax-newMax)*attnScale);
rowSum[mi] *= scale;  for(ni) accORowCol(mi,ni) *= scale;                          // = Õ *= e^{m_old-m_new}
rowMax[mi] = newMax;
// 3) 求 P̃ = exp2(s·attnScale - m·attnScale) 并累加 rowSum（延迟归一化：这里不除）
scores(mi,ni) = exp2f(scores(mi,ni)*attnScale - maxScaled);  rowSum[mi] += scores(mi,ni);
```
两个值得记的工程细节：
- **`exp2f` 而非 `expf`**：`AttentionScale`（`utils.cuh:69-72`）把 `1/√d` 和 `log₂e` 合并成一个常数 `attnScale`，于是 `expf(x/√d) = exp2f(x·attnScale)`。硬件 `exp2` 走 SFU 单周期，比 `exp` 快。
- **`update` 里不除 `rowSum`**：完全就是 FA2 延迟归一化。归一化推迟到 `finalize`。

`finalize(accO)`（softmax.cuh:113-132）= epilogue：先把 `rowSum` 做一次 warp-4 归约拿到整行真实分母，再 `accO *= 1/rowSum`。对应 `fa_cc_02` 的 `O[...] = acc[d]/l`，**全程只此一次除法**。

---

## 4. cp.async 单缓冲 overlap + swizzle

### cp.async 异步搬运与计算重叠（`kernel.cuh` + `memory.cuh`）
搬运走 `SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>`（`memory.cuh:34`，16B/线程向量化，绕过寄存器直达 smem）。主循环把“搬下一块”和“算这一块”交叠：
```cpp
// kernel.cuh:149-183（tileIteration 内）
MemLoader::gm2sm(..., tVsV, tVgV, ...); cp_async_fence();   // 异步发起 V[tileIdx] 搬运
GemmOp::computeScore(accS, ...);                            // 同时用 Tensor Core 算 S=QKᵀ
cpAsyncWaitGroup<0>(); __syncthreads();                     // 等 V 到位
if (hasNext) { MemLoader::gm2sm(..., tKsK, ...); cp_async_fence(); }  // 异步发起下一个 K 搬运
softmax.update(accS, accO, kAttnScale);                     // 同时做 softmax（标量单元）
GemmOp::computeOutput<Config>(accO, accS, ...);             // 再算 O+=PV（Tensor Core）
if (hasNext) { cpAsyncWaitGroup<0>(); __syncthreads(); }    // 等下一个 K
```
这是**单缓冲（single-buffer）**流水：K/V 在 smem 里只各留一份，靠 `cp.async` 的异步性把“下一块的 DRAM 延迟”藏在“这一块的 MMA + softmax”后面。相较双缓冲省一半 smem（可换更大 `kBr/kBc` 或更高 occupancy），代价是块间要 `__syncthreads`。启动序（`kernel.cuh:130-139`）先预取 Q、再预取第一个 K，进入循环。

### swizzle 消 bank conflict（`layout.cuh`）
smem 里 Q/K/V 用 CuTe `Swizzle` 布局：
```cpp
// layout.cuh:43-50
static constexpr int kSwizzle = kBlockKSmem == 32 ? 2 : 3;
using SmemLayoutSwizzle = Swizzle<kSwizzle, 3, 3>;
using SmemLayoutAtom = composition(SmemLayoutSwizzle{}, SmemLayoutAtomNoSwizzle{});
```
`ldmatrix` 一次让 warp 内 32 线程各取 smem 一行，若行按 128B 朴素排布，多个线程会撞进同一 bank（32-way conflict）。`Swizzle<B,M,S>` 按异或规则打乱列地址，使 `ldmatrix` 的 32 个访问散到 32 个 bank——**零 bank conflict**。仅在 `kHeadDim` 为 2 的幂时启用（`config.cuh:33 kUseSwizzle`）。`SmemLayoutVTransposed`（layout.cuh:55-57）另配一套转置视图给 GEMM-2 的 Vᵀ。

---

## 5. Causal：KV 反向循环 + masked/unmasked 分段

`kernel.cuh` 的 KV 循环是**从后往前**跑的（`kvTileIdx = numTilesKV-1; --kvTileIdx`）。配合 causal 分两段：
```cpp
// kernel.cuh:141-194
const int nMaskingSteps = Softmax::numMaskingSteps<kIsCausal>(blockInfo);
const int nNoMaskEnd = numTilesKV - nMaskingSteps;
for (; kvTileIdx >= nNoMaskEnd; --kvTileIdx)   tileIteration(kvTileIdx, true_tag{});   // 需要 mask 的块（对角线附近）
for (; kvTileIdx >= 0;         --kvTileIdx)   tileIteration(kvTileIdx, false_tag{});  // 完全在下三角内，免 mask
```
思路：causal 掩码只在**对角线所在的少数 KV 块**上有“半掩”边界，其余块要么全保留、要么全丢弃。`numMaskingSteps`（softmax.cuh:27-39）算出只有 `⌈kBr/kBc⌉(+对齐修正)` 个块需要逐元素判掩码；剩下的块走 `kNeedsMask=false` 的编译期分支，**完全不执行 mask 代码**。`applyMask`（softmax.cuh:41-65）用 identity tensor 还原每个 fragment 元素的 `(row, col)` 全局坐标，`col > row` 的置 `-INF`。

这是 `fa_cc_02` 里 `loopEnd` 块级剪枝 + `j>jmax break` 行级掩码的**编译期特化升级版**：把“绝大多数块无需掩码”这件事变成两个不同 template 实例，把分支从热循环里彻底删掉。

---

## 6. Epilogue 与写回

`softmax.finalize(accO)`（§3）归一化后，`MemStore::storeO`（`kernel.cuh:200`）把 `accO` fragment 经 smem 中转、向量化写回 HBM 的 `gO`。**O 只写一次**——FA2 的 IO 承诺兑现。

---

## 7. 用 ncu 看什么（性能验收）

这支内核的目标是 ~94–96% Dao FA2，profiling 时盯这几类计数器：

| 关注点 | ncu 指标（section / metric） | 期望 |
|---|---|---|
| **Tensor 管线打满** | `sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active`（Tensor Active %） | 越高越好；两个 GEMM 应主导 |
| **cp.async / 访存重叠** | `SpeedOfLight` 里 Compute vs Memory throughput；`gpu__time_duration` 与 `l1tex__data_pipe_lsu_wavefronts` | Compute-bound 而非 Memory-bound |
| **L1/TEX 命中** | `l1tex__t_sector_hit_rate`、`l1tex__throughput` | cp.async 走 L1/TEX，命中率高说明 KV 复用好 |
| **bank conflict** | `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld/st.sum` | swizzle 生效应 ≈ 0 |
| **occupancy** | `sm__warps_active.avg.pct_of_peak_sustained_active` | 单缓冲省 smem 是否换到了足够 occupancy |
| **SFU 压力** | `sm__inst_executed_pipe_xu`（exp2/rsqrt 等） | 延迟归一化后应远小于 Tensor 周期 |

排障直觉：Tensor Active % 上不去 → 多半卡在 softmax 的标量/归约或 cp.async 没盖住延迟；shared bank conflict 非 0 → 检查 `kUseSwizzle` 是否因 `kHeadDim` 非 2 的幂而被关掉。

> 更完整的 A800 上 profiling 方法论与调优扫描思路，可参照同仓库 `gemm/docs/01_性能分析方法论.md` 与 `gemm/docs/a800`。

---

## 8. 小结

TinyFA = 把 `fa_cc_02` 的 **FA2 语义**（Q 外 / KV 内、online 递推、延迟归一化、causal 剪枝）
逐条映射到 Ampere 原生原语：
**两个 `mma.sync m16n8k16` GEMM + `ldmatrix` 喂料 + `cp.async` 单缓冲 overlap + swizzle smem + fragment 级 warp-4 online softmax + KV 反向 masked/unmasked 分段**。
算法一个字没改，只是把 `for d` 标量 FMA 换成了张量核、把协作 smem 搬运换成了异步向量搬运。
这就是它能吃到 ~94–96% Dao FA2 的原因，也是下一篇要点破的主线：**FA 几乎没有新原语，它只是把已有原语“融合”起来。**
