# 逐档深度分析：sc_01 → sc_05

统一约定：默认 `M=N=8192`，度量口径 `ideal_bytes = 2*M*N*4`，`eff_BW = ideal_bytes / time`，roofline = 4000 GB/s（H20 HBM3，见 `bench.cu`）。本机锁频 1980 MHz。
下面每一档给出：**(a) 唯一改动**、**(b) 访存/规约机制与关键代码 idiom**、**(c) 该看哪些 ncu counter**、**(d) H20 实测结论**。

> 全局提醒（H20 的大 L2）：H20 有 60 MB L2，会把每个 block 在**同一行内的重复读就地命中**，因此“读 3 次”的多趟 kernel 实际 DRAM 流量往往接近 2·MN、并**超过**按 pass 数算的朴素天花板。这条贯穿 sc_02→sc_05，也决定了 online 的“少读一遍”在 8192 上收益甚微（见 sc_05）。

---

## sc_01 naive — 1 线程 / 行，三趟顺序扫

### (a) 唯一改动
基线。每个线程负责一整行，串行做 max / sum / write 三趟。

### (b) 机制与代码 idiom
```cuda
int r = blockIdx.x * blockDim.x + threadIdx.x;   // 1 线程 = 1 行
const float* xr = x + (size_t)r * N;
for (int j = 0; j < N; ++j) m = fmaxf(m, xr[j]);          // pass 1
for (int j = 0; j < N; ++j) s += __expf(xr[j] - m);       // pass 2
for (int j = 0; j < N; ++j) yr[j] = __expf(xr[j] - m)*inv;// pass 3
```
**致命点**：一个 warp 里 32 个线程处理 32 个**相邻的行**。同一时刻它们都在读各自行的 `xr[0]`，地址相隔 `N*4 = 32 KB`。于是一次 warp 访存请求落在 **32 个不同的 32B sector** 上，每个 sector 只有 4 字节有用 → 有效率 `4/32 = 12.5%`。这就是 stride-N 的**完全非合并**。加之三趟串行，warp 长期卡在访存延迟上。

### (c) 要读的 ncu counter
- `l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio`：预期 ≈ **32 sectors/request**（合并时应为 4）。
- 全局加载效率 `gld_efficiency`（旧口径）≈ **12.5%**。
- `dram__throughput.avg.pct_of_peak_sustained_elapsed`：**很低**（个位数），因为大量搬来的字节是浪费的、且延迟掩盖不住。
- warp stall 原因 `smsp__pcsamp_warps_issue_stalled_long_scoreboard`（内存记分板等待）：**占主导**。
- 达成占用率 `sm__warps_active.avg.pct_of_peak_sustained_active`：即便占用率不低，也救不回来（延迟受限）。

### (d) H20 实测结论
`eff_BW = **102.9 GB/s（2.6% 屋顶）**`。这是“正确但最慢”的基线，唯一价值是确立 safe-softmax 的正确性和标尺。**下一档必须先解决非合并。**

---

## sc_02 block-reduce — 1 block / 行，coalesced + shared-mem 树规约

### (a) 唯一改动
把“1 线程 / 行”换成“1 block / 行”；行内元素由整块线程 **stride 遍历**；行 max / 行 sum 用 shared-mem 树形规约。

### (b) 机制与代码 idiom
```cuda
int r = blockIdx.x;                       // 1 block = 1 行
for (int j = t; j < N; j += nt) m = fmaxf(m, xr[j]);   // stride = blockDim，相邻线程读相邻元素
red[t] = m; __syncthreads();
for (int s = nt>>1; s>0; s>>=1) {         // 树形规约
  if (t < s) red[t] = fmaxf(red[t], red[t+s]);
  __syncthreads();
}
```
**关键**：`j += nt` 的 stride 遍历让 warp 内相邻 `lane` 读**相邻地址**（`xr[t], xr[t+1], ...`）→ 一次请求 32 个连续 float = 128 字节 = 4 个 sector，**完全合并**。规约走 shared memory，需要 `log2(256)=8` 轮 + 每轮一个 `__syncthreads()`。

**瓶颈转移**：`x` 名义上被读 3 次（max、sum、write），但从这一档起，kernel 从“延迟受限”转为“**流量受限**”——在 H20 上，重复读大多被 60 MB L2 藏住。

### (c) 要读的 ncu counter
- `sectors_per_request` ≈ **4**（合并达成），`gld_efficiency` ≈ **100%**。
- `dram__throughput.avg.pct_of_peak_sustained_elapsed`：**大幅上升**（本机实测 **61.6%**）。
- stall 原因从 `long_scoreboard` 部分转移到 `stall_barrier`（`__syncthreads`）和 `stall_mio_throttle`（shared-mem/特殊函数单元压力）。
- 达成占用率 `sm__warps_active`：本机实测 **96%**（block-per-row 有充裕并行度）。

### (d) H20 实测结论
`eff_BW = **2491 GB/s（62.3% 屋顶）**`，ncu：**DRAM 61.6% / 占用 96%**。从个位数 % 一跃到 62%，这是整条阶梯**收益最大**的一步（修合并）。此后 sc_03/04 都在这条“合并已达成、瓶颈在 DRAM”的线上做精修。

---

## sc_03 warp-shuffle — 寄存器内规约，去掉 smem + barrier

### (a) 唯一改动
规约从 shared-mem 树换成 `__shfl_xor_sync` 蝶形规约；每行映射到**一个 warp**（32 线程 stride-32 遍历），无 shared memory、无 `__syncthreads`。

### (b) 机制与代码 idiom
```cuda
__device__ float warpMax(float v){
  for (int o=16;o>0;o>>=1) v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, o));
  return v;                       // 5 步 butterfly，纯寄存器
}
int warp = (blockIdx.x*blockDim.x + threadIdx.x) >> 5;   // 1 warp = 1 行
int lane = threadIdx.x & 31;
for (int j=lane;j<N;j+=32) m = fmaxf(m, xr[j]);          // stride-32，仍合并
m = warpMax(m);
```
**关键**：`__shfl_xor` 在寄存器之间直接交换，5 步完成 32-lane 规约，**没有 shared-mem 往返、没有块级 barrier**。相邻 lane 读相邻元素（stride-32）→ 仍然合并。

**代价（H20 上放大）**：一行只剩 **32 个线程**在处理。与 sc_02 的 256 线程/行相比，每行暴露的**内存级并行（MLP）少了 8×**。在 H20 的 4 TB/s HBM3 上，喂满带宽需要大量在途访存请求；warp-per-row 的 32-way 并行**喂不满**。而此时瓶颈本就在 DRAM（不在规约），去掉 barrier 省下的开销**买不回**丢失的带宽。

### (c) 要读的 ncu counter
- `stall_barrier` 应**明显下降**（不再有 `__syncthreads`）。
- shared-mem 相关事务 `l1tex__data_pipe_lsu_wavefronts_mem_shared*` 大幅减少乃至为 0。
- `dram__throughput`：**低于** sc_02——不是因为流量变多，而是 32-way MLP 撑不起 4 TB/s。这是判断“瓶颈已在 DRAM 且受并行度制约”的关键证据。
- 占用率 `sm__warps_active`：warp-per-row 映射下每行并行度低，关注是否欠喂内存系统。

### (d) H20 实测结论
`eff_BW = **1663 GB/s（41.6% 屋顶）**`，**低于 sc_02 的 62.3%**。这与“去掉同步一定更快”的直觉相反：因为瓶颈已经是 DRAM 带宽、而非规约本身，warp-per-row 反而因**每行并行度不足**喂不满 4 TB/s HBM3。教学意义：**规约已不是瓶颈，真正的墙是带宽/并行度**——warp-per-row 是“小 N”的正确工具（见 `04`），不是 N=8192 的。

---

## sc_04 vectorized — float4 128-bit 访存

### (a) 唯一改动
在 sc_02 的 block-per-row 结构上，把每次 `float` 读写换成 **`float4`（128-bit）**。规约仍是 shared-mem 树。

### (b) 机制与代码 idiom
```cuda
const float4* xr = reinterpret_cast<const float4*>(x + (size_t)r*N);
int N4 = N >> 2;
for (int j=t;j<N4;j+=nt){
  float4 v = xr[j];                                     // 一条指令搬 16 字节
  m = fmaxf(m, fmaxf(fmaxf(v.x,v.y), fmaxf(v.z,v.w)));
}
```
**关键**：每个线程每次访存搬 16 字节，warp 一次请求 `32×16 = 512` 字节 = 16 个 sector。请求**数量减少 4 倍**，每请求的有效载荷更大，访存指令 issue 开销更低，内存级并行（MLP）更饱满 → 更接近“每请求峰值带宽”。要求 `N % 4 == 0`，否则回退到 sc_02。

**流量真相**：名义仍读 3 次 x，但 H20 的 60 MB L2 把后两次“重复读”**就地命中**，实际 DRAM 流量已经**≈2·MN**（几乎就是理想）。因此它**跑赢了朴素的“50% (k=4)”天花板**。

### (c) 要读的 ncu counter
- 全局 load/store **请求数** `l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum`：应降到 sc_02 的约 **1/4**。
- `dram__throughput.avg.pct_of_peak_sustained_elapsed`：本机实测 **77.4%**（8192 上全阶梯最高之一）。
- `stall_lg_throttle` / `stall_mio_throttle`：观察访存管线是否更顺畅。
- 占用率 `sm__warps_active`：本机实测 **71%**（float4 每线程寄存器更多、占用略降，但**不影响喂满 DRAM**）。

### (d) H20 实测结论
`eff_BW = **3163 GB/s（79.1% 屋顶）**`，ncu：**DRAM 77.4% / 占用 71%**。这是 8192 上**最快**的一档：宽事务 + L2 藏住重复读，让实际 DRAM 流量逼近 2·MN。到此“把带宽喂饱”已接近极限；**要再快，只能从算法上真正少读一遍 x**（sc_05/sc_06）。

---

## sc_05 online — 融合 max/sum，两趟读 x ⭐（FlashAttention 桥梁）

### (a) 唯一改动
把 max pass 和 sum pass **融合成一趟流式扫描**：每个线程维护 running `(m, l)`，边扫边更新；遇到更大的 max 就把已累积的 `l` 按 `exp(m_old - m_new)` **rescale**。于是 x 只被读 **2 次**（融合统计 + 写出）而非 3 次。

### (b) 机制与代码 idiom
```cuda
// online 合并：把 (m2,l2) 折进 running (m,l)
__device__ void mergeMl(float& m, float& l, float m2, float l2){
  float mn = fmaxf(m, m2);
  l = l*__expf(m - mn) + l2*__expf(m2 - mn);   // 关键：rescale 已累积的 l
  m = mn;
}
// pass 1（融合）：一趟扫描同时得到 max 和 sum
float m=-INFINITY, l=0.f;
for (int j=t;j<N4;j+=nt){
  float4 v = xr[j];
  mergeMl(m,l,v.x,1.f); mergeMl(m,l,v.y,1.f);
  mergeMl(m,l,v.z,1.f); mergeMl(m,l,v.w,1.f);
}
// 块内规约也用同一个 mergeMl 合并 per-thread (m,l)
sm[t]=m; sl[t]=l; __syncthreads();
for (int s=nt>>1;s>0;s>>=1){
  if (t<s) mergeMl(sm[t],sl[t], sm[t+s],sl[t+s]);
  __syncthreads();
}
// pass 2：normalize + write（第 2 次读 x）
```
**关键洞察**：普通 softmax 之所以要两趟，是因为“`exp(x-m)` 里的 `m` 要先知道全局 max”。online 的技巧是——**先用当前局部 max 累加，等 max 变大时把旧的和乘上一个修正因子** `exp(m_old-m_new)`，代数上等价于一开始就用全局 max 累加。这样 max 和 sum 就能在**同一趟**里同时算出来。规约阶段把 `mergeMl` 用作 combine 算子（它满足结合律），block 内一次树规约合并所有 per-thread partial。

### DRAM 流量账本（本档设计意图）

用 `MN×4` 作为单位（一份 = 读或写整个矩阵一次）：

| kernel | max pass | sum pass | write pass | 读 x 次数 | 写 y 次数 | **名义总流量** |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|
| sc_02 / sc_03 / sc_04 | 读 x | 读 x | 读 x + 写 y | **3** | 1 | **4 份** |
| **sc_05 online** | ——（与 sum 融合）—— | 读 x | 读 x + 写 y | **2** | 1 | **3 份** |

- **名义削减**：`(4 - 3) / 4 = 25%` 更少的名义 DRAM 流量。
- **理想模型（若重复读全部落到 DRAM）**：度量分子固定为 `ideal = 2 份`，若都跑满 HBM：
  ```
  sc_04:  eff_BW = 2 / (4/peak) = peak/2   ≈ 2000 GB/s  (50.0% 达成)
  sc_05:  eff_BW = 2 / (3/peak) = 2peak/3  ≈ 2667 GB/s  (66.7% 达成)
  ```
  理想上相对提升 `(2/3)/(1/2) = 4/3`，即 +33%。
- **H20 实测：这个 +33% 不兑现**。因为 60 MB L2 早已把 sc_04 的“第 3 次读”就地命中——sc_04 的**实际** DRAM 流量本就 ≈2·MN，online 想省的那一份 L2 已经替它省了。于是 **sc_05 ≈ sc_04**：`3075（77.0%）` vs `3163（79.1%）`。两者都**超过**了各自的理想 pass-count 天花板。

### (c) 要读的 ncu counter
- `dram__bytes.sum`：理想上应从 ≈`4*M*N*4` 降到 ≈`3*M*N*4`；但在 H20 上 sc_04 因 L2 命中本就接近 `2*M*N*4`，两者差距不大。
- `dram__throughput.avg.pct_of_peak_sustained_elapsed`：本机实测 **83.1%**（多趟里**最高**——online 的 DRAM 效率确实最好），但 wall-time 与 sc_04 差别很小。
- 占用率 `sm__warps_active`：本机实测 **90%**。
- `__expf`/SFU 压力 `stall_mio_throttle`：online 融合趟每元素多算了 rescale 的 `exp`，需确认没有因此从 memory-bound 掉进 SFU-bound（H20 上 SFU 吞吐足够，仍是 memory-bound）。

### (d) H20 实测结论
`eff_BW = **3075 GB/s（77.0% 屋顶）**`，ncu：**DRAM 83.1% / 占用 90%**，与 sc_04 **基本打平**。长行也一样：`16384` 上 sc_05 = `2451（61.3%）` ≈ sc_04 = `2442（61.1%）`，两者都被 sc_06 的 `3557（88.9%）` 甩开。

结论有两层：(1) 作为**独立 softmax**，online 在 H20 上并不比 sc_04 更快——它想省的读被 L2 提前省掉了；真正“少读一遍”一路走到底、拿到带宽的是 **sc_06 的单次寄存器驻留读**（长行夺冠）。(2) online 的**持久价值**不在这里，而在于 `mergeMl` 这个 `(m,l)` 递推**正是 FlashAttention 的内核**——把 softmax 从“独立算子”推向“可融合进 attention”。详见 `03_conclusion_online_bridge.md` 与 `04_beyond_online.md`。

---

## 阶梯小结（H20 实测，8192²，锁频 1980 MHz）

| 档 | 瓶颈状态转移 | eff_BW（GB/s，% 屋顶） |
|----|-------------|------------------------|
| sc_01 | 非合并 + 延迟受限 | 102.9（2.6%） |
| sc_02 | → 合并达成，转入带宽受限 | 2491（62.3%） |
| sc_03 | 去规约开销，但 warp-per-row 每行 MLP 不足 → **退步** | 1663（41.6%） |
| sc_04 | 宽事务 + L2 藏住重复读 → **8192 最快** | 3163（79.1%） |
| sc_05 | 算法削名义流量，但 L2 已替 sc_04 省掉 → ≈ sc_04 | 3075（77.0%） |

> H20 的两个反直觉点：sc_03 因并行度不足而慢于 sc_02；online 的“少读一遍”被 60 MB L2 吃掉，故 sc_05 ≈ sc_04。真正贴屋顶线的“少读”是 **sc_06 的单次读**，且要在**长行**上兑现（见 `04`）。
