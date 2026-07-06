# 逐档深度分析：sc_01 → sc_05

统一约定：默认 `M=N=8192`，度量口径 `ideal_bytes = 2*M*N*4`，`eff_BW = ideal_bytes / time`，roofline = 2039 GB/s（见 `bench.cu`）。
下面每一档给出：**(a) 唯一改动**、**(b) 访存/规约机制与关键代码 idiom**、**(c) 该看哪些 ncu counter**、**(d) 预期结论**。

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
- `dram__throughput.avg.pct_of_peak_sustained_elapsed`：**很低**（个位数~十几 %），因为大量搬来的字节是浪费的、且延迟掩盖不住。
- warp stall 原因 `smsp__pcsamp_warps_issue_stalled_long_scoreboard`（内存记分板等待）：**占主导**。
- 达成占用率 `sm__warps_active.avg.pct_of_peak_sustained_active`：即便占用率不低，也救不回来（延迟受限）。

### (d) 预期结论
`eff_BW` 停在个位数百分比（几十 GB/s）。这是“正确但最慢”的基线，唯一价值是确立 safe-softmax 的正确性和标尺。**下一档必须先解决非合并。**

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

**仍然的瓶颈**：`x` 依旧被读 3 次（max、sum、write 各扫一遍），总 DRAM 流量 = `3 读 + 1 写 = 4 份 MN×4`。从这一档起，kernel 从“延迟受限”转为“**流量受限**”。

### (c) 要读的 ncu counter
- `sectors_per_request` ≈ **4**（合并达成），`gld_efficiency` ≈ **100%**。
- `dram__throughput.avg.pct_of_peak_sustained_elapsed`：**大幅上升**（几百 GB/s → 40~50% 达成率）。
- stall 原因从 `long_scoreboard` 部分转移到 `stall_barrier`（`__syncthreads`）和 `stall_mio_throttle`（shared-mem/特殊函数单元压力）。
- `dram__bytes.sum`：应约等于 `4*M*N*4`，之后 sc_05 会看到它下降。

### (d) 预期结论
`eff_BW` 从个位数 % 跃升到 **~40~50% 达成率**。这是整条阶梯**收益最大**的一步（修合并）。此后 sc_03/04 都在这条“3 次读”的天花板下做精修。

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

**仍然的瓶颈**：还是读 3 次 x。这一档省的是**规约开销**（smem 带宽 + 同步），不是 DRAM 流量。

### (c) 要读的 ncu counter
- `stall_barrier` 应**明显下降**（不再有 `__syncthreads`）。
- shared-mem 相关事务 `l1tex__data_pipe_lsu_wavefronts_mem_shared*` 大幅减少乃至为 0。
- `dram__throughput` 与 sc_02 **基本持平**（流量没变），这是判断“瓶颈已在 DRAM”的关键证据。
- 占用率 `sm__warps_active`：warp-per-row 映射下需关注是否因寄存器压力/尾部效应变化。

### (d) 预期结论
相对 sc_02 **小幅提升或持平**。因为瓶颈已经是 DRAM 流量而非规约本身，去掉 smem/barrier 的收益被“3 次读”的墙吸收掉了。这一档的教学意义 > 性能意义：它证明“规约已经不是瓶颈了，真正的墙是流量”。

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

**仍然的瓶颈**：读 3 次 x 不变，物理天花板仍是 `peak/2 ≈ 1020 GB/s`。

### (c) 要读的 ncu counter
- 全局 load/store **请求数** `l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum`：应降到 sc_02 的约 **1/4**。
- `dram__throughput.avg.pct_of_peak_sustained_elapsed`：**进一步逼近峰值**（这一档最接近 50% 达成率的物理天花板）。
- `stall_lg_throttle` / `stall_mio_throttle`：观察访存管线是否更顺畅。
- `dram__bytes.sum`：仍 ≈ `4*M*N*4`（宽事务不改变总字节，只改变搬运效率）。

### (d) 预期结论
`eff_BW` 逼近 **~50% 达成率（≈ 1020 GB/s）**，即“3 次读”结构的物理极限。到此为止，`sc_02→sc_04` 已经把带宽榨干；**要再快，只能少读一遍 x**。

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

### DRAM 流量账本（本档收益的核心）

用 `MN×4` 作为单位（一份 = 读或写整个矩阵一次）：

| kernel | max pass | sum pass | write pass | 读 x 次数 | 写 y 次数 | **总流量** |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|
| sc_02 / sc_03 / sc_04 | 读 x | 读 x | 读 x + 写 y | **3** | 1 | **4 份** |
| **sc_05 online** | ——（与 sum 融合）—— | 读 x | 读 x + 写 y | **2** | 1 | **3 份** |

- **流量削减**：`(4 - 3) / 4 = 25%` 更少的总 DRAM 流量。
- **有效带宽为何上升**：度量分子固定为 `ideal = 2 份`。若都跑满 HBM（时间 ∝ 实际流量）：
  ```
  sc_04:  eff_BW = 2 / (4/peak) = peak/2   ≈ 1020 GB/s  (50.0% 达成)
  sc_05:  eff_BW = 2 / (3/peak) = 2peak/3  ≈ 1359 GB/s  (66.7% 达成)
  ```
  相对提升 `(2/3)/(1/2) = 4/3`，即 **有效带宽 +33%**。少搬 25% 字节 → 时间少 25% → `1/(1-0.25) = 1.33×` 吞吐，两种算法一致。

### (c) 要读的 ncu counter
- `dram__bytes.sum`：应从 ≈`4*M*N*4` **降到 ≈`3*M*N*4`**（下降约 25%）—— 这是本档最直接的证据。
- `dram__throughput.avg.pct_of_peak_sustained_elapsed`：仍应接近峰值（依旧带宽受限），但因总字节变少，**wall-time 下降 → `eff_BW` 上升**。
- `__expf`/SFU 压力 `stall_mio_throttle`：online 融合趟里每元素多算了 rescale 的 `exp`，需确认没有因此从 memory-bound 掉进 compute/SFU-bound（在 A800 上 SFU 吞吐足够，通常仍是 memory-bound）。
- 与 sc_04 对比 `gpu__time_duration.sum`：应缩短约 25%。

### (d) 预期结论
`eff_BW` 从 sc_04 的 ~50% 跃到 **~66% 达成率（≈ 1360 GB/s）**。这是整条阶梯上**唯一靠“少搬字节”而非“搬得更快”**取得的提升，也是把 softmax 从“独立算子”推向“可融合进 attention”的关键一步。详见 `03_conclusion_online_bridge.md`。

---

## 阶梯小结

| 档 | 瓶颈状态转移 | eff_BW 预期 |
|----|-------------|------------|
| sc_01 | 非合并 + 延迟受限 | 个位数 % |
| sc_02 | → 合并达成，转入流量受限（3 读） | ~40–50% |
| sc_03 | 去规约开销（瓶颈已在 DRAM） | ≈ sc_02，小幅 |
| sc_04 | 宽事务榨干带宽，撞“3 读”天花板 | ~50%（≈1020 GB/s） |
| sc_05 | **算法削流量**，3 读 → 2 读 | ~66%（≈1360 GB/s） |
