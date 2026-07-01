# A800 补齐 Ampere 原生张量核：mma.sync + ldmatrix + 多级 cp.async（tc_06）

> 背景：手写张量核阶梯原本在 A800（Ampere sm_80）上**止步 WMMA**——`tc_03`（cp.async 流水
> 的 WMMA）锁频 4096³ 只有 **43.1 TFLOPS = 13.8% 的 312T 峰值**，而 cuBLAS BF16 到 85%，差 ~6×。
> 本文用 ncu 证据说明：为什么 WMMA C++ API 到头了，以及绕开它、直接用 PTX 的
> `mma.sync.aligned.m16n8k16` + `ldmatrix` + 多级 `cp.async`（`tc_06`）如何把手写一举拉到
> **150.3 TFLOPS = 48.2% 峰 = 3.49× tc_03**。

---

## 1. 结果（A800-SXM，锁频 @1410MHz，4096³）

| 用例 | 手写指令 | TFLOPS | % of 312T 峰 | vs tc_03 |
| --- | --- | ---: | ---: | ---: |
| tc_03 WMMA_pipe | `nvcuda::wmma` + cp.async | 43.1 | 13.8% | 1.0× |
| **tc_06 MMA_pipe** | **`mma.sync`+`ldmatrix`+cp.async** | **150.3** | **48.2%** | **3.49×** |
| cuBLAS BF16（参考） | 库 | 264.7 | 85% | 6.1× |

一句话：**同样是 cp.async 多级流水、同样的 bf16 输入/f32 累加，只是把"喂张量核"的方式从 WMMA
API 换成底层 `mma.sync`+`ldmatrix`，A800 手写就从 14% 峰跳到 48% 峰。** 瓶颈也从"共享内存 bank
冲突墙"变成了"占用率/延迟墙"（详见 §3）。

---

## 2. 为什么 WMMA 在 Ampere 到头了

`nvcuda::wmma` 是 warp 级 C++ 封装：`load_matrix_sync` 把一块 16×16 从 smem 读进不透明的
`fragment`，`mma_sync` 再算。问题是**这个 load 的 smem 访问模式对使用者不可见、也不可控**——
在 Ampere 上它产生严重的共享内存 bank 冲突，把 L1/SMEM 管线打满，张量核反而饿着。

### ncu 铁证（2048³，`--set full`，见 `baselines/a800/`）

| 指标 | tc_03 WMMA | tc_06 mma.sync | 解读 |
| --- | ---: | ---: | --- |
| Elapsed Cycles | 611,162 | **240,149** | tc_06 少 **2.54×** 周期 |
| **L1/TEX Cache Throughput** | **93.80%** | **44.52%** | tc_03 的 L1/SMEM 管线**打满** |
| Compute (SM) Throughput | 24.78% | 32.40% | tc_03 张量核在**挨饿** |
| Shared-load bank conflict | **6.0-way，4194万次冲突 = 83.1% 的 shared-load 波前** | 未被 ncu 标记 | **这就是 WMMA 的墙** |
| Warp Cycles Per Issued Inst | 27.36 | **12.85** | 停顿减半 |
| Registers / Thread | 66 | 123 | mma.sync 版寄存器多（换来 ILP）|
| Achieved Occupancy | 31.50% | 21.72% | tc_06 占用率反而**更低**却更快 |

> ncu 会串行化并禁用 boost（SM 被降到 ~1.7GHz），故绝对 GFLOPS 以非-ncu 的 bench 为准；
> 但**占用率 / 各管线 throughput% / bank 冲突 / 停顿这些结构性指标有效**（见 `baselines/README.md`）。

**关键反直觉点**：tc_03 的"Memory Throughput 88%"不是 DRAM 带宽墙（DRAM 只有 47 GB/s），而是
**共享内存/L1 管线被 bank 冲突打满**。ncu 直接点名：`load_matrix_sync` 的 shared load 平均
**6 路 bank 冲突**、产生 **4194 万次冲突、占全部 shared-load 波前的 83.1%**。WMMA 用户**无法**从
C++ API 层面消除它——这就是 Ampere 上 WMMA 手写的天花板。

---

## 3. tc_06 做了什么，为什么快

三件事，对应三条 sm_80+ 通用 PTX 指令（`kernels/tensor_core/tc_06_mma_pipe.cu`）：

1. **`mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`**：直接下 warp 级张量核 MMA，
   累加寄存器（每线程 64 个 f32）与调度完全可控，不再经过 WMMA 的不透明 fragment。
2. **`ldmatrix.sync.aligned.m8n8.x4/.x2`**：从 swizzle/padding 后的 smem **一条指令**把 fragment
   喂进寄存器，访问模式对齐张量核所需布局 —— 这是干掉 WMMA 那 6 路 bank 冲突的关键。
   布局上把 B 一次性转成 Bt(N×K)，让 A、Bt 两块 smem tile 都以 K 连续，A/B fragment 走**同一条
   非转置 ldmatrix 路径**（对称、最不易错）。
3. **多级 `cp.async.cg`（16B 向量化）+ K_STAGE=3 环形缓冲**：异步预取后续 K-tile，与当前
   tile 的张量核计算重叠（骨架同 tc_03，被验证过的部分原样复用）。

**效果**（回看 §2 表）：ldmatrix 把 L1/TEX 吞吐从 93.8% 压到 44.5%、bank 冲突不再被标记，
每条指令的停顿周期从 27.4 砍到 12.85，总周期少 2.5×。ncu 对 tc_06 的评语变成
*"Shared/Tensor 是最高利用管线(43.7%)……well-utilized, 不应是瓶颈"* —— 已经**搬离了内存墙**。

---

## 4. tc_06 的新天花板：占用率 / 延迟（为什么是 48% 而不是 70%）

tc_06 到 48% 峰后，瓶颈不再是 bank 冲突，而是**指令级/线程级并行不足以掩盖延迟**：

- 大 warp tile（64×32、64 个 f32 累加寄存器）带来高数据复用与 ILP，是它跑赢高占用率小 tile 的
  原因——扫参实测 16-warp（32×32 tile）、BN256、BK64 都更差。
- 但 64 累加寄存器 + fragment 使**每线程 123 寄存器**，占用率被寄存器限死在 **2 block/SM（~22%）**，
  每调度器只有 **0.44 个 eligible warp**、72.95% 周期"无可发射 warp" → 仍是**延迟受限**。
- 朴素 XOR swizzle 未跑赢 padding（padding LDS=40 已把 8 行错开到不同 bank）；B 用 `ldmatrix.x4`
  批量装载也未跑赢 `x2`（多出的寄存器搬运抵消了指令数收益）。

作为**教学阶梯的"承上启下"一级，tc_06 已把 Ampere 手写从 14% 拉到 48%、并把瓶颈从"访存墙"讲清成
"延迟墙"**。那"延迟墙"能不能再靠寄存器级双缓冲捅破到 60–70%？下一节是一次认真的尝试与实测结论。

---

## 5. 冲 60–70% 的尝试与天花板：寄存器级双缓冲为什么没用

直觉上，既然是"ldmatrix→mma 依赖延迟"受限，那就上**寄存器级 fragment 双缓冲**（预取下一 ks 的
A/B fragment，与当前 mma 重叠）。实测（A800 锁频 4096³，约 25 组配置）结论是：**在纯 CUDA C++ 下这
条路已到头，~48–49% 就是天花板**——并且能逐一排除"常见嫌疑"：

| 尝试 | TFLOPS | %峰 | 结论 |
| --- | ---: | ---: | --- |
| 基线 BM128 BN128 BK32 · W2×4 · 3 级 | 150.3 | 48.2% | 当前 |
| 同上 W2×2（warp tile 64×64, 4 warp） | 152.1 | 48.8% | 并列最佳 |
| **+ 寄存器级 fragment 双缓冲** | 146.4 | 46.9% | **无益，反略降** |
| + 单 barrier 主循环（cp.async 计算期发射） | 151.0 | 48.4% | 无益（非 sync 受限）|
| + 强制 3 block/SM（regcap 72–85，无 spill） | 147.7 | 47.3% | 无益（**非占用率受限**）|
| warp tile 64×64（BN256 W2×4，206 reg→1 block/SM） | 121.7 | 39.0% | 更差（占用率塌）|
| BK64 / 16-warp / naive-swizzle / B 用 x4 | 114–137 | 37–44% | 全更差 |

**为什么双缓冲没用**：`-Xptxas -v` 显示加双缓冲后寄存器 122→124 几乎没变、性能反降——因为
**nvcc 对完全展开的内层循环本来就做了软件流水**（把下一 ks 的 ldmatrix 提前到当前 mma 之前发射）。
手写 ping-pong 只是重复了编译器已经在做的事，还多占了寄存器。

**真正的墙是"寄存器文件的两难"**（Ampere 特有）：要把张量核喂得更满，只有两条路——
(a) **更大 warp tile**（提高 mma:ldmatrix 比、增加独立累加链的 ILP），或 (b) **更多 warp**（TLP 藏延迟）。
但 64×32 warp tile 的 64 个 f32 累加已用掉 122 寄存器，A800 的 64K 寄存器/SM **两者只能取一**：
- 走 (a)：64×64 warp tile 需 128+ 累加寄存器 → 只能 1 block/SM → 张量核饿死到 39%。
- 走 (b)：强制 3 block/SM（cap 寄存器）实测**并不提速**——说明当前根本不是占用率受限，而是
  **每 warp 的 mma 发射受累加器依赖链 + 发射端口限制**，多给 warp 也喂不动更多 mma。

**结论**：`Issue Slots Busy` 仅 27%、Tensor 管线 43.7% 却已是最高利用——**纯 CUDA C++ 下 mma+ldmatrix
指令流的发射/依赖结构就卡在这里，~48% 是这一类可读手写 kernel 在 A800 的现实上限**。60–70% 属于
**CUTLASS/SASS 专属区间**（下一节）。

---

## 6. 再往 cuBLAS(85%) 还差什么——诚实的路线

cuBLAS BF16 在同卡到 264.7T=85% 峰（≈1.76× tc_06）。从 48% 到 85% 缺的**不是**某个能在 CUDA C++
里补的小优化，而是几层需要**汇编级掌控**的东西：

1. **SASS 级寄存器分配 + 指令调度**（最关键）：cuBLAS/CUTLASS 用手排的 SASS 把大 warp tile 的累加器
   塞进更少寄存器、同时保持 2 block/SM，从而同时拿到"大 tile 的 ILP"与"够的占用率"——这正是上面
   "寄存器两难"里 CUDA C++ 做不到的。nvcc 无法产出同等质量的调度。
2. **精确无冲突 swizzle + 双操作数 `ldmatrix.x4`**：本文的 padding 与朴素 swizzle 已消掉大部分 bank
   冲突，但 cuBLAS 的 128B swizzle + 最优 ldmatrix 排布还能再省一点访存/发射（本文实测这些在 CUDA C++
   里是中性偏负，需 SASS 才划算）→ 约 +3–5%。
3. **block rasterization / L2-aware 调度**：改块的执行顺序让并发块共享 A/B tile 命中 L2，主要救**大尺寸**
   （tc_06 8192³ 掉到 ~123T=L2/访存受限；4096³ 影响小）。
4. **必要时 split-K**：K 很大或机器填不满时并行累加再规约（本例 4096³ 块已够，不是主要杠杆）。

**一句话**：`mma.sync`+`ldmatrix`+`cp.async`（tc_06）已经把"手写能到的、可读的"那部分吃干（48%）；
剩下到 85% 的路本质是 **CUTLASS 那套 SASS 级机器**——所以工程上"逼近 cuBLAS"的正解是**对照/接入
CUTLASS 或 `cublasLt`**，而不是继续在 CUDA C++ 里堆手写技巧。这也划清了本教学阶梯的诚实边界。

---

## 7. 可移植性与对照

`mma.sync`/`ldmatrix`/`cp.async` 都是 **sm_80+ 通用指令**，故 tc_06 在 A800 与 Hopper（H20）上
**都能编都能跑**（不放进 `#ifndef NO_HOPPER`，id 固定=6）。在 Hopper 上它是"warp 级 `mma.sync`
vs warpgroup 级 `wgmma`(tc_04)"的对照：Hopper 的 WGMMA+TMA 能进一步把异步搬运/大 tile 交给硬件，
这也是为什么 Hopper 手写能到 82% 而 Ampere（无 WGMMA/TMA）手写天花板更依赖 TLP/ILP。

## 复跑

```bash
make tc ARCH=-arch=sm_80 TC_HOPPER=0
sudo nvidia-smi -lgc 1410
./build/tensor_core/verify 6 4096 4096 4096      # PASS（与 cuBLAS BF16 对拍）
./build/tensor_core/bench  6 4096 4096 4096      # ≈150T
sudo nvidia-smi -rgc
# ncu 逐核剖析（需 root）：
sudo /usr/local/cuda/bin/ncu --set full -k regex:mma_pipe_kernel -s 1 -c 1 \
     -o baselines/a800/a800_tc_06_mma_pipe ./build/tensor_core/bench 6 2048 2048 2048
```

**数据**：`baselines/a800/a800_tc_06_mma_pipe.details.txt`（本文引用）、
`a800_tc_03_wmma_pipe_reprofile.details.txt`（同轮对照）。相关：[[A800 GEMM 复现总结]] · [跨代际适配设计 §5](../跨代际适配设计.md)。
