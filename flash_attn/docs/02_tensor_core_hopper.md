# 02 · Tensor-Core 终态：手写 Hopper WGMMA + TMA 前向内核（H20, sm_90a）

> 前置：`00_principle.md`（FA 算法）、`01_cuda_core_scaffold.md`（fp32 语义参照）。

`flash_attn` 算子的张量核终态：一个从零手写的 Hopper flash-attention **前向**内核
——`flash_attn/include/fa_hopper.cuh`——在 H20 的**原生异步张量核路径**上跑
FlashAttention-2 数学。它是全仓库的综合：**两次 GEMM**（`Q·Kᵀ` 与 `P·V`）与
**online softmax** 融合，常驻 SRAM，使分数矩阵 `S` 从不落 HBM。

与 `cuda_core` 脚手架（用慢速 FP32 讲清算法）不同，这是性能路径：bf16/fp16 输入、
FP32 累加、`wgmma` + `TMA`。

---

## 1、为什么要手写

在 H20 上，注意力真正的性能天花板是库（PyTorch SDPA / FA2，~90–96% MFU，见
`baselines/`）。本内核的意义是**教学与架构性的**：用 ~300 行可读代码展示三件
Hopper 原语（`TMA`、`WGMMA`、寄存器常驻 online softmax）如何组合成一个真实的 FA
内核——与 CUTLASS/FA3 相同的骨架，但没有那层抽象。它落在 **148 TFLOPS BF16 峰值的
80–86%**，与库同一量级。

## 2、三件 Hopper 原语

| 原语 | 在这里做什么 | PTX / API |
| --- | --- | --- |
| **TMA**（张量内存加速器） | 单线程描述符驱动地把 `K`、`V` tile 从 global 搬到 smem，128B swizzle，`mbarrier` 完成同步 | `cp.async.bulk.tensor.2d`（`cuda::device::experimental`） |
| **WGMMA**（warpgroup MMA） | **两次矩阵乘都用它**：warpgroup 级（128 线程）、异步、f32 累加 | `wgmma.mma_async.sync.aligned.m64n64k16.f32.{bf16,f16}` |
| **寄存器常驻 online softmax** | 行 max/sum 归约沿 WGMMA 累加器布局走 4-lane warp shuffle——`S`/`P` 无需 smem 往返 | `__shfl_xor_sync` |

## 3、算法：一个 warpgroup 负责一个 query tile

grid 为 `(S/Br, H, B)`；一个 **warpgroup（128 线程）**负责某个 `(batch, head)` 的
一个 `Br = 64` query tile。对 `Bc = 64` 的 key/value tile 循环：

```
一次性载入 Q tile (TMA) ─┐
for 每个 KV tile kv:      │  (causal：到对角 tile 即止)
    TMA 载入 K,V tile ────┘  (双缓冲 smem，128B swizzle)
    S = scale · Q·Kᵀ            ← WGMMA，A=Q(smem) B=K(smem)，累加在寄存器
    m_new = max(m, rowmax(S))   ← 沿 Bc 列做 4-lane shfl 归约
    P = exp(S - m_new)          ← 就在寄存器里，直接排成下一次矩阵乘的 A-fragment
    l = l·exp(m-m_new) + rowsum(P)
    O = O·exp(m-m_new) + P·V    ← WGMMA，A=P(寄存器) B=V(smem)，累加
O /= l ; 写回 O tile
```

### 3.1 关键：P→A-操作数 的直接交接（无 smem 往返）

FA 在张量核上跑得快的原因、也是最难写对的一处：**matmul-1 的 WGMMA 累加器布局
与 matmul-2 的 A-操作数寄存器布局逐比特相同**。对 `m64nNk16`，每条 lane 为它的
两行持有同样 8 个元素 `{col, col+1, col+8, col+9}`——在 `S` 累加器里是 f32、在
`P` 的 A-fragment 里是 bf16。于是 `softmax(S)` **直接作为 `P·V` 的 `A` 寄存器喂回
去**——`P` 无需写 smem 再读回，也无需转置。这正是 FA3/CUTLASS 依赖的寄存器常驻
设计，这里把它显式写了出来（`fa_hopper.cuh` 中 `sd[g][0..7]` → `pk[g][0..3]`）。

`V` 也无需转置：它是 `P·V` 沿 key 维收缩的 `B` 操作数，用 `wgmma` **`transB=1`**
（MN-major）直接吃它 TMA 载入的 `[Bc, D]` tile。

### 3.2 128B-swizzle 的「inner-64」几何

WGMMA 的 smem 操作数必须是 TMA 的 128B-swizzle *core-matrix* 布局，而描述符魔数
（`LBO=16, SBO=1024`）绑死在**64 元素宽**的连续 tile（bf16 恰好一个 128B swizzle
atom）。所以 head dim `D` 以 **64 宽的块**流入：`D=64` 是一块，`D=128` 是两块。
`Q·Kᵀ` 沿 `D` 收缩（对块循环）；`P·V` 的输出 `N=D` 拆成 `NC = D/64` 个独立的
`n64` WGMMA，每个 V 列块一个。这复用了 `gemm/tc_04` 里那份已验证的描述符。

## 4、正确性

对 FP32 CPU 参考注意力（`fa_cpu_ref`，`allclose` atol/rtol `2e-2`，即 bf16/fp16
张量核容差）验证：`D ∈ {64,128}`、bf16 **与** fp16、causal **与** 非 causal、多
KV-tile 序列（`S ≤ 512` = 至多 8 个 KV tile；更大 `S` 走完全相同的代码路径），全部
**PASS**，`bad=0`。典型 `max_abs ≈ 3e-3`（bf16）、`≈ 3e-4`（fp16）。

> CPU 参考是 `O(B·H·S²·D)` 标量 C++，故只跑到 `S=512`；`bench` 驱动在 `S=4096`
> 只单独计时 GPU 内核。

## 5、性能（H20，SM 锁频 1980 MHz）

| shape | dtype | causal | TFLOPS | 占 148T |
| --- | --- | --- | ---: | ---: |
| B2 H32 S4096 D128 | bf16 | 否 | **122.5** | **82.8%** |
| B2 H32 S4096 D128 | fp16 | 否 | 122.2 | 82.6% |
| B2 H32 S4096 D128 | bf16 | 是 | 113.8（有效） | 76.9% |
| B4 H32 S8192 D128 | bf16 | 是 | 127.9（有效） | 86.4% |
| B2 H16 S2048 D64  | bf16 | 否 | 120.1 | 81.1% |

**复现：** `flash_attn/build/tensor_core/bench bf16 2 32 4096 128 0`

### 5.1 单次 grid 启动这一课（35% → 83%）

第一版**每个 `(b,h)` 启一个 grid**（`B·H` 次串行启动），只到峰值的 **35%**：每次
启动都被它最慢的 block（一个满长度 query tile）卡住，causal 跳过毫无收益，启动之间
也不重叠。塌缩成对 `(S/Br, H, B)` 的**单次 grid**——把每个 `(b,h)` 的 TMA map 缓存
成设备数组、用 `blockIdx` 索引——一跃到 **83%**，并让 causal 的墙钟正确地快约 2×。
启动结构值 2.4×。

## 6、为什么 FA 在 H20 上能到 83% —— ncu Speed-of-Light

`ncu --set full` 于 `fa_kernel`（`baselines/tensor_core/fa_tc_hopper_wgmma.details.txt`）：

| 指标 | 值 | 解读 |
| --- | ---: | --- |
| DRAM Throughput | **1.3%** | 计算受限——4 TB/s HBM 闲置；148T 张量核才是墙 |
| Tensor (FP) 管线 | 充分利用 | `wgmma` 把张量管线喂满 |
| Achieved Occupancy | **12%** | 每 SM 仅 ~2 个 block（127 寄存器/线程，~80 KB smem） |
| 寄存器/线程 | 127 | 寄存器常驻的 `S`/`P`/`O` 累加器 |

最耐人寻味的是：**12% 占用率仍拿到 83% MFU**。因为 `wgmma` 是*异步*且 warpgroup
级的，单个常驻 warpgroup 就能发出足够多的独立张量工作把管线填满——「靠高占用率
掩盖延迟」这条常规在这里不成立。这是一课 Hopper 特有的经验，且在 H20 上被*放大*：
只有 148 TFLOPS 张量吞吐、却有 4 TB/s 带宽去喂它，矩阵乘从不挨饿，所以一个朴素内核
也能拿到高 MFU。（换到 990 TFLOPS 张量核的 H100，同样没有软件流水的内核就会挨饿。）

## 7、局限与优化前沿

本内核刻意做成*最小可用正确*的 Hopper FA。FA3 在其上再加的、也是能补上最后 ~10–15%
去追平库的，是：

- **Warp specialization**（一个生产者 warpgroup 发 TMA + 一个消费者 warpgroup 做
  WGMMA）配 **2 级异步流水**，用下一个 tile 的 TMA 重叠当前 tile 的计算。这里 smem
  虽双缓冲，但是同步消费（compute 前 barrier-wait），故 TMA 与 WGMMA 不重叠。
- **Ping-pong 调度**：两个 warpgroup 互相把 softmax（非张量工作）藏到对方的 WGMMA 后面。
- `{64,128}` 之外的 head dim、GQA（`Hkv < Hq`）、以及 `bf16` 输出经 smem 的 TMA store。

这些是自然的下一级；而正确、已验证的 WGMMA+TMA+online-softmax 内核，是它们的地基。

---

**另见：** `00_principle.md`（FA2 数学）、`01_cuda_core_scaffold.md`（FP32 教学
脚手架）、`03_synthesis_gemm_softmax_fa.md`（GEMM + softmax + FA 在 H20 上如何贯通），
以及 `../../gemm/docs/`（本内核描述符几何借自那里的 WGMMA/TMA GEMM `tc_04`）。
