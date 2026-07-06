# Softmax 算子总览：为什么它是 memory-bound，以及优化阶梯

## 1. 算子定义

对输入矩阵 `X[M, N]`（row-major, fp32）在**最后一维**上做行 softmax，得到 `Y[M, N]`：

```
y[i, j] = exp(x[i,j] - max_k x[i,k]) / Σ_k exp(x[i,k] - max_k x[i,k])
```

其中减去行最大值 `max_k x[i,k]` 是数值稳定的 **safe-softmax** 写法（防止 `exp` 上溢）。
CPU 参考实现（`softmax.h::softmax_cpu_ref`）用 double 精度做同样三步：求行 max → 求 exp 和 → 归一化。
本仓库 6 档 kernel（sc_01…sc_06）对 CPU double 参考**全部 PASS**（max_rel ≈ 2e-6）。

每一行 softmax 在数学上需要三次遍历这一行数据：

1. **max pass**：扫一遍求 `m = max_j x[j]`
2. **sum pass**：再扫一遍求 `l = Σ_j exp(x[j] - m)`
3. **write pass**：第三遍扫，写出 `y[j] = exp(x[j] - m) / l`

朴素实现里，这意味着 `x` 被从显存读 **三次**，`y` 被写 **一次**。这正是后面阶梯要攻击的核心。

## 2. 为什么 softmax 是 memory-bound（bytes/FLOP 分析）

Roofline 的判据是**算术强度**（arithmetic intensity, FLOP/byte）与硬件**脊点**（ridge point）的比较。

每个元素在三趟里做的算术量级：
- max pass：1 次比较
- sum pass：1 次减 + 1 次 `exp` + 1 次加
- write pass：1 次减 + 1 次 `exp` + 1 次乘

把 `exp` 慷慨地算作 ~10 FLOP，每个元素总计约 **20~30 FLOP**。而理论上每个元素只需搬运 **8 字节**（读 4 + 写 4）：

```
算术强度 ≈ 25 FLOP / 8 byte ≈ 3.1 FLOP/byte
```

H20 的 roofline 脊点（softmax 走 **FP32 CUDA 核 + SFU**，不碰张量核）：

```
ridge = FP32 CUDA-core 峰值 / HBM 峰值带宽
      ≈ 39.5 TFLOP/s / 4000 GB/s
      ≈ 9.9 FLOP/byte
# 39.5 TFLOPS = 78 SM × 128 FP32 lane × 2(FMA) × 1.98 GHz（gpu_specs.py 口径，锁频 1980 MHz）
```

**3.1 << 9.9**，落在 roofline 的左侧（带宽受限区）。而且注意 3.1 还是按“理想 8 字节”算的；朴素实现实际搬 `4×MN×4` 字节，真实强度更低。结论：**softmax 的性能上限由 HBM 带宽决定，不由算力决定**。所有优化都应该围绕“少搬字节 / 搬得更高效”，而不是“算得更快”。

这恰好是 **H20 的主场**：它是「算力穷、带宽富」的 Hopper 裁剪版——张量核峰值仅 **148 TFLOPS（BF16）/ 296 TFLOPS（FP8）**（Hopper 家族里的低配），却配了 **~4 TB/s 的 HBM3**。memory-bound 的 softmax 不吃算力、只吃带宽，正好把这 4 TB/s 吃满。

## 3. H20 HBM Roofline 与度量口径

- **硬件**：NVIDIA H20，Hopper（cc 9.0, sm_90a），**78 SM**，96 GB HBM3，峰值带宽 ≈ **4000 GB/s**，228 KB smem/SM，60 MB L2；CUDA 12.8 / driver 570。这 4000 GB/s 是所有 kernel 的理论天花板。
- **度量（见 `bench.cu`）**：softmax 的“有用工作”只有“读一遍 x + 写一遍 y”，所以定义

  ```
  理想流量 ideal_bytes = 2 * M * N * 4 bytes   （读一次 + 写一次）
  有效带宽 eff_BW = ideal_bytes / time
  达成率 = eff_BW / 4000
  ```

  （`bench.cu` 默认除数为 4000，可用环境变量 `PEAK_BW_GBS` 覆盖；峰值口径的单一来源是 `gpu_specs.py`。）
  注意这是一个**固定分子**的口径：不管 kernel 内部实际把 `x` 读了几遍，分子永远按“只读一次”算。因此：

  - 读 3 次 + 写 1 次（k=4）的 kernel，*朴素* `eff_BW` 天花板 ≈ `4000 × 2/4 = 2000 GB/s`（50%）。
  - 读 2 次 + 写 1 次（k=3）的 online kernel，≈ `4000 × 2/3 ≈ 2667 GB/s`（66.7%）。
  - 读 1 次 + 写 1 次（k=2）的寄存器驻留 kernel（sc_06），= `4000 × 2/2 = 4000 GB/s`（100%，即屋顶）。

- **H20 关键警示（软天花板）**：上面这几个 pass-count 天花板在 H20 上是**“软”的**——实测多趟 kernel 会**超过**它们（例如 sc_04 达 **79.1%**，高过 k=4 的 50%）。原因是 **H20 的 60 MB L2** 把同一行的**重复读就地命中**，实际 DRAM 流量远小于 `pass 数 × MN`。所以在 H20 上，“pass 数”只是流量的**宽松上界**；真正决定“少读一遍”能否兑现的，是**重复读能否被 L2 藏住**——短行（8192）藏得住，长行（16384）藏不住。这也是为什么 sc_06 的“单次 DRAM 读”只在**长行**上决定性取胜（见 §4 与 `04_beyond_online.md`）。

- **默认规模**：`M=N=8192`，即 64M 元素、256 MB 输入 + 256 MB 输出。整个矩阵（256 MB）远大于 60 MB L2，**无法整体缓存**；但每个 block 在**同一行内的重复读**（一行 32 KB、pass 之间紧邻）会命中 L2 —— 这正是多趟 kernel 在 H20 上跑赢“pass-count 天花板”的机制。把 N 拉到 **16384**（一行 64 KB）后，L2 对重复读的遮蔽变差，才轮到 sc_06 的寄存器驻留单次读发威。

## 4. 优化阶梯（ladder）

每一档只改一件事，攻击一个明确的瓶颈；末列是**本机 H20 实测**（8192²，锁频 1980 MHz）：

| 档位 | 唯一改动（one delta） | 攻击的瓶颈 | H20 实测有效带宽 |
|------|----------------------|-----------|------------------|
| **sc_01 naive** | 1 线程处理 1 行，三趟顺序扫 | 相邻线程访问相隔 N 的地址，**完全非合并**（stride-N），延迟受限 | **102.9 GB/s（2.6%）** |
| **sc_02 block-reduce** | 1 block 处理 1 行，块内 stride 遍历 + shared-mem 树规约 | 修复非合并：相邻线程读相邻元素 → **coalesced** | **2491（62.3%）** —— 全阶梯**收益最大**的一步 |
| **sc_03 warp-shuffle** | 规约改用 `__shfl_xor`，纯寄存器，无 smem 无 `__syncthreads` | 去掉 shared-mem 往返和块内 barrier | **1663（41.6%）** —— **H20 上反而退步**：warp-per-row 每行只有 32-way 内存并行，喂不满 4 TB/s（瓶颈本在 DRAM，不在规约） |
| **sc_04 vectorized** | 128-bit `float4` 读写 | 加宽访存事务（16 B/事务），减少请求数、提高每请求有效带宽 | **3163（79.1%）** —— 8192 上**最快**；L2 藏住重复读，实际 DRAM 流量已≈2·MN |
| **sc_05 online** ⭐ | 把 max pass 与 sum pass **融合成一趟**流式扫描，维护 running `(m, l)` 并在遇到更大 max 时 rescale | 从算法层**削减 DRAM 读流量**：读 3 次 → 读 2 次 | **3075（77.0%）** —— 与 sc_04 **基本打平**（L2 已藏住 sc_04 的第 3 读）；真正价值在长行 + **FA 桥梁** |
| **sc_06 resident** | 首次读入整行到**寄存器**，之后 max/sum/write 全在片上；读 1 写 1 | 把 DRAM 读压到**单次**（2·MN=屋顶理想流量） | 8192: **2575（64.4%）**；**16384: 3557（88.9%，长行夺冠）** |

阶梯的前四档（sc_01→sc_04）都在“把带宽用满”这条路上走。在 H20 上有两处与经典直觉不同、值得注意：

- **sc_03 退步**：去掉 barrier 本身不解决 DRAM 瓶颈；warp-per-row 把每行的内存并行从 256-way 砍到 32-way，在 4 TB/s 的 HBM3 上**喂不满**，反而慢于 sc_02。它是“小 N”的正确工具（见 `04`），不是 N=8192 的。
- **online 的收益被 L2 吃掉**：sc_05 从“3 读”降到“2 读”，但 H20 的 60 MB L2 本就把 sc_04 的第 3 读**就地命中**，所以 sc_05 ≈ sc_04。真正“少搬字节”一路走到底的是 **sc_06 的单次读**——而它要在**长行（16384）**上、L2 藏不住重复读时才决定性取胜（88.9%）。

> 一句话总结：sc_01→sc_04 教你“如何把 4 TB/s 的 HBM3 喂饱”（硬天花板 4000 GB/s）；sc_05/sc_06 教你“如何少喂”。在 H20 的大 L2 下，这份收益要在**长行**上由 sc_06 的**寄存器驻留单次读**兑现（88.9% 屋顶）；而 online（sc_05）的持久价值不在“独立 softmax 更快”，在于它**正是 FlashAttention 的内核**——那才是通往 attention 融合的门（见 `03_conclusion_online_bridge.md`）。
