# Softmax 算子总览：为什么它是 memory-bound，以及优化阶梯

## 1. 算子定义

对输入矩阵 `X[M, N]`（row-major, fp32）在**最后一维**上做行 softmax，得到 `Y[M, N]`：

```
y[i, j] = exp(x[i,j] - max_k x[i,k]) / Σ_k exp(x[i,k] - max_k x[i,k])
```

其中减去行最大值 `max_k x[i,k]` 是数值稳定的 **safe-softmax** 写法（防止 `exp` 上溢）。
CPU 参考实现（`softmax.h::softmax_cpu_ref`）用 double 精度做同样三步：求行 max → 求 exp 和 → 归一化。

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

A800 的 roofline 脊点：

```
ridge = FP32 峰值算力 / HBM 峰值带宽
      ≈ 19.5 TFLOP/s / 2039 GB/s
      ≈ 9.6 FLOP/byte
```

**3.1 << 9.6**，落在 roofline 的左侧（带宽受限区）。而且注意 3.1 还是按“理想 8 字节”算的；朴素实现实际搬 `4×MN×4` 字节，真实强度更低。结论：**softmax 的性能上限由 HBM 带宽决定，不由算力决定**。所有优化都应该围绕“少搬字节 / 搬得更高效”，而不是“算得更快”。

## 3. A800 HBM Roofline 与度量口径

- **硬件**：A800，HBM2e，峰值带宽 ≈ **2039 GB/s**。这是所有 kernel 的理论天花板。
- **度量（见 `bench.cu`）**：softmax 的“有用工作”只有“读一遍 x + 写一遍 y”，所以定义

  ```
  理想流量 ideal_bytes = 2 * M * N * 4 bytes   （读一次 + 写一次）
  有效带宽 eff_BW = ideal_bytes / time
  达成率 = eff_BW / 2039
  ```

  注意这是一个**固定分子**的口径：不管 kernel 内部实际把 `x` 读了几遍，分子永远按“只读一次”算。因此：

  - 一个真正跑满 HBM 的 kernel，如果实际搬了 `k` 份 `MN×4` 的流量，其 `eff_BW ≈ 2039 × (2/k)`。
  - 读 3 次 + 写 1 次（k=4）的 kernel，理论 `eff_BW` 天花板 ≈ `2039 × 2/4 ≈ 1020 GB/s`（50% 达成率）。
  - 读 2 次 + 写 1 次（k=3）的 online kernel，天花板 ≈ `2039 × 2/3 ≈ 1359 GB/s`（66.7%）。

  **这个口径的妙处**：它把“减少 pass 数”直接翻译成“有效带宽变高”，让 online 那一档的收益一眼可见。

- **默认规模**：`M=N=8192`，即 64M 元素、256 MB 输入 + 256 MB 输出。数据远大于 L2（A800 40 MB），无法缓存复用，每一 pass 都是真实 DRAM 往返 —— 这保证了 pass 数直接体现在流量上。

## 4. 优化阶梯（ladder）

每一档只改一件事，攻击一个明确的瓶颈：

| 档位 | 唯一改动（one delta） | 攻击的瓶颈 | 对有效带宽的预期效果 |
|------|----------------------|-----------|---------------------|
| **sc_01 naive** | 1 线程处理 1 行，三趟顺序扫 | 基线：相邻线程访问相隔 N 的地址，**完全非合并**（stride-N），延迟受限 | 极低，个位数 % 达成率（几十 GB/s 量级） |
| **sc_02 block-reduce** | 1 block 处理 1 行，块内 stride 遍历 + shared-mem 树形规约 | 修复非合并访问：相邻线程读相邻元素 → **coalesced** | 跃升到约 40~50% 达成率；此后瓶颈变为“读 3 次 x”的流量 + smem/barrier 开销 |
| **sc_03 warp-shuffle** | 规约改用 `__shfl_xor`，纯寄存器，无 smem 无 `__syncthreads` | 去掉 shared-mem 往返和块内 barrier | 相对 sc_02 小幅提升；仍是 3 次读 x 封顶 |
| **sc_04 vectorized** | 128-bit `float4` 读写 | 加宽每次访存事务（16 B/事务），减少请求数、提高每请求有效带宽 | 逼近“3 次读”的物理天花板（≈ peak/2 ≈ 1020 GB/s，约 50%） |
| **sc_05 online** ⭐ | 把 max pass 与 sum pass **融合成一趟**流式扫描，维护 running `(m, l)` 并在遇到更大 max 时 rescale | **削减 DRAM 读流量**：读 3 次 → 读 2 次（总流量 4 份 → 3 份，**-25%**） | 有效带宽相对 sc_04 提升约 **+33%** → 达到约 66% 达成率（≈ 1360 GB/s 天花板） |

阶梯的前四档（sc_01→sc_04）都在“把带宽用满”这条路上走，最终撞到“读 3 次 x”这堵墙。**sc_05 是唯一一档从算法层面拆墙的**：它不再更高效地搬同样多的字节，而是**根本少搬字节**。这一步——online 流式合并 `(m, l)`——正是 **FlashAttention 的桥梁**（详见 `03_conclusion_online_bridge.md`）。

> 一句话总结：sc_01→sc_04 教你“如何把 HBM 喂饱”，sc_05 教你“如何少喂”。前者有 2039 GB/s 的硬天花板，后者才是通往 attention 融合的门。
