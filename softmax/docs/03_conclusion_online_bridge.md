# 结论：Online Softmax 是通往 FlashAttention 的桥梁

前四档（sc_01→sc_04）都在回答同一个问题：“如何把 HBM 带宽喂满？”它们的天花板是硬件的 2039 GB/s，而“读 3 次 x”的结构把有效带宽死死压在 `peak/2 ≈ 1020 GB/s`。sc_05 换了个问题：“**能不能少读一遍？**”答案就是 online softmax 的 `(m, l)` 递推——而这**恰好**是 FlashAttention 赖以成立的那块拼图。

## 1. 从两趟到一趟：为什么需要 rescale

Safe-softmax 天生是两趟的：
```
m = max_j x[j]                 # 趟 1：必须先扫完全部，才知道全局 max
l = Σ_j exp(x[j] - m)          # 趟 2：exp 里的 m 依赖趟 1 的结果
```
第二趟的每个 `exp(x[j]-m)` 都要用到**全局** `m`，所以看似必须等趟 1 结束。online 的核心是打破这个依赖：**用“当前见过的局部 max”先累加，等 max 变大时对已累加的和做一次统一修正**。

## 2. Online (m, l) 递推的推导

维护两个 running 量：当前最大值 `m` 与“以 `m` 为基准的 exp 部分和” `l = Σ exp(x - m)`。
来一个新元素 `x`（把它看成一个 `(m2=x, l2=1)` 的单元素 partial），合并规则是：

```
m_new = max(m, x)
l_new = l * exp(m - m_new) + exp(x - m_new)
m ← m_new
```

**为什么正确**：`l = Σ exp(x_k - m_old)`。当 `m_old → m_new` 时，每一项的基准都要从 `m_old` 平移到 `m_new`：
```
exp(x_k - m_new) = exp(x_k - m_old) · exp(m_old - m_new)
```
所以整个旧和乘上同一个因子 `exp(m_old - m_new)` 即可，无需重扫历史元素——这就是那一项 `l * exp(m - m_new)`。新元素 `x` 直接以新基准记为 `exp(x - m_new)`。归纳可知：扫完整行后，`m` 是全局 max、`l = Σ_k exp(x_k - m)`，与两趟法**逐位等价**。

代码里就是 `sc_05_online.cu::mergeMl`：
```cuda
__device__ void mergeMl(float& m, float& l, float m2, float l2){
  float mn = fmaxf(m, m2);
  l = l*__expf(m - mn) + l2*__expf(m2 - mn);   // 对齐到共同基准 mn
  m = mn;
}
```

## 3. 同一个 merge，用在三个层级

`mergeMl` 满足**结合律**（它本质是在“log-sum-exp 半环”上的加法），所以可以自由地分块、并行、再合并。在 sc_05 里它被复用于三个层级，形态完全一致：

1. **元素级**：`float4` 的 4 个分量折进线程的 running `(m, l)`（`mergeMl(m,l,v.x,1.f)…`）。
2. **线程级 → 块级**：block 内树规约把每个线程的 partial `(m,l)` 两两 `mergeMl` 合并成整行结果。
3. **（本仓库外推）tile 级**：如果一行被切成多个 tile 分别算出 `(m,l)`，再用同一个 `mergeMl` 把 tile 结果拼起来——**这正是 FlashAttention 的做法**。

关键点：**跨 softmax 行**（这里每行独立）和**跨 attention 的 score 分块**（FA 里一个 query 的 K 维被切成多个 tile），用的是**同一个** `(m,l)` 合并算子。sc_05 把这个技巧从 attention 里剥离出来，放在一条普通的行上单独练熟。

## 4. 连接 FlashAttention：这一档是 FA 的先决条件

标准 attention 对每个 query 要算 `softmax(q·Kᵀ) · V`。分数矩阵 `S = q·Kᵀ` 的长度是全部 key（可能上万），如果按经典 softmax：
- 必须先扫完整行 `S` 求全局 max（趟 1），
- 再扫一遍求 exp 和（趟 2），
- 才能加权 `V`。

这意味着**整行 `S` 必须先物化到 HBM**——这正是 attention 的 `O(N²)` 显存瓶颈。FlashAttention 的破局点就是：**不物化 `S`**，而是把 K/V 切成 tile，在片上流式地扫过每个 tile，用 online `(m, l)` 递推**增量地**更新 softmax 统计量，同时增量地累加输出 `O`：

```
对每个 K/V tile:
  m_new = max(m, rowmax(S_tile))
  修正因子 α = exp(m - m_new)
  l = α·l + rowsum(exp(S_tile - m_new))
  O = α·O + exp(S_tile - m_new) · V_tile     # 输出也用同一个 α rescale
  m = m_new
```

把上面这段和 `mergeMl` 并排看：**完全是同一个 `(m, l)` 合并**，只是多带了一个被同一个 `α = exp(m_old-m_new)` 修正的输出累加器 `O`。也就是说：

> **没有 online softmax 的 `(m, l)` rescale 递推，FlashAttention 就无法在“不物化整行分数”的前提下算出正确的 softmax。sc_05 学会的，正是 FA 的先决 trick。**

## 5. 收束

- **前四档的教训**：softmax 是 memory-bound，把带宽喂满能到 `peak/2 ≈ 1020 GB/s`，但“读 3 次 x”是道结构性的墙。
- **sc_05 的突破**：online 融合把读 x 从 3 次降到 2 次，总流量 -25%，有效带宽 +33% → `≈ 1360 GB/s`（66% 达成率）。这是唯一靠“少搬字节”而非“搬得更快”拿到的提升。
- **更大的意义**：sc_05 的 `(m, l)` 递推不是一个孤立的访存优化，而是 **FlashAttention 的数学内核**。在这条普通的 softmax 行上练熟它，就等于握住了把 softmax 融进 attention、突破 `O(N²)` 显存墙的钥匙。

这就是为什么在这个学习阶梯里，online 那一档被称作 **FlashAttention 桥梁**：它是从“单算子带宽优化”跨到“算子融合 / 长序列注意力”的那一步。
