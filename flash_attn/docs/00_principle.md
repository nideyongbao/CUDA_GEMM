# 00 · FlashAttention 原理：Online Softmax 与 IO 优化

> 本文是整个 flash_attn 算子的**数学地基**。它只讲“为什么 FA 是对的、为什么 FA 快”，
> 不碰任何 CUDA 代码。代码怎么落地见 `01_cuda_core_scaffold.md`（fp32 教学版）与
> `02_tensor_core_tinyfa.md`（张量核性能版）。
>
> 材料来源：`FA 细节.md`（Online-Softmax 四步推导 + 具体序列推演 + FA1/FA2 伪代码 + 微型 IO 推演）。

---

## 0. 一句话总览

标准 Attention 的瓶颈**不是算力，是访存**：它要把 `N×N` 的中间矩阵 `S = QKᵀ`、`P = softmax(S)`
写进 HBM 再读回来。FlashAttention 用两把武器干掉这个中间矩阵：

1. **Online Softmax**（本文 Step 1→4）——让 softmax 可以“边扫边算”，无需先看到整行 `S` 再归一化，
   于是 `S/P` 永远不落 HBM，只在 SRAM 里以 `(m, l)` 两个标量的形式流动。
2. **Tiling + 循环顺序**（本文 FA1 vs FA2）——把 Q/K/V 分块塞进 SRAM 复用，并且把 **Q 放到外循环**，
   让输出块 `O` 从头到尾钉死在 SRAM，只在最后写一次 HBM。

结论先行：在下文的微型环境里，HBM 访存从标准的 **192 B** 一路降到 FA2 的 **96 B**（省一半）。

---

## 1. Step 1 → Step 2：标准 Softmax → Safe Softmax

**标准定义**：

$$\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_{j=1}^{B} e^{x_j}}$$

问题很直接：`x_i` 稍大就溢出。float16 最大约 65504，`x_i = 100` 时 `e^100` 直接变 `NaN`。

**Safe Softmax**：分子分母同乘 `e^{-m(x)}`（`m(x)` 是全局最大值），指数一律变成 `≤ 0`，`e` 的结果落在 `(0, 1]`：

$$m(x) = \max_i x_i,\qquad \tilde f_i = e^{x_i - m(x)},\qquad \ell(x) = \sum_i \tilde f_i$$

$$\text{softmax}(x_i) = \frac{e^{x_i - m(x)}}{\sum_j e^{x_j - m(x)}} = \frac{\tilde f_i}{\ell(x)}$$

数值上完全等价（分子分母同乘一个常数不变），但不再溢出。**代价**：它需要 `m(x)`，也就是必须**先扫一遍整行**拿到最大值，才能开始算 `e`。这正是 online softmax 要打破的约束。

---

## 2. Step 3：Online Softmax v1（分块递推 / FA1 思想）

如果一行 `x ∈ ℝ^N` 长到装不进 SRAM，就必须切块 `x = [x^{(1)}, …, x^{(T)}]` 逐块处理。难点：处理第 k 块时，**还没看到后面的块**，此刻的“最大值”只是局部的、可能被后面推翻。Online softmax 的做法是维护两个 running 标量并在遇到更大值时**追溯修正**。

处理第 k 块，已知历史 `(m_{k-1}, ℓ^{all}_{k-1})`，当前块局部量 `m(x^{(k)})`、`ℓ(x^{(k)})`：

$$\boxed{m_k = \max\big(m_{k-1},\, m(x^{(k)})\big)}$$

$$\boxed{\ell^{all}_k = e^{m_{k-1} - m_k}\cdot \ell^{all}_{k-1} \;+\; e^{m(x^{(k)}) - m_k}\cdot \ell(x^{(k)})}$$

推导核心只是“换底”——把历史那部分从旧基准 `m_{k-1}` 校准到新基准 `m_k`：

$$
\ell^{all}_k = \sum_{i<k} e^{x_i - m_k} + \sum_j e^{x^{(k)}_j - m_k}
= \underbrace{e^{m_{k-1}-m_k}}_{\text{历史校正因子}}\ell^{all}_{k-1} + \underbrace{e^{m(x^{(k)})-m_k}}_{\text{当前块校正因子}}\ell(x^{(k)})
$$

> **关键发现**：整行 softmax 的正确性，只需维护 `m_k`（running max）和 `ℓ^{all}_k`（running sum）**两个标量**，不需要存任何 `N` 长度的中间向量。

**v1 的特征**：它在每块结束后，都用下式把之前所有块的 softmax **重标定回完全归一化状态**：

$$\text{softmax}_k(x^{(i)}) = \text{softmax}_{k-1}(x^{(i)})\cdot \frac{\ell^{all}_{k-1}}{\ell^{all}_k}\cdot e^{m_{k-1}-m_k},\quad i<k$$

也就是说，v1 每处理一块，**当前的输出 O 都是合法的、已归一化的概率**。代价是内层循环里那个 `/ ℓ^{all}_k`——每块都要除一次。

---

## 3. Step 4：Online Softmax v2（延迟归一化 / FA2）

FA2 的洞察：既然最终只关心比例，**分母的除法可以推迟到最后统一做一次**。中间过程带着“未归一化的裸值”跑就行。

把输出写成累积形式（`Õ` = 未归一化累积输出，`P̃` = 未归一化权重）：

$$O^{(2)} = \text{diag}(\ell)^{-1}\Big(\text{diag}(\ell^{(1)})\,e^{m^{(1)}-m}\, O^{(1)} + e^{m^{(2)}-m}\,\tilde P^{(2)} V^{(2)}\Big)$$

把最外层的 `diag(ℓ)^{-1}` 拎出循环，循环体内只保留“校正历史 + 累加当前”：

$$\boxed{\tilde O^{(k)} = e^{m_{k-1}-m_k}\cdot \tilde O^{(k-1)} \;+\; e^{m(x^{(k)})-m_k}\,\tilde P^{(k)} V^{(k)}}$$

内层循环全部跑完后，**只做一次**归一化（epilogue）：

$$\boxed{O = \frac{\tilde O^{(T)}}{\ell^{all}_T}}$$

| 版本 | 归一化时机 | 非矩阵乘（除法/缩放）次数 |
|------|-----------|--------------------------|
| **v1** | 每块都重标定 | O(T) |
| **v2 (FA2)** | 仅最后一次 | O(1) |

> **为什么这很重要**：除法、`exp` 走的是 GPU 的 SFU / CUDA-core 标量单元，**不是** Tensor Core。把 O(T) 次除法压成 O(1) 次，等于把标量单元从关键路径上挪开，让 Tensor Core 专心做矩阵乘 → 提升 MFU。

---

## 4. 把 Step 4 算成具体数字

设分数 `X = [1, 3, 2, 4]`，值 `V = [10, 20, 30, 40]`（这里 V 简化成 1 维标量，真实模型里是 d 维向量，逐维逻辑一致）。切成两块：

- 块 1（k=1）：`X⁽¹⁾ = [1, 3]`, `V⁽¹⁾ = [10, 20]`
- 块 2（k=2）：`X⁽²⁾ = [2, 4]`, `V⁽²⁾ = [30, 40]`

**一次性 Safe Softmax（对账基准）**：`m = 4`，`f̃ = [e⁻³, e⁻¹, e⁻², e⁰] ≈ [0.0498, 0.3679, 0.1353, 1]`，`ℓ = 1.553`，权重 `≈ [0.032, 0.237, 0.087, 0.644]`。

**处理块 1（FA2）**：`m₁ = 3`，`P̃⁽¹⁾ = [e⁻², e⁰]`

$$\tilde O^{(1)} = e^{-2}\cdot 10 + e^{0}\cdot 20 \approx 1.353 + 20 = 21.353$$

**处理块 2（FA2）**：`m₂ = max(3,4) = 4`，`P̃⁽²⁾ = [e⁻², e⁰]`

$$\tilde O^{(2)} = e^{3-4}\cdot \tilde O^{(1)} + e^{4-4}\cdot(e^{-2}\cdot 30 + e^{0}\cdot 40) \approx 0.3679\cdot 21.353 + 44.060 = 51.915$$

**Epilogue（唯一一次除法）**：先前递推得 `ℓ^{all}_2 = e^{3-4}·1.1353 + e^{4-4}·1.1353 = 0.4176 + 1.1353 = 1.553`，

$$O = \frac{\tilde O^{(2)}}{\ell^{all}_2} = \frac{51.915}{1.553} \approx 33.43$$

**对账**：直接用 Safe Softmax 权重加权 V：`0.032·10 + 0.237·20 + 0.087·30 + 0.644·40 ≈ 33.43`。✅ 完全一致——延迟归一化没有牺牲任何精度。

---

## 5. FA1 vs FA2：循环顺序互换（最致命的 IO 优化）

上面的递推是“单行”视角。落到多 Query 并行时，分块的**循环嵌套顺序**决定了 IO 成败。

### FA1 伪代码（外循环 K/V，内循环 Q）

```python
O = zeros(N, d); M = full(N, -inf); L = zeros(N)     # 全局状态在 HBM
for j in range(0, N, Bc):            # 外循环：K/V 块
    k_block, v_block = K[j:j+Bc], V[j:j+Bc]
    for i in range(0, N, Br):        # 内循环：Q 块
        q_block = Q[i:i+Br]
        o_block, m_block, l_block = O[i:i+Br], M[i:i+Br], L[i:i+Br]   # ← 从 HBM 读半成品
        s_block = q_block @ k_block.T
        for r in range(Br):          # 逐行 online 更新（v1 风格，除以 l_new）
            ...
            o_block[r] = o_old*l_old*exp(m_old-m_new)/l_new + (p̃ @ v_block)*exp(row_m-m_new)/l_new
        O[i:i+Br], M[i:i+Br], L[i:i+Br] = o_block, m_block, l_block   # ← 写回 HBM
```

痛点：K/V 在外，同一个 Q 块的半成品 `(O, m, l)` **无法常驻 SRAM**。每轮外循环都得把它从 HBM 读出来、更新、再写回去——`O` 在 SRAM 和 HBM 之间反复横跳（ping-pong）。

### FA2 伪代码（外循环 Q，内循环 K/V）

```python
for i in range(0, N, Br):            # 外循环：Q 块（各块无数据依赖 → 可并行到 seq 维度）
    q_block = Q[i:i+Br]
    o_block = zeros(Br, d); m_block = full(Br, -inf); l_block = zeros(Br)   # ← 常驻 SRAM
    for j in range(0, N, Bc):        # 内循环：K/V 块
        k_block, v_block = K[j:j+Bc], V[j:j+Bc]
        s_block = q_block @ k_block.T
        for r in range(Br):
            m_old, l_old = m_block[r], l_block[r]
            row_m = rowmax(s_block[r]); row_ell = rowsum(exp(s_block[r]-row_m))
            m_new = max(m_old, row_m)
            l_new = l_old*exp(m_old-m_new) + row_ell*exp(row_m-m_new)
            p̃ = exp(s_block[r] - m_new)                       # 对齐全局 max
            o_block[r] = o_block[r]*exp(m_old-m_new) + p̃ @ v_block   # ← 延迟归一化，不除 l_new
            m_block[r], l_block[r] = m_new, l_new
    for r in range(Br): o_block[r] /= l_block[r]              # ← Epilogue，唯一一次除法
    O[i:i+Br] = o_block                                        # ← 只写一次 HBM
```

### FA2 相对 FA1 的三大区别

1. **循环顺序互换（IO 决胜手）**：Q 外循环让 `O_block/m/l` 全程锚定 SRAM，内层疯狂灌 K/V 更新即可。`O` 对 HBM 的读写次数从 FA1 的 `N/Bc` 次直降到 **1 次**。
2. **延迟归一化**：内层公式从 `O·(l_old/l_new)·e^{Δm} + (e^{S-row_m}·V)/l_new` 简化成 `O·e^{Δm} + e^{S-m_new}·V`，把昂贵的除法踢出内循环，只在 epilogue 除一次。
3. **并行维度扩展**：Q 块之间无依赖，FA2 可以把并行度铺到 **序列长度** 维。哪怕 batch=1、head 少，只要 N 够长也能喂饱几百个 SM（长上下文场景 FA1 会 SM 饥饿）。

---

## 6. 微型环境 IO 推演（192 → 160 → 128 → 96 字节）

**环境**：`N=4, d=2, float16(2 B/元素)`。`Q/K/V` 各 `4×2×2 = 16 B`。SRAM 上限 40 B，HBM 搬 1 B 记 1 单位时间。

### 演进 1 · 标准 Attention（显存刺客，192 B）
生成 `4×4` 的 `S`(32 B) 与 `P`(32 B)，两者都要写 HBM 再读回：
读(Q16+K16) + 写S(32) + 读S(32) + 写P(32) + 读(P32+V16) + 写O(16) = **192 B**。SRAM 峰值 32 B，且随 `N²` 爆炸。

### 演进 2 · 单向量融合 + Online Softmax（160 B）
逐个 Query 在 SRAM 边算边累加，消灭中间矩阵。但**每算一个 q 都要把整个 K、V 重读一遍**：
读Q(16) + `4×(K16+V16)`=128 + 写O(16) = **160 B**。SRAM 峰值仅 20 B（彻底摆脱 `N²`），但 KV 重复搬运 N 次。

### 演进 2.5 · FA1 分块（K/V 外循环，128 B）
`2×2` 分块（每块 8 B）。K/V 各读一次(32 B) + Q 读两次(32 B) + **半成品 O 反复读写**(读16×2 + 写16×2 = 64 B) = **128 B**。SRAM 峰值 32 B。瓶颈：`O` 在 SRAM/HBM 间 ping-pong。

### 演进 3 · FA2 分块（Q 外循环，96 B）
Q 外循环，`O_block` 锚定 SRAM。Q 读一次(16 B) + K/V 被 2 个 Q 块各轮询一次(2×32 = 64 B) + **O 只写一次**(16 B) = **96 B**。SRAM 峰值 32 B。`O` 零换页。

### 总结对比

| 算法版本 | 循环逻辑 | 中间矩阵 | SRAM 峰值 | **HBM 访存** | 核心瓶颈 / 优势 |
|---|---|---|---|---|---|
| 标准 Attention | 全矩阵 | 是 (32 B) | 32 B（随 N² 爆） | **192 B** | S、P 撑爆 HBM 带宽 |
| 单向量融合 | 逐向量 | 否 | 20 B | **160 B** | K、V 被重复读 N 次 |
| FA1（初级 Tiling） | K/V 外, Q 内 | 否 | 32 B | **128 B** | 半成品 O 反复读写 |
| **FA2（最终形态）** | **Q 外, K/V 内** | 否 | 32 B | **96 B** | **O 零换页，极致压缩** |

一句话收束三层优化：**Online Softmax** 换取“不生成中间大矩阵”的特权；**Tiling** 减少 KV 重复搬运；**FA2 倒置循环**彻底终结输出 `O` 的 IO。`N=8000` 甚至更长时，这种访存缩减是压倒性的。

---

## 7. 接下来

- 这套递推怎么用 fp32 CUDA-core 一行行写出来（先单 Query 版 `fa_cc_01`，再 FA2 分块版 `fa_cc_02`）→ **`01_cuda_core_scaffold.md`**
- 怎么把同样的算法搬到 Tensor Core（mma.sync / ldmatrix / cp.async，~94–96% Dao FA2）→ **`02_tensor_core_tinyfa.md`**
- FA 到底“新”在哪：它 = 两个 GEMM + online softmax **融合**，几乎没有全新原语 → **`03_synthesis_gemm_softmax_fa.md`**
