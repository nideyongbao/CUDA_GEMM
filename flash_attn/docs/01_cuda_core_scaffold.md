# 01 · CUDA-Core 脚手架：把 FA2 算法写成 fp32 内核

> 前置：`00_principle.md`（online softmax v1/v2、FA1/FA2 循环顺序、IO 推演）。
> 本文对象：`flash_attn/kernels/cuda_core/` 下两支 **纯 fp32、无 Tensor Core** 的教学内核，
> 注册表见 `flash_attn/include/fa_cc.h`。
>
> | id | 文件 | 角色 | 对应 `FA 细节.md` |
> |----|------|------|-----------------|
> | 1 | `fa_cc_01_stream.cu` | 单 Query 单遍 online | Step 4「单 Query 迭代版」 |
> | 2 | `fa_cc_02_tiled.cu` | FA2 分块 + smem K/V | 「FA2 完整分块算法」 |

## 定位：这两支内核是“算法正确性基线”，故意慢

它们的唯一目的是**把 `00_principle.md` 的递推公式一比一翻译成可运行、可对拍的 CUDA**，
让你在没有 CuTe / PTX / 张量核这些噪音的情况下，先把 FA2 的数据流看清楚：

- **纯 fp32 FMA**，一个 `q[d]*k[d]` 一个 `s`，没有 `mma.sync`、没有 fragment；
- 累加器、`(m, l)` 全是**普通寄存器标量**，online 重标定就是几行 `__expf`；
- `fa_cc_01` 甚至是**完全非合并访存**（一个线程走完一整行 K/V）——慢是刻意的。

它们跑得慢，但它们是 `02_tensor_core_tinyfa.md` 里那支高性能内核的**语义参照物**：
张量核版做的每一步（online 重标定、延迟归一化、Q 外 / KV 内），都能在这里找到标量对应。
数据布局统一为 `[B, S, H, D]` 行主序，索引宏 `idx4` 见 `fa_common.h`。

---

## 1. `fa_cc_01_stream.cu`：单 Query，单遍 online

### 并行结构
`grid = ⌈B·H·S / 128⌉, block = 128`。**一个线程负责一个 `(b, h, i)` 三元组**，即输出矩阵的一整行 `o ∈ ℝ^D`：

```cuda
int gid = blockIdx.x * blockDim.x + threadIdx.x;   // 一线程一个 (b,h,i)
if (gid >= B * H * S) return;
int i = gid % S; int t = gid / S; int h = t % H; int b = t / H;
```
这正是 `00_principle.md`「演进 2：单向量融合」的线程化——每个线程持有一个 query，独立扫完所有 key。

### 状态：未归一化累加器 + running (m, l)
```cuda
float acc[kMaxD];                 // 未归一化累积输出 Õ（栈上寄存器数组，kMaxD=128）
for (int d = 0; d < D; ++d) acc[d] = 0.f;
float m = -INFINITY, l = 0.f;     // running max 与 running sum
```
`acc` 就是 Step 4 的 `Õ`，`m/l` 就是递推里的 `m_k / ℓ^{all}_k`。**注意 `S = QKᵀ` 矩阵从未被物化**——`acc` 是 `D` 长，`m/l` 是标量，全程 `O(D)` 内存，没有任何 `N` 长向量。

### 单遍在线循环（Step 4 递推逐字翻译）
```cuda
int jmax = causal ? i : S - 1;          // causal：key j 只能到 i
for (int j = 0; j <= jmax; ++j) {
  const float* k = K + idx4(b, j, h, 0, S, H, D);
  float s = 0.f;
  for (int d = 0; d < D; ++d) s += q[d] * k[d];   // s = q·kⱼ（CUDA-core FMA）
  s *= scale;                                       // scale = 1/√D
  float mnew = fmaxf(m, s);                          // m_new = max(m, s)
  float c = __expf(m - mnew);                        // 历史校正因子 e^{m_old-m_new}
  float p = __expf(s - mnew);                        // 当前 key 权重 e^{s-m_new}
  l = l * c + p;                                     // ℓ 递推
  const float* v = V + idx4(b, j, h, 0, S, H, D);
  for (int d = 0; d < D; ++d) acc[d] = acc[d] * c + p * v[d];  // Õ 递推：先缩历史，再加当前
  m = mnew;
}
```
逐行对齐 `00_principle.md` 的 boxed 公式：

| 代码 | 公式 |
|------|------|
| `mnew = fmaxf(m, s)` | `m_k = max(m_{k-1}, m(x^{(k)}))` |
| `l = l*c + p` | `ℓ_k = e^{m_{k-1}-m_k}·ℓ_{k-1} + e^{...}·ℓ(x^{(k)})` |
| `acc[d] = acc[d]*c + p*v[d]` | `Õ^{(k)} = e^{m_{k-1}-m_k}·Õ^{(k-1)} + p·v` |

这里“块”退化成单个 key（`Bc = 1`），所以当前块局部量 `m(x^{(k)}) = s`、`ℓ(x^{(k)}) = 1`，递推里的当前块校正因子 `e^{s-m_new}` 就直接是权重 `p`。这是 FA2 递推最干净的特例。

### Epilogue：延迟归一化（唯一一次除法）
```cuda
float inv = 1.f / l;                       // 循环结束后才求 1/ℓ
for (int d = 0; d < D; ++d) o[d] = acc[d] * inv;   // O = Õ / ℓ
```
对应 Step 4 的 `O = Õ^{(T)} / ℓ^{all}_T`。整个 kernel 只有这一个除法——**延迟归一化落地**。

### 为什么慢（且不打算修）
- 访存**完全非合并**：相邻线程处理相邻 query `i`，但它们同一时刻读的是各自行的同一个 `k[d]`——线程走的是“行”方向，warp 内 32 个线程的 `K` 地址相差整整一行，cache line 利用率极差。
- K、V 被**每个 query 重读一遍**（正是 IO 表里“单向量融合 160 B”的重复搬运缺陷）。
- 纯标量 FMA，Tensor Core 完全闲置。

`fa_cc_02` 修掉前两条（smem 分块 + KV 复用），Tensor Core 那条留给 `02` 文档。

---

## 2. `fa_cc_02_tiled.cu`：FA2 分块 + shared memory K/V

### 并行结构：Q 分块进外循环（FA2 的灵魂）
```cuda
static constexpr int kBr = 32, kBc = 32, kMaxD = 128;
dim3 grid((S + kBr - 1) / kBr, H, B);
fa_cc02_kernel<<<grid, kBr, ...>>>(...);   // block = kBr = 32 线程
```
`grid.x` 是 Q-tile 维、`grid.y=H`、`grid.z=B`。**一个 block 负责一个 `[b, h, Q-tile]`**，block 内 `kBr=32` 个线程、**每个线程仍然一行 query**：

```cuda
int qtile = blockIdx.x, h = blockIdx.y, b = blockIdx.z;
int t = threadIdx.x;              // tile 内的 query 行号
int i = qtile * kBr + t;         // 全局 query 行
bool active = i < S;             // 尾块越界保护
```
这就是 `00_principle.md` FA2 伪代码的 `for i in range(0, N, Br)` ——**Q 在外**。每个 query 行的 `(m, l, acc)` 会**全程驻留寄存器**，直到 KV 扫完才写回，正是 FA2「O 只写一次 HBM」。

### 状态常驻寄存器
```cuda
float qreg[kMaxD], acc[kMaxD];
for (int d = 0; d < D; ++d) { qreg[d] = active ? Q[idx4(b,i,h,d,S,H,D)] : 0.f; acc[d] = 0.f; }
float m = -INFINITY, l = 0.f;
```
`qreg` 把这一行 Q **一次性载入寄存器**（内层不再重读 Q），`acc/m/l` 是钉在 SRAM/寄存器里的 FA2 状态块。

### 内层：K/V 分块流过 shared memory
```cuda
__shared__ float sK[kBc * kMaxD];
__shared__ float sV[kBc * kMaxD];

int i0 = qtile * kBr;
int loopEnd = causal ? min(i0 + kBr - 1, S - 1) : S - 1;   // 本 tile 任一行需要的最后一个 key

for (int j0 = 0; j0 <= loopEnd; j0 += kBc) {
  // 协作把 K/V 块 [kBc, D] 合并搬进 smem
  for (int idx = t; idx < kBc * D; idx += blockDim.x) {
    int jj = idx / D, dd = idx % D, j = j0 + jj;
    sK[idx] = (j < S) ? K[idx4(b,j,h,dd,S,H,D)] : 0.f;
    sV[idx] = (j < S) ? V[idx4(b,j,h,dd,S,H,D)] : 0.f;
  }
  __syncthreads();
  ...
  __syncthreads();
}
```
对应 FA2 伪代码内层 `for j in range(0, N, Bc)`。相对 `fa_cc_01` 的两处进步：
1. **访存合并**：`K/V` 由 block 内 32 线程**协作、跨步搬运**进 smem（`idx += blockDim.x` 保证连续地址落到连续线程），cache line 用满。
2. **KV 复用**：一个 K/V 块搬进 smem 后，被 tile 内 32 个 query 行**共享**，不再 per-query 重读——干掉了 `fa_cc_01` 的 160 B 缺陷。

`loopEnd` 是 causal 下的块级剪枝：整个 Q-tile 里下标最大的行是 `i0+kBr-1`，它之后的 K 块谁都用不到，直接不搬。（这在 `02` 的张量核版里升级为“masked / unmasked 分段”。）

### 逐 key 的 online 更新（与 `fa_cc_01` 完全同构）
```cuda
if (active) {
  int jmax = causal ? i : S - 1;
  for (int jj = 0; jj < kBc; ++jj) {
    int j = j0 + jj;
    if (j > jmax) break;                          // 行级 causal mask
    float s = 0.f;
    for (int d = 0; d < D; ++d) s += qreg[d] * sK[jj*D + d];   // q·kⱼ，K 来自 smem
    s *= scale;
    float mnew = fmaxf(m, s);
    float c = __expf(m - mnew), p = __expf(s - mnew);
    l = l * c + p;
    for (int d = 0; d < D; ++d) acc[d] = acc[d]*c + p*sV[jj*D + d];   // V 来自 smem
    m = mnew;
  }
}
```
递推公式与 `fa_cc_01` **一字不差**，唯一区别是 `k/v` 从 HBM 指针换成了 smem 数组 `sK/sV`。这说明：**tiling 不改算法，只改数据从哪来**——`00_principle.md` 反复强调的“Online Softmax 的正确性与分块方式无关”，在这里得到代码级印证。

### Epilogue
```cuda
if (active) {
  float inv = 1.f / l;
  for (int d = 0; d < D; ++d) O[idx4(b,i,h,d,S,H,D)] = acc[d] * inv;   // O 只写一次
}
```
KV 全扫完后一次性归一化、一次性写 HBM——FA2 三大区别里的「延迟归一化」+「O 零换页」。

---

## 3. 两支内核 vs FA2 伪代码：对照表

| FA2 伪代码（`00_principle.md`） | `fa_cc_01` | `fa_cc_02` |
|---|---|---|
| `for i in range(0,N,Br)`（Q 外循环） | 一线程一 query（`Br` 退化到 1） | 一 block 一 Q-tile（`kBr=32`），Q 外循环 ✅ |
| `o_block/m/l` 常驻 SRAM | `acc/m/l` 在寄存器 | `acc/m/l` 在寄存器，全程不写回 ✅ |
| `for j in range(0,N,Bc)`（K/V 内循环） | 逐 key（`Bc`=1），K/V 从 HBM | K/V 分块入 smem，块被 tile 内 32 行复用 ✅ |
| `m_new=max; l_new=...; o=o·e^{Δm}+p̃·V` | 三行标量递推 | 同左，K/V 取自 smem |
| Epilogue `o/=l`，写 HBM 一次 | `o[d]=acc[d]/l` | `O[...]=acc[d]/l`，一次写 ✅ |
| causal 掩码 | `jmax=i`，`j<=jmax` | 块级 `loopEnd` 剪枝 + 行级 `j>jmax break` |

**核心结论**：`fa_cc_02` 已经是**语义完整的 FA2**——Q 外 / KV 内、smem 分块复用、online 递推、延迟归一化、causal 剪枝五件套齐全。它和工业级内核的差距**只剩执行层**：矩阵乘还在 CUDA-core 上逐元素 FMA，而不是喂给 Tensor Core。补上这最后一块，就是下一篇。

---

## 4. 下一步

`02_tensor_core_tinyfa.md`：把 `fa_cc_02` 里的两处 `for d: s += q*k` / `acc += p*v` 换成
`mma.sync m16n8k16` 两个 GEMM，把 smem 搬运换成 `cp.async` + `ldmatrix` + swizzle，
把标量 `(m, l)` 递推换成 fragment 级 warp-4 归约——算法不变，只换执行引擎，冲到 ~94–96% Dao FA2。
