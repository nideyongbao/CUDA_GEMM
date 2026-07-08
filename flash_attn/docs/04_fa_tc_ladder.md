# FA tensor-core 增量阶梯：fa_tc_01 → fa_tc_05（from-scratch raw-CUDA）

补上 FA 算子缺的那条"增量 tensor-core 阶梯"——**手写 `mma.sync.m16n8k16` + `ldmatrix(.trans)` + `cp.async`** 的 FA2 前向，一 delta 一级，和 GEMM 的 `tc_01→tc_06` 对称。终点参照是 vendored 的 TinyFA（CuTe，204 TFLOPS）。本梯为**我们自己手写**（canonical Ampere 布局，参考 CUTLASS/TinyFA/lubits.ch `flash_attention_from_scratch`），逐级对 `fa_cpu_ref` 验证通过。

## 阶梯（B2 H32 S4096 D128 fp16 非causal，A800 锁频 1410MHz）

| id | rung | 单一 delta | TFLOPS | vs 上一级 | %FA2(214) | 正确性 |
| ---: | --- | --- | ---: | ---: | ---: | :-: |
| 1 | fa_tc_01_base | 裸 `mma.sync`+`ldmatrix`，同步拷贝，无 swizzle，`expf` | 33.3 | — | 16% | PASS |
| 2 | **fa_tc_02_swizzle** | **+ smem XOR-swizzle（消 ldmatrix bank conflict）** | **104.9** | **+3.1×** | 49% | PASS |
| 3 | fa_tc_03_cpasync | + `cp.async` GMEM→SMEM | 134.9 | +29% | 63% | PASS |
| 4 | fa_tc_04_exp2 | + `exp2f` + 折叠 log2e | 136.0 | +1% | 63% | PASS |
| 5 | **fa_tc_05_occ** | **+ `__launch_bounds__` 抬占用率 12.5%→18.4%** | **146.3** | **+8%** | **68%** | PASS |
| — | TinyFA（CuTe 终态，参照） | 完整多级流水线+CuTe | 204.8 | — | 93% | PASS |

寄存器：base 184 → fa_tc_05 168 regs / **0 spill**（D=128 靠**流式 operand 载入**——K/V 每 k16-tile 现载现用，全驻留会撑爆 RF）。所有 rung causal + 非causal、S≤1024 全 PASS（对拍 fp32 CPU 参考注意力）。

## 头号一课：swizzle 的 +3× 为什么发生——SASS 与 ncu 两层缺一不可

`fa_tc_02` 只加了一个 XOR-swizzle，却 **3.03×**。用我们的两层分析工具拆开看，会得到一个反直觉但极重要的结论：

**SASS 层（`common/sass_count.sh` + `sass_compare.py`，静态、无 GPU）**：swizzle 之后**指令不减反增 +96 条**（`+32 LOP3` = XOR 本身、`+58 IMAD` = 地址计算），而 **`HMMA` 恒为 128**（张量核工作量不变）。

| Instr | base | swizzle | Δ |
| --- | ---: | ---: | ---: |
| HMMA（张量核） | 128 | 128 | 0 |
| LOP3（XOR swizzle） | 25 | 57 | +32 |
| IMAD（地址） | 254 | 312 | +58 |
| TOTAL | 1504 | 1600 | **+96** |

**ncu 层（动态）**：真相在这里——base **每 kernel 有 1.179 亿次 shared-memory bank conflict**（4.2M 次 shared load 全被 ldmatrix 的固定访问模式打成冲突），`Compute(SM) 8.8%`、Duration 1.54ms；swizzle 把 bank conflict **清零**，`Compute(SM) 26.5%`（3×）、Duration 527µs。

> **结论**：swizzle **指令更多、却快 3×**——加速 100% 来自消除 bank conflict，这**在指令计数里完全看不见**，只在 ncu 的 bank-conflict 计数里现形。所以"改动是否有效"必须**两层一起看**：SASS 说明你改了哪些机器指令（这里：多了 XOR/地址算），ncu 说明它为什么快（这里：消了 1.18 亿次冲突）。这正是把 SASS 层（借自 `flash_attention_from_scratch`）补进我们 ncu-only 方法学的意义——见 `common/sass_*.sh`、`common/ptxas_usage.sh`。
>
> 附：为什么 swizzle 在 FA 上收益（3×）比在 GEMM 上大——FA 内层两个 GEMM（QKᵀ、PV）都是 **ldmatrix-heavy、SMEM 带宽 bound**，一次消冲突几乎放开整条张量核流水；GEMM 的 smem 复用模式没这么极端。XOR-swizzle 本体极小（`fa_tc.cuh::swz`：`cf ^ (row & 7)`，给每行不同 bank rotation，让 cp.async 存与 ldmatrix 取都 128-bit conflict-free）。

## 后续 rung（03/04/05）

- `fa_tc_03 cp.async`：GMEM→SMEM 用 `cp.async.cg` 异步、不占寄存器搬运（+29%）。本级只单缓冲、发了即等（无真正 overlap）。
- `fa_tc_04 exp2`：`exp2f`(MUFU.EX2)+折叠 `1/√d·log2e`（+1%）；张量核 bound 下 softmax 非瓶颈。
- **`fa_tc_05 occupancy`**：`__launch_bounds__(128, 3)` 强制每 SM ≥3 block → 寄存器 184→168、占用率 **12.5%→18.4%** → **+8%（136→146）**。这直接印证下面的 ncu 诊断——瓶颈是延迟/占用率。

## 为什么止于 146 TFLOPS（68% FA2）——ncu 诊断 + 与 flash_attention_from_scratch 16 级的关系

`flash_attention_from_scratch`（fafs）用 **16 级**打到官方 FA2 的 **99.2%**（A100）。三段：**①1-2 内存布局**（base→swizzle，15.8%→72.6%）**②3-7 流水线/重叠/autotune**（→80.3%）**③8-16 SASS 指令级微优化**（→99.2%，"16 Static GMEM Stride"=把行 stride 变编译期常量、折掉地址 IMAD）。

**本梯 5 级 vs fafs 16 级的对照（%-of-FA2，本机 A800 FA2=214.5）**：

| 本梯 | %FA2 | ≈ fafs | fafs %FA2 |
| --- | ---: | --- | ---: |
| fa_tc_01 base | 16% | 1 base | 15.8% ✅ 吻合 |
| fa_tc_02 swizzle | 49% | 2 swizzle | 72.6% |
| fa_tc_03 cpasync | 63% | 3 eager-load | 77.6% |
| fa_tc_05 occ | **68%** | 7 autotune | 80.3% |
| （未实现） | — | **4,5,8–16** | →99.2% |

**base 与 fafs 几乎完全吻合（16% vs 15.8%）——证明这是同一个正确起点、不是 bug；分歧全来自"缺后面的级"。**

**ncu 诊断（fa_tc_04，未 occ 调优时）**：Achieved Occupancy **12.4%**（184 寄存器→2 block/SM）、Issued Warps/Sched **0.38**、Compute(SM) 39% / Memory 44%（**都没打满**）、Tensor pipe 空等 60% → **典型 latency-bound + 低占用率**。`fa_tc_05` 用一行 `__launch_bounds__` 把占用率抬到 18.4%、issue 0.47、Compute 48.9% → +8%，**验证了诊断**。

**到 FA2(214)/TinyFA(200) 剩余的 32% 差距 = fafs 的 rung 4/5/8–16 我全未实现**：
1. **真·多级流水线双缓冲**（rung 3-5）：K/V 双 smem buffer，prefetch tile j+1 与 compute tile j 重叠 → 藏 GMEM+ldmatrix 延迟（我现在是 `load→__syncthreads→compute` 全串行、零重叠）。
2. **d_head 分块降寄存器**（rung 15）：把 O 累加器（这里 64 个 f32）切成 64-块 → 寄存器再降 → 占用率再升。
3. **SASS 指令级微优化**（rung 8-16）：削 IMAD.MOV/LOP3/CS2R、encoded swizzle、static GMEM stride——HMMA 恒定、只削整数/开销指令。**这一层正好用我们的 `common/sass_compare.py` 复现其证据**。

**能否直接借用 fafs 的 16 个 tc 示例？** 作**参考/续梯蓝本/SASS 证据**——强烈推荐；**直接 vendor 进 Makefile**——不建议：① 无 LICENSE；② build 绑死 torch/pybind + python codegen（非干净单 `.cu`）；③ 拖进 ~2750 行 mini-CuTe 抽象；④ rung 8-16 是针对其自身代码的微优化、非通用可移植 delta。故本仓库**手写自己的 raw 梯**（base 已吻合、swizzle/occ 已验证），把 fafs 当续梯参考、TinyFA 当 CuTe 终态。

## 与 GEMM 阶梯的对称

| | GEMM tensor_core | FA tensor_core（本梯） |
| --- | --- | --- |
| 起点 | tc_01 WMMA naive | fa_tc_01 raw mma.sync（已在 tc_06 高度起步） |
| 关键跳 | tc_06 ldmatrix 消 bank 冲突（+3.2×） | **fa_tc_02 swizzle 消 bank 冲突（+3.1×）** |
| 抽象 | 手写 PTX | 手写 PTX（+ CuTe 终态 = TinyFA） |
| 天花板 | tc_06 150.3T（48% 峰） | fa_tc_05 146.3T（68% FA2）/ TinyFA 205T |

FA 现在有和 GEMM 一样的完整故事：**标量 scaffold（fa_cc_*）→ raw-mma 增量阶梯（fa_tc_01..05）→ CuTe 终态（TinyFA）**；且诚实标注了到 FA2/TinyFA 的剩余差距与其 ncu 根因和续梯路线。

## 运行

```bash
cd flash_attn && make fatc
./build/tensor_core/fa_tc_verify <id> [B H S causal]   # 对拍 CPU 参考，id 1..5
./build/tensor_core/fa_tc_bench  <id> [B H S causal]   # TFLOPS
# 指令级 diff（相邻 rung）：
../common/sass_count.sh build/tensor_core/fa_tc_bench 'Lb0ELb0ELb0ELb0E' > /tmp/a.txt   # base
../common/sass_count.sh build/tensor_core/fa_tc_bench 'Lb1ELb0ELb0ELb0E' > /tmp/b.txt   # swizzle
python3 ../common/sass_compare.py /tmp/a.txt /tmp/b.txt --before-name base --after-name swizzle
```
