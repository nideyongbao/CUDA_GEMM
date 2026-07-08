# FA tensor-core 增量阶梯：fa_tc_01 → fa_tc_06（from-scratch raw-CUDA） + fafs 实测参考

补上 FA 算子缺的那条"增量 tensor-core 阶梯"——**手写 `mma.sync.m16n8k16` + `ldmatrix(.trans)` + `cp.async`** 的 FA2 前向，一 delta 一级，和 GEMM 的 `tc_01→tc_06` 对称。终点参照是 vendored 的 TinyFA（CuTe，204 TFLOPS）。本梯为**我们自己手写**（canonical Ampere 布局，参考 CUTLASS/TinyFA/lubits.ch `flash_attention_from_scratch`），逐级对 `fa_cpu_ref` 验证通过。

## 阶梯（B2 H32 S4096 D128 fp16 非causal，A800 锁频 1410MHz）

| id | rung | 单一 delta | TFLOPS | vs 上一级 | %FA2(214) | 正确性 |
| ---: | --- | --- | ---: | ---: | ---: | :-: |
| 1 | fa_tc_01_base | 裸 `mma.sync`+`ldmatrix`，同步拷贝，无 swizzle，`expf` | 33.3 | — | 16% | PASS |
| 2 | **fa_tc_02_swizzle** | **+ smem XOR-swizzle（消 ldmatrix bank conflict）** | **104.9** | **+3.1×** | 49% | PASS |
| 3 | fa_tc_03_cpasync | + `cp.async` GMEM→SMEM | 134.9 | +29% | 63% | PASS |
| 4 | fa_tc_04_exp2 | + `exp2f` + 折叠 log2e | 136.0 | +1% | 63% | PASS |
| 5 | **fa_tc_05_occ** | **+ `__launch_bounds__` 抬占用率 12.5%→18.4%** | **146.3** | **+8%** | **68%** | PASS |
| 6 | fa_tc_06_pipeline | + 双缓冲 K/V 流水（**反例:smem 翻倍→占用率↓,净亏**） | 137.7 | −6% | 64% | PASS |
| — | fafs 峰值(Br=128,实测) | autotune 更大 tile | 159.9 | — | 75% | — |
| — | TinyFA（CuTe 终态，参照） | 完整多级流水线+CuTe | 204.8 | — | 93% | PASS |

寄存器：base 184 → fa_tc_05 168 regs / **0 spill**（D=128 靠**流式 operand 载入**——K/V 每 k16-tile 现载现用，全驻留会撑爆 RF）。所有 rung causal + 非causal、S≤1024 全 PASS（对拍 fp32 CPU 参考注意力）。
> **`fa_tc_06` 是诚实的反例级**:朴素双缓冲把 smem 48→80KB → 占用率 3→2 block/SM,重叠收益盖不过占用率损失,**净亏 6%**——与 fafs 的 buffer 级只 +0.5% 一致(**双缓冲要配合 d_head 分块降寄存器才划算**)。像 GEMM 的 bank-conflict 反例级一样,教"不是每个优化都赢"。

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

**我们把 fafs 实际编译跑在了本机 A800 上**（见 `reference/`），据此对账——结论:**我们的手写阶梯与 fafs 的进阶逐级吻合**。

| 特性层（本机 A800，B2 H32 S4096） | fafs | 我们 fa_tc | 结论 |
| --- | ---: | ---: | --- |
| base（无 swizzle） | 34.9 | 33.3 | ✅ 吻合 |
| swizzle | 139.6（async+swz） | 134.9（fa_tc_03 cpasync+swz） | ✅ 吻合（±3%）|
| Br=64 进阶封顶 | 145.3 | **146.3（fa_tc_05）** | ✅ **吻合/略高** |
| double-buffer | 145.3（+0.5%） | 137.7（fa_tc_06 净亏） | 都印证 buffer 在此配置**边际/无用** |
| **autotune 峰值** | **159.9（Br=128）** | —（未实现 2-M-tile） | fafs +10% |

> **更正一个早先的误比**:曾说"fa_tc_02 swizzle 49% << fafs 72.6%"——那是拿我们的 **sync**-swizzle 比 fafs 的 **async**-swizzle。对齐 async 后（我们 fa_tc_03 vs fafs rung2）是 **134.9 vs 139.6，吻合**。**没有"神秘 gap":我们的 raw 梯在同配置封顶都在 ~145–146。**

**ncu 诊断（fa_tc_04）**:Occupancy **12.4%**（184 寄存器→2 block/SM）、Issued/Sched **0.38**、Compute 39%/Memory 44%（都没打满）→ latency-bound。`fa_tc_05` 一行 `__launch_bounds__` 抬占用率到 18.4% → +8%，验证诊断。

**到 fafs 峰值(160)/TinyFA(200)/FA2(214) 的差距 = 三层，已全部定位**（详见 `reference/fafs_a800_results.md`）:
1. **Br=128**（autotune,+10%→160）:4warps×2-M-tile 更大 Q tile;fafs 靠它拿 A800 峰值。**8-warp×16-row 变体实测崩到 22T（MINBLK 与寄存器打架）——必须 2-M-tile 重构。**
2. **CuTe 级 tiling/多级流水**（160→200 TinyFA）:超出可读手写 raw-CUDA 的性价比区。
3. **官方 FA2 warp-specialization**（200→214）。
> **"99.2% FA2" 是 A100 + fafs 自身 benchmark 配置**、对 A100 FA2(186) 算的;**在 A800 本配置下 fafs 自己也只到 75% FA2(160)**。所以"本地没复现 99.2%"**不是 bug**,而是机器/配置口径差异 + 我们止于 Br=64 可读阶梯。

**能否直接借用 fafs 的 tc 示例?** ① 作**性能参考**——✅ 已做（编译跑在 A800,`reference/bench_fafs.py` 可复现全曲线;修的只是 `-lcuda` stub 路径 + torch lib 的 `LD_LIBRARY_PATH`,kernel 本身 0 spill 干净编译）;② **直接 vendor 进本仓库 Makefile**——不建议:无 LICENSE、build 绑死 torch/pybind+codegen、拖 ~2750 行 mini-CuTe。故:**手写自己的可读 raw 梯（已与 fafs 逐级对齐）,fafs 当实测性能参考,TinyFA 当 CuTe 终态。**

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
