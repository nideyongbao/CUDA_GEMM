# FA tensor-core 增量阶梯：fa_tc_01 → fa_tc_04（from-scratch raw-CUDA）

补上 FA 算子缺的那条"增量 tensor-core 阶梯"——**手写 `mma.sync.m16n8k16` + `ldmatrix(.trans)` + `cp.async`** 的 FA2 前向，一 delta 一级，和 GEMM 的 `tc_01→tc_06` 对称。终点参照是 vendored 的 TinyFA（CuTe，204 TFLOPS）。本梯为**我们自己手写**（canonical Ampere 布局，参考 CUTLASS/TinyFA/lubits.ch `flash_attention_from_scratch`），逐级对 `fa_cpu_ref` 验证通过。

## 阶梯（B2 H32 S4096 D128 fp16 非causal，A800 锁频 1410MHz）

| id | rung | 单一 delta | TFLOPS | vs 上一级 | 正确性 |
| ---: | --- | --- | ---: | ---: | :-: |
| 1 | fa_tc_01_base | 裸 `mma.sync`+`ldmatrix`，同步拷贝，无 swizzle，`expf` | 34.7 | — | PASS |
| 2 | **fa_tc_02_swizzle** | **+ smem XOR-swizzle（消 ldmatrix bank conflict）** | **105.2** | **+3.03×** | PASS |
| 3 | fa_tc_03_cpasync | + `cp.async` GMEM→SMEM | 119.2 | +13% | PASS |
| 4 | fa_tc_04_exp2 | + `exp2f` + 折叠 log2e | 121.3 | +2% | PASS |
| — | TinyFA（CuTe 终态，参照） | 完整多级流水线+CuTe | 204.8 | — | PASS |

寄存器：184 regs / **0 spill**（D=128 靠**流式 operand 载入**——K/V 每 k16-tile 现载现用，全驻留会撑爆 RF）。所有 rung causal + 非causal、S≤1024 全 PASS（对拍 fp32 CPU 参考注意力）。

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

## 后续 rung（03/04）与到 TinyFA 的差距

- `fa_tc_03 cp.async`（+13%）：GMEM→SMEM 用 `cp.async.cg` 异步、不占寄存器搬运；本级只做单缓冲，未做多级流水，故增益温和。
- `fa_tc_04 exp2`（+2%）：`exp2f`(MUFU.EX2)+把 `1/√d·log2e` 折成一个常数；DRAM/张量核 bound 下 softmax 非瓶颈，故增益小（但零成本、对齐 FA 内层口径）。
- **121 → 205（TinyFA）的剩余差距**是**进阶级**：多级 `cp.async` 流水线（rotating buffers）、SMEM→RF 双缓冲、`d_head` 分块降寄存器压力、以及 SASS 指令级微优化（削 IMAD.MOV/CS2R——见 `flash_attention_from_scratch` rung 8–16，用我们的 SASS-diff 工具可复现其"HMMA 恒定、只削整数指令"的证据）。这些属于"可读 CUDA 的最后一截"，本梯诚实止步于**单缓冲 + 手写 swizzle**（≈ 59% TinyFA / 55% FA2），把 TinyFA 留作 CuTe 终态标杆。

## 与 GEMM 阶梯的对称

| | GEMM tensor_core | FA tensor_core（本梯） |
| --- | --- | --- |
| 起点 | tc_01 WMMA naive | fa_tc_01 raw mma.sync（已在 tc_06 高度起步） |
| 关键跳 | tc_06 ldmatrix 消 bank 冲突（+3.2×） | **fa_tc_02 swizzle 消 bank 冲突（+3.0×）** |
| 抽象 | 手写 PTX | 手写 PTX（+ CuTe 终态 = TinyFA） |
| 天花板 | tc_06 150.3T（48% 峰） | fa_tc_04 121T / TinyFA 205T |

这样 FA 就有了和 GEMM 一样的完整故事：**标量 scaffold（fa_cc_*）→ raw-mma 增量阶梯（fa_tc_01..04）→ CuTe 终态（TinyFA）**。

## 运行

```bash
cd flash_attn && make fatc
./build/tensor_core/fa_tc_verify <id> [B H S causal]   # 对拍 CPU 参考，id 1..4
./build/tensor_core/fa_tc_bench  <id> [B H S causal]   # TFLOPS
# 指令级 diff（相邻 rung）：
../common/sass_count.sh build/tensor_core/fa_tc_bench 'Lb0ELb0ELb0ELb0E' > /tmp/a.txt   # base
../common/sass_count.sh build/tensor_core/fa_tc_bench 'Lb1ELb0ELb0ELb0E' > /tmp/b.txt   # swizzle
python3 ../common/sass_compare.py /tmp/a.txt /tmp/b.txt --before-name base --after-name swizzle
```
