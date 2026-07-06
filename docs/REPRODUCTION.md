# H20 复现报告（Reproduction Report）

把三个主流算子（GEMM / softmax / flash-attention）从 A800（Ampere, sm_80）**完整迁移到
NVIDIA H20（Hopper, sm_90a）**，全部适配、重写、测试、给出结论。本文是汇总口径。

## 1、环境（2026-07-06 实测）

| 项 | 值 |
| --- | --- |
| GPU | **NVIDIA H20**（Hopper，CC 9.0，sm_90a，78 SM） |
| 显存 / 带宽 | 96 GB **HBM3** / **~4000 GB/s** |
| L2 / Shared | 60 MB / 228 KB per SM |
| FP32 峰值 | ~40 TFLOPS（CUDA core，78·128·2·1.98GHz） |
| BF16/FP16 张量核峰值 | **148 TFLOPS**（大幅裁剪版 Hopper） |
| FP8(e4m3) 张量核峰值 | **296 TFLOPS**（= 2× BF16） |
| CUDA / Driver | 12.8（V12.8.93） / 570.195.03 |
| 编译 | `nvcc -arch=sm_90a -O3` |
| 锁频 | SM 1980 MHz（`run_all.sh` 自动锁） |

> **H20 的一句话画像：算力弱、带宽大。** 148 TFLOPS 张量核（H100 的 ~1/7）配 4 TB/s
> HBM3（与 H100 同级）。算术强度交叉点极低（148T/4TB ≈ 37 FLOP/byte，bf16）——**计算受
> 限**算子（GEMM、FA）被 148T 张量核卡住，FP8 是把算力翻倍的杠杆；**访存受限**算子
> （softmax）则在 4 TB/s 上尽情起飞。本仓库三个算子恰好把这条主线走全。

## 2、为适配 H20 做的改动

| 算子 | 改动 |
| --- | --- |
| **gemm** | `Makefile` 默认 `sm_80`→`sm_90a`、`TC_HOPPER 0`→`1`；由此 **Hopper 独占的 tc_04（WGMMA+TMA）/ tc_05（FP8 WGMMA）在 H20 上被启用**（A800 曾用 `-DNO_HOPPER` 跳过）。kernel 源码与已验证的 H20 GEMM 参考逐字节一致。张量核 tc_03/tc_06 注释里的 A800 数字更新为 H20 实测。|
| **softmax** | `Makefile` → `sm_90a`；`bench.cu` 里写死的 A800 带宽 **2039 GB/s → 4000 GB/s**（并加 `PEAK_BW_GBS` 环境变量覆盖，不再钉死单卡）。kernel 访存无关，直接受益于 4 TB/s。|
| **flash_attn** | `Makefile` → `sm_90a`；**删除 vendored TinyFA（CuTe/CUTLASS，sm_80）**，改为**从零手写的 Hopper WGMMA + TMA 前向内核** `include/fa_hopper.cuh`（寄存器常驻 online softmax，两次矩阵乘都上 WGMMA，K/V 走 TMA）。cuda_core FP32 脚手架仅重编。|
| **基础设施** | `run_all.sh`（选卡/锁频 1410→**1980**/命名 a800→h20/张量核加测 tc_04/05）；根 `Makefile`、`common/{summary,run_ncu}.py/sh` 更新。`common/gpu_specs.py` 已含 H20（148T/78SM/4000GB/s）。|
| **A800 物料** | 按「H20-only」全部删除（`gemm/docs/a800/`、`*a800*` 基线、TinyFA 及其外部基线）。|

## 3、结论矩阵

| 交付项 | 状态 | 依据 |
| --- | --- | --- |
| 全量 build（3 算子，sm_90a） | ✅ | `make all` 全绿，12 个 bench/verify 二进制 |
| 正确性对拍（3 算子） | ✅ 全 PASS | GEMM vs cuBLAS(allclose 1e-2)；softmax vs CPU double；FA vs CPU 参考注意力(2e-2) |
| GEMM 性能 + WGMMA/FP8 | ✅ 已闭环 | cuBLAS BF16 90.6%，手写 WGMMA 81.5%，**FP8 226 TFLOPS**，见 §GEMM |
| softmax 性能 | ✅ 已闭环 | 好内核达 **77–89% of 4 TB/s**，见 §softmax |
| FA 手写 Hopper 内核 | ✅ 已闭环 | **122 TFLOPS = 82.8% 峰**，见 §FA |
| 全量 ncu（`--set full`） | ✅ 已采 | 9 份 `*/baselines/*.details.txt`（锁频卡） |
| 逐级分析 + 结论文档 | ✅ | `gemm/softmax/flash_attn 各自 docs/`、本文、`CURRICULUM.md`、`README.md` |

快照：`result/20260706_174822/`（`00_build`/`01_verify`/`02_bench` 日志），干净口径同
`docs/clean_{bench,verify}_h20_20260706.txt`。

## §GEMM — 计算受限，FP8 是杠杆

4096³，SM 1980 MHz，best-of（warmup2/repeat10）：

**CUDA core（FP32，天花板 ~40 TFLOPS）**：naive 3.35T → double_buffer **23.2T**
（= cuBLAS SGEMM 28.0T 的 82.8%，≈ FP32 峰值 58%）。`warptile` 标量版 11.9T 反低于
`vectorized` 21.1T——**负优化教材**，须配 float4 才追回。

**张量核（BF16 输入，天花板 148 TFLOPS）**：

| id | kernel | GFLOPS | 占 148T |
| --- | --- | ---: | ---: |
| tc_01 | WMMA_naive | 16946 | 11.5% |
| tc_02 | WMMA_smem | 30123 | 20.4% |
| tc_03 | WMMA_pipe（wmma API 天花板） | 38659 | 26.1% |
| tc_06 | mma.sync + ldmatrix + cp.async（warp 级 PTX） | 75332 | 50.9% |
| tc_04 | **手写 WGMMA + TMA**（warpgroup 级，Hopper 原生） | 120646 | **81.5%** |
| tc_04 | cuBLAS BF16（同 bench 参考） | 133802 | 90.4% |
| **tc_05** | **WGMMA FP8（e4m3）** | **226026** | **76.4% of 296T = 152.7% of 148T** |

阶梯清清楚楚：`WMMA(API) → mma.sync(warp) → WGMMA(warpgroup)+TMA`，手写到 81.5%；
**FP8（tc_05）把算力天花板从 148T 抬到 296T，实测 226 TFLOPS——在算力弱的 H20 上收回
算力的唯一杠杆。** ncu：tc_04/tc_05 的 **DRAM 仅 3–10%**（4 TB/s 闲置），占用率 7–8% 却
拿到 81% MFU——**WGMMA 异步，低占用不碍事**。

## §softmax — 访存受限，吃满 4 TB/s

有效带宽 = 读 x + 写 y = `2·M·N·4` 字节 / 时间，占 4000 GB/s 峰值：

| kernel | @8192² | @16384²（长行） |
| --- | ---: | ---: |
| sc_01 naive | 2.6% | — |
| sc_02 block_reduce | 62.3% | 41.7% |
| sc_03 warp_shuffle | 41.6% | — |
| sc_04 vectorized | **79.1%** | 61.1% |
| sc_05 online | 77.0% | 61.3% |
| sc_06 resident | 64.4% | **88.9%** |

屋顶是 HBM 带宽，不是 FLOPs。好内核在 H20 的 4 TB/s 上到 **77–89% 峰**；naive（多趟/未
合并）只 2.6%。**single-pass online（sc_05）与寄存器常驻（sc_06）靠最小化 DRAM 流量取胜；
sc_06 在长行上单次 DRAM 读，达 88.9%。** ncu：sc_05 DRAM 83.1% / 占用 90%，sc_06 DRAM
88.0%。online-softmax 正是通往 FlashAttention 的桥。

## §FA — 综合：两次 GEMM + online softmax，融合于 SRAM

手写 Hopper WGMMA+TMA 内核（`include/fa_hopper.cuh`），SM 1980 MHz：

| shape | dtype | causal | TFLOPS | 占 148T |
| --- | --- | --- | ---: | ---: |
| B2 H32 S4096 D128 | bf16 | 否 | **122.5** | **82.8%** |
| B2 H32 S4096 D128 | bf16 | 是 | 113.8（有效） | 76.9% |
| B4 H32 S8192 D128 | bf16 | 是 | 127.9（有效） | 86.4% |
| B2 H16 S2048 D64 | bf16 | 否 | 120.1 | 81.1% |

**正确性**：`D∈{64,128}` × bf16/fp16 × causal/非 × 多 KV-tile，全部 PASS（对拍 fp32 CPU
参考，`S≤512`；更大 `S` 同代码路径）。**性能 80–86% 峰**，与库（SDPA/FA2 ~90–96% MFU）
同一量级。关键设计：**寄存器常驻 online softmax**（`S`/`P` 不落 smem）、**P 直接当作
`P·V` 的 A-fragment**（matmul-1 累加器布局 = matmul-2 A 操作数布局）、**两次矩阵乘都上
WGMMA**、K/V 走 **TMA**、**单次 grid 启动**（把 35%→83% 的关键，见 `flash_attn/docs/02`）。
ncu：DRAM 仅 1.3%（计算受限），占用 12% 却 83% MFU（异步 WGMMA）。

**cuda_core FP32 脚手架**（教学，非性能路径）：fa_cc_01 0.44 / fa_cc_02 0.65 TFLOPS——
张量核路径比它快 ~180×，实证「注意力的 matmul 必须上张量核」。

## 4、H20 统一结论

一张卡三个算子，把「算力弱、带宽大」讲透（ncu DRAM% 一栏即证）：

| 类别 | 算子 | ncu DRAM% | 瓶颈 | H20 之道 |
| --- | --- | ---: | --- | --- |
| **计算受限** | GEMM / FA | 1–10% | 148T 张量核 | 打满张量管线；**FP8** 把天花板翻到 296T |
| **访存受限** | softmax | 60–88% | 4 TB/s HBM | 最小化 DRAM 趟数；4 TB/s 让它飞 |

- 计算受限算子上，4 TB/s **几乎闲置**（DRAM<11%）——H20 的大带宽对 GEMM 无用武之地，
  真正的墙是被砍到 148T 的张量核。这也解释了 FP8 为何在 H20 上格外关键。
- 访存受限算子上，张量核**完全闲置**，4 TB/s 才是屋顶——H20 富余的带宽让 softmax/
  归一化/逐元素类算子（推理解码、MoE、norm）如鱼得水。
- **异步 WGMMA 让低占用率也能高 MFU**（FA 12% 占用→83% MFU，tc_04 7.7%→81.5%）——
  这条 Hopper 经验在带宽富余的 H20 上被放大：矩阵乘从不挨饿。

## 5、一条命令复现

```bash
bash run_all.sh --ncu     # 自动选空闲 H20、锁 1980MHz、build+verify+bench+ncu，快照 result/<ts>/
```

产物：`result/<ts>/{00_build,01_verify,02_bench,03_ncu}.log` + 各算子
`baselines/*.details.txt`。无 sudo 时用 `bash run_all.sh --no-lock`（计时略抖，量级不变）。

## 6、诚实边界

- FA CPU 参考是 `O(B·H·S²·D)` 标量，正确性只对拍到 `S=512`（=8 个 KV tile，已覆盖 online
  softmax 跨 tile 与 causal 全逻辑）；`S=4096` 只计时 GPU。
- 手写 FA 是**最小可用正确**版：smem 双缓冲但同步消费（TMA 与 WGMMA 未重叠），无 warp
  specialization / ping-pong，head dim 限 `{64,128}`、仅 MHA。追平库的最后 ~10–15% 即这些，
  见 `flash_attn/docs/02` §7。
- tc_05 FP8 用 e4m3，`verify` 对 CPU double 的 `max_abs≈2.9e-2`（512³）属 FP8 量化正常范围。
