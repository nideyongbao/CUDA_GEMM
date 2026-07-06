# cuda-ops-h20 — 主流算子的 CUDA 实现（H20 / Hopper sm_90a，前向）

在 **NVIDIA H20（Hopper, CC 9.0, sm_90a, 78 SM）** 真机上，用**编号、逐级 ncu 验证的阶梯**从零讲清三个主流算子的 CUDA 前向实现。每个算子是一个**自包含的 CUDA_GEMM 风格模块**（`kernels/{cuda_core,tensor_core}/` + `include/` + `docs/` + `baselines/`），有统一的 `bench <id>` / `verify <id>` 入口、统一的锁频基准与对拍方法。

> **H20 的一句话画像：算力弱、带宽大。** BF16 张量核 **148 TFLOPS**（H100 的 ~1/7）、FP8 **296 TFLOPS**、FP32 ~40 TFLOPS；HBM3 **~4000 GB/s**（与 H100 同级）。于是**计算受限**算子（GEMM、FA）被 148T 张量核卡住、**FP8 是收回算力的杠杆**；**访存受限**算子（softmax）在 4 TB/s 上尽情起飞。三个算子恰好把这条主线走全。

- **只做前向（forward-only）**，不含反向。目标是把"算子怎么在 GPU 上从 naive 一步步优化到接近硬件峰值"讲透。
- **三个算子，一条主线**：`gemm/`（快矩阵乘 + Hopper WGMMA/TMA/FP8 张量核）→ `softmax/`（快归约 + online-softmax）→ `flash_attn/`（两次 GEMM + online softmax 融合于 SRAM，**手写 Hopper WGMMA+TMA**）。
- **每一级都可复现、可对拍、可 profile**：`verify` 先证正确，`bench` 锁频测性能，`--ncu` 逐级抓瓶颈。

| 算子 | cuda_core 阶梯 | tensor_core | 性能口径 | 正确性对拍 |
| --- | --- | --- | --- | --- |
| **gemm** | `01_naive` … `10_doublebuffer`（FP32/BF16） | `tc_01 wmma`→`tc_03`，`tc_06 mma.sync`，**`tc_04 WGMMA+TMA`**，**`tc_05 FP8`** | GFLOPS / %峰值(148T/296T) / %cuBLAS | vs cuBLAS（allclose 1e-2） |
| **softmax** | `sc_01 naive` → `sc_05 online` → `sc_06 resident`（online = FA 的桥） | —（访存受限归约，无需张量核） | 有效 HBM 带宽 GB/s（÷4000） | vs CPU double 参考 |
| **flash_attn** | `fa_cc_01 stream`、`fa_cc_02 tiled`（from-scratch fp32 FA2 脚手架） | **手写 Hopper WGMMA+TMA 前向**（`include/fa_hopper.cuh`，80–86% 峰） | TFLOPS | vs fp32 CPU 参考注意力（= SDPA 数学定义） |

---

## 目录布局（算子为主，operator-major）

```
cuda-ops-h20/
├── Makefile                 # 顶层派发：make all → gemm + softmax + flash_attn
├── run_all.sh               # 一键：选空闲 GPU、锁 1980MHz、build+verify+bench(+ncu)，快照到 result/<ts>/
├── common/                  # 跨算子共享：gpu_specs.py(算力事实表, 含 H20) summary.py run_ncu.sh
│
├── gemm/                    # 算子①：GEMM —— 快矩阵乘 + Hopper WGMMA/TMA/FP8 张量核
│   ├── kernels/
│   │   ├── cuda_core/       # 01_naive .. 10_doublebuffer（FP32）+ bf16_cudacore + 4 个 bench/verify 驱动
│   │   └── tensor_core/     # tc_01/02/03 wmma · tc_06 mma.sync · tc_04 WGMMA+TMA · tc_05 FP8（全部在 H20 编译运行）
│   ├── include/             # common.h · kernels.h · tc_common.cuh · 06_autotuning.cuh · bf16.h
│   ├── docs/                # 00 硬件前置 · 01 性能方法论 · README（H20 复现总结）
│   └── baselines/           # GROUNDTRUTH_cuda_gemm_h20.txt + {cuda_core,tensor_core}/*.details.txt（ncu）
│
├── softmax/                 # 算子②：softmax —— 快归约 + online-softmax（FA 的桥）
│   ├── kernels/cuda_core/   # sc_01 naive · sc_02 block · sc_03 warp · sc_04 vectorized · sc_05 online · sc_06 resident
│   ├── include/softmax.h    # 声明 + CPU double 安全 softmax 参考
│   └── baselines/           # ncu details（run_all.sh --ncu 采）
│
└── flash_attn/              # 算子③：FlashAttention —— 两次 GEMM + online softmax 融合于 SRAM
    ├── kernels/
    │   ├── cuda_core/       # fa_cc_01 stream · fa_cc_02 tiled（from-scratch fp32 FA2 脚手架）
    │   └── tensor_core/     # bench/verify 驱动 → 调 include/fa_hopper.cuh 的手写 Hopper 前向
    ├── include/             # fa_common.h(FLOP 模型 + CPU 参考注意力) · fa_cc.h · fa_hopper.cuh(手写 WGMMA+TMA 内核)
    └── baselines/           # ncu details（fa_tc_hopper_wgmma.details.txt）
```

每个算子目录都能**独立 `make` / `bench` / `verify`**；顶层 `Makefile` 与 `run_all.sh` 只是把三者串起来。

---

## 课程主线：GEMM → softmax → FA，以及"基元交接（primitive handoff）"

三个算子不是并列的三个例子，而是**后一个复用前一个交付的基元**，最后一课只新增一件事——**融合（FUSION）**。

```
   GEMM                         softmax                       FlashAttention
 ┌────────────────┐          ┌────────────────┐          ┌──────────────────────────┐
 │ 快矩阵乘        │          │ 快归约          │          │  = 两次 GEMM              │
 │ (CUDA core)    │          │ (block/warp     │          │      (QKᵀ, PV, 用张量核)   │
 │ 分块+向量化+双缓冲│          │  reduce)        │          │  + online softmax(夹中间) │
 │                │          │                │          │  全程留在 SRAM，不落回 HBM  │
 │ Hopper 张量核   │──基元──▶ │ online softmax  │──基元──▶ │                          │
 │ WGMMA + TMA     │  (tc_04) │ 一遍流式        │ (sc_05)  │  新增的唯一东西 = FUSION   │
 │ (+FP8 tc_05)    │          │ running max/sum │          │                          │
 └────────────────┘          └────────────────┘          └──────────────────────────┘
```

- **GEMM 交付两把基元**：① CUDA-core 的分块 / 向量化 / 双缓冲**快矩阵乘**；② tensor_core 的 **Hopper `WGMMA` + `TMA`（tc_04）**——warpgroup 级异步张量核 matmul（外加 FP8 `tc_05` 把算力天花板翻倍）。
- **softmax 交付**：**快归约**（block / warp shuffle reduce）+ **online-softmax** 一遍流式技巧（`sc_05`）——"边扫边维护 running (max, sum) 并用校正因子 rescale"。这正是 FA 内层需要的 softmax 形态。
- **FA = 复用上面两把基元 + 融合**：把 attention 的两次 matmul（QKᵀ 与 PV）交给 **Hopper WGMMA**，把中间 softmax 换成 **online 技巧**，三者**融合在一个 kernel、全程留在 SRAM**（不物化 S=QKᵀ、不落回 HBM）。FA **不引入新的计算基元**，只引入"融合"。

> **硬依赖**：`gemm/` 的 tensor_core（尤其 **tc_04 WGMMA+TMA**）是 `flash_attn/` tensor_core 的**前置**——手写 `fa_hopper.cuh` 复用了它的 128B-swizzle 描述符几何；`softmax/` 的 **sc_05 online** 是 FA online-softmax 的算法前置。逐级依赖见 [`docs/CURRICULUM.md`](docs/CURRICULUM.md)。

---

## 构建与运行

### 1. 一键（推荐）

```bash
bash run_all.sh                # 自动选最空闲 GPU、锁 1980MHz、build+verify+bench，快照到 result/<时间戳>/
bash run_all.sh --gpu 3        # 强制用 GPU 3
bash run_all.sh --ncu          # 额外采集 ncu profile（需 sudo）
bash run_all.sh --no-lock      # 不锁频（无 sudo 时）
```

`run_all.sh`：**选空闲 GPU** → **锁 SM 时钟 1980MHz** → `make all` → **先 verify 再 bench**（可选 ncu）→ 落盘 `result/<ts>/`（`00_build`/`01_verify`/`02_bench`/`03_ncu`.log），软链 `result/latest`。

### 2. 手动构建

```bash
make all          # 三个算子全建（gemm 含 FP32+BF16+tensor_core；softmax；flash_attn）
make gemm         # 只建 gemm（FP32 + BF16 + tc_01..06 含 WGMMA/FP8）
make softmax
make flash_attn
make clean
```

各算子 `Makefile` 默认 `ARCH=-arch=sm_90a`；`gemm/` 默认 `TC_HOPPER=1`（编译全部 6 个张量核用例，含 Hopper 独占的 `tc_04`(WGMMA+TMA) / `tc_05`(FP8)，需 driver API `-lcuda`）。为可移植性保留 `TC_HOPPER=0 -DNO_HOPPER` 分支（老架构只编 tc_01/02/03/06）。产物 out-of-source 落各算子 `build/`。

### 3. 逐算子 bench / verify（按 id）

| 算子 / 引擎 | 可执行 | 用法 | 合法 id |
| --- | --- | --- | --- |
| GEMM FP32 cuda_core | `gemm/build/cuda_core/{bench,verify}` | `<id> M N K` | `0`=cublas_ref，`1..12` |
| GEMM BF16 cuda_core | `gemm/build/cuda_core/{bench_bf16,verify_bf16}` | `<id> M N K` | 同上 |
| GEMM tensor_core | `gemm/build/tensor_core/{bench,verify}` | `<id> M N K` | `1,2,3,4,5,6` |
| softmax cuda_core | `softmax/build/cuda_core/{bench,verify}` | `<id> M N` | `1..6` |
| FA tensor_core（手写 Hopper） | `flash_attn/build/tensor_core/{bench,verify}` | `<fp16\|bf16> B H S D causal` | dtype 选择，D∈{64,128} |
| FA cuda_core 脚手架 | `flash_attn/build/cuda_core/{bench,verify}` | `<id> B H S D causal` | `1,2` |

例子：

```bash
# GEMM：先证对，再测速
gemm/build/tensor_core/verify 4 2048 2048 2048     # tc_04 WGMMA 对拍 cuBLAS
gemm/build/tensor_core/bench  4 4096 4096 4096      # tc_04 WGMMA+TMA 张量核
gemm/build/tensor_core/bench  5 4096 4096 4096      # tc_05 FP8 WGMMA（226 TFLOPS）
# softmax：bench 报有效 HBM 带宽（÷4000）
softmax/build/cuda_core/bench  5 8192 8192          # sc_05 online eff_BW GB/s
# FA：dtype + 形状
flash_attn/build/tensor_core/bench bf16 2 32 4096 128 0   # 手写 Hopper 前向 TFLOPS
flash_attn/build/tensor_core/verify fp16 2 8 512 128 1    # 对拍 CPU 注意力（causal）
```

---

## 验证方法论

四条铁律（`run_all.sh` 强制）：

1. **先验证，再基准**：`01_verify` 全绿才进 `02_bench`。**快而错是零分**。
2. **allclose 对拍权威参考**：GEMM→cuBLAS（atol=rtol=1e-2；张量核常 bit 级一致）；softmax→CPU double 安全 softmax；FA→fp32 CPU 参考注意力（`fa_cpu_ref`，double 累加 = PyTorch SDPA 数学定义，支持 causal）。
3. **锁频基准**：`bench` 前 `nvidia-smi -lgc 1980,1980` 把 SM 钉在 H20 boost 1980MHz，消除 boost 漂移。canonical 数据一律锁频、best/median-of-N。
4. **逐级 ncu**：`--ncu` 对代表级抓 profile（落 `baselines/*.details.txt`），看 **Compute(SM)% / DRAM% / 占用率 / stall**，用计数器**解释每一级瓶颈**。

---

## 复现（Reproduction，H20 sm_90a，锁频 1980MHz，4096³）

> 完整口径与 H20 统一结论见 [`docs/REPRODUCTION.md`](docs/REPRODUCTION.md)；干净快照 `docs/clean_{bench,verify}_h20_20260706.txt`。

### GEMM 地面真值（`gemm/baselines/GROUNDTRUTH_cuda_gemm_h20.txt`）

分母：**FP32 峰值 ≈ 39.5 TFLOPS**（78·128·2·1.98GHz）；**BF16 张量核峰值 148 TFLOPS**；**FP8 峰值 296 TFLOPS**。

**① FP32 cuda_core** — 快矩阵乘阶梯

| 运行 id | kernel | GFLOPS | %峰值(39.5T) | %cuBLAS |
| ---: | --- | ---: | ---: | ---: |
| 0 | cublas_ref (SGEMM) | 28039 | 70.9% | 100% |
| 1 | naive | 3352 | 8.5% | 12.0% |
| 2 | smem | 5042 | 12.8% | 18.0% |
| 4 | 2Dblocktiling | 14489 | 36.7% | 51.7% |
| 5 | vectorized (float4) | 21064 | 53.3% | 75.1% |
| 9 | warptile（标量，**负优化**） | 11902 | 30.1% | 42.4% |
| 10 | warptile_vec | 22233 | 56.3% | 79.3% |
| 12 | **double_buffer** | **23221** | **58.8%** | **82.8%** |

> 手写最佳 `double_buffer` = **23.2 TFLOPS = cuBLAS SGEMM 的 82.8%**。`warptile` 标量版(11.9T)反低于更早的 `vectorized`(21.1T)——**负优化教材**，须配 float4（`warptile_vec` 22.2T）才追回。id 6/7/8 是 `06_autotuning.cuh` 自动调参实例。

**② tensor_core** — 张量核 matmul 阶梯（**WMMA → mma.sync → WGMMA → FP8**）

| id | 用例 | GFLOPS | 占 148T | 说明 |
| ---: | --- | ---: | ---: | --- |
| — | cuBLAS BF16（基线/上限） | 134032 | 90.6% | `bench_bf16 0` |
| 1 | tc_01 wmma_naive | 16946 | 11.5% | fragment 直取 global，张量核饿死 |
| 2 | tc_02 wmma_smem | 30123 | 20.4% | smem 复用 |
| 3 | tc_03 wmma_pipe | 38659 | 26.1% | cp.async 流水，**wmma API 天花板** |
| 6 | tc_06 mma_pipe | 75332 | 50.9% | warp 级 `mma.sync`+`ldmatrix`+`cp.async` |
| **4** | **tc_04 WGMMA+TMA** | **120646** | **81.5%** | **Hopper 原生 warpgroup 异步张量核 + TMA** |
| **5** | **tc_05 WGMMA FP8(e4m3)** | **226026** | **76.4% of 296T = 152.7% of 148T** | **FP8 把算力天花板翻倍** |

> 阶梯清清楚楚：`WMMA(API) → mma.sync(warp) → WGMMA(warpgroup)+TMA` 手写到 **81.5%**（cuBLAS 90.4%）；**FP8（tc_05）实测 226 TFLOPS——在算力弱的 H20 上收回算力的唯一杠杆**。ncu：tc_04/tc_05 的 **DRAM 仅 3–10%**（4 TB/s 闲置），占用率 7–8% 却拿 81% MFU——**WGMMA 异步，低占用不碍事**。深剖见 `gemm/docs/`。

### softmax cuda_core（`bench <id> 8192 8192`）

访存受限归约；口径 = 有效 HBM 带宽（`2·M·N·4` 字节 ÷ 时间），分母 **H20 HBM3 峰值 4000 GB/s**。

| id | kernel | 有效带宽(GB/s) | %屋顶(4000) |
| ---: | --- | ---: | ---: |
| 1 | sc_01 naive（非合并） | 102.9 | 2.6% |
| 2 | sc_02 block_reduce | 2491 | 62.3% |
| 3 | sc_03 warp_shuffle | 1663 | 41.6% |
| 4 | **sc_04 vectorized** | **3163** | **79.1%** |
| 5 | sc_05 online | 3075 | 77.0% |
| 6 | sc_06 resident | 2575 | 64.4% |

> 屋顶是 HBM 带宽，不是 FLOPs。好内核在 H20 的 4 TB/s 上达 **77–89% 峰**；naive 只 2.6%。**长行（16384²）时 `sc_06 resident` 单次 DRAM 读达 88.9%**（`bench 6 16384 16384`）。最优内核**按 N 分派**（小 N warp-per-row、大 N register-resident），详见 [`softmax/docs/`](softmax/docs/)。ncu：sc_05 DRAM 83.1% / 占用 90%——访存受限算子在 H20 的大带宽上如鱼得水。

### FlashAttention（手写 Hopper WGMMA+TMA，`flash_attn/build/tensor_core/bench`）

| 引擎 | 配置 | TFLOPS | 占 148T |
| --- | --- | ---: | ---: |
| **tensor_core（手写 Hopper）** | bf16 B2 H32 S4096 D128 | **122.5** | **82.8%** |
| tensor_core（手写 Hopper） | fp16 同上 | 122.2 | 82.6% |
| tensor_core（手写 Hopper） | bf16 causal | 113.8（有效） | 76.9% |
| cuda_core 脚手架 | fa_cc_02 tiled fp32 B2 H16 S2048 D64 | 0.65 | — |
| cuda_core 脚手架 | fa_cc_01 stream 同上 | 0.44 | — |

> **张量核 vs CUDA 核 ≈ 122.5 / 0.65 ≈ 188×**——一句话钉死"注意力的两次 matmul 必须上张量核"。手写 Hopper 前向达 **80–86% 峰**（跨 D=64/128、bf16/fp16、causal），与库（PyTorch SDPA / FA2 在 H20 ~90–96% MFU，见平台归档）**同一量级**。关键设计（`flash_attn/docs/02_tensor_core_hopper.md`）：寄存器常驻 online softmax、**P 直接当作 PV 的 A-fragment**（无 smem 往返）、两次矩阵乘都上 WGMMA、K/V 走 TMA、**单次 grid 启动**（把 35%→83%）。ncu：DRAM 仅 1.3%（计算受限），占用 12% 却 83% MFU（异步 WGMMA）。

> ✅ **H20 统一结论**：三个算子把「算力弱、带宽大」讲透（ncu DRAM% 一栏即证）——GEMM/FA **计算受限**（DRAM<11%，被 148T 张量核卡住，**FP8 是杠杆**）；softmax **访存受限**（DRAM 60–88%，被 4 TB/s 托起）。异步 WGMMA 让低占用率也高 MFU，这条 Hopper 经验在带宽富余的 H20 上被放大：矩阵乘从不挨饿。

---

## 延伸阅读

- [`docs/REPRODUCTION.md`](docs/REPRODUCTION.md) — **H20 复现报告**（环境、适配改动、全量结果、统一结论）。
- [`docs/CURRICULUM.md`](docs/CURRICULUM.md) — 跨三算子的**完整学习阶梯**（R01–R23，每个技术在哪一级定型）。
- `gemm/docs/` — GPU 硬件前置 · ncu 三板斧 · H20 GEMM 复现总结（含 WGMMA/FP8）。
- `softmax/docs/` — 快归约 / 带宽屋顶 / online 桥 / beyond-online（按 N 分派）。
- `flash_attn/docs/02_tensor_core_hopper.md` — **手写 Hopper WGMMA+TMA FA 深剖**（P→A-fragment 交接、128B-swizzle 几何、单次 grid 启动、ncu）。
