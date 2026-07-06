# cuda-ops-a800 — 主流算子的 CUDA 实现（A800 / sm_80，前向）

在 **NVIDIA A800-SXM4-80GB（Ampere, sm_80）** 真机上，用**编号、逐级 ncu 验证的阶梯**从零讲清三个主流算子的 CUDA 前向实现。每个算子是一个**自包含的 CUDA_GEMM 风格模块**（`kernels/{cuda_core,tensor_core}/` + `include/` + `docs/` + `baselines/`），有统一的 `bench <id>` / `verify <id>` 入口、统一的锁频基准与对拍方法。

- **只做前向（forward-only）**，不含反向。目标是把"算子怎么在 GPU 上从 naive 一步步优化到接近硬件峰值"讲透，而非训练框架。
- **三个算子，一条主线**：`gemm/`（快矩阵乘 + Ampere 原生张量核）→ `softmax/`（快归约 + online-softmax 技巧）→ `flash_attn/`（两次 GEMM + online softmax 融合于 SRAM）。
- **每一级都可复现、可对拍、可 profile**：`verify` 先证正确，`bench` 锁频测性能，`--ncu` 逐级抓瓶颈。

| 算子 | cuda_core 阶梯 | tensor_core | 性能口径 | 正确性对拍 |
| --- | --- | --- | --- | --- |
| **gemm** | `01_naive` … `10_doublebuffer`（FP32/BF16） | `tc_01 wmma_naive` → `tc_03 wmma_pipe`，`tc_06 mma_pipe`(=mma.sync+ldmatrix+cp.async) | GFLOPS / %峰值 / %cuBLAS | vs cuBLAS（allclose 1e-2） |
| **softmax** | `sc_01 naive` → `sc_05 online`（online = FA 的桥） | —（访存受限归约，无需张量核） | 有效 HBM 带宽 GB/s（÷2039） | vs CPU double 参考 |
| **flash_attn** | `fa_cc_01 stream`、`fa_cc_02 tiled`（from-scratch fp32 FA2 脚手架） | vendored **TinyFA** CuTe 前向（~94–96% Dao FA2） | TFLOPS | vs fp32 CPU 参考注意力（= SDPA 数学定义） |

---

## 目录布局（算子为主，operator-major）

```
cuda-ops-a800/
├── Makefile                 # 顶层派发：make all → gemm + softmax + flash_attn
├── run_all.sh               # 一键：选空闲 GPU、锁 1410MHz、build+verify+bench(+ncu)，快照到 result/<ts>/
├── common/                  # 跨算子共享：gpu_specs.py(算力事实表) summary.py run_ncu.sh
├── third_party/cutlass/     # CuTe / CUTLASS 头文件（flash_attn tensor_core 依赖）
│
├── gemm/                    # 算子①：GEMM —— 快矩阵乘 + Ampere mma.sync 张量核
│   ├── kernels/
│   │   ├── cuda_core/       # 01_naive .. 10_doublebuffer（FP32）+ bf16_cudacore + 4 个 bench/verify 驱动
│   │   └── tensor_core/     # tc_01 wmma_naive · tc_02 wmma_smem · tc_03 wmma_pipe · tc_06 mma_pipe
│   │                        #   (+ tc_04 WGMMA / tc_05 FP8：Hopper 独占，A800 -DNO_HOPPER 跳过)
│   ├── include/             # common.h · kernels.h · tc_common.cuh · 06_autotuning.cuh · bf16.h
│   ├── docs/                # 00 硬件前置 · 01 性能方法论 · a800/(复现总结 · mma.sync 张量核 · ncu 分析)
│   └── baselines/           # GROUNDTRUTH_cuda_gemm_a800.txt + a800_*.details.txt（ncu 全量）
│
├── softmax/                 # 算子②：softmax —— 快归约 + online-softmax（FA 的桥）
│   ├── kernels/cuda_core/   # sc_01 naive · sc_02 block_reduce · sc_03 warp_shuffle · sc_04 vectorized · sc_05 online
│   ├── include/softmax.h    # 声明 + CPU double 安全 softmax 参考
│   └── baselines/           # run_all.sh 在空闲 GPU 上回填
│
└── flash_attn/              # 算子③：FlashAttention —— 两次 GEMM + online softmax 融合于 SRAM
    ├── kernels/
    │   ├── cuda_core/       # fa_cc_01 stream · fa_cc_02 tiled（from-scratch fp32 FA2 脚手架）
    │   └── tensor_core/     # bench/verify 驱动 → 调 vendor/tfa 的 CuTe 前向
    ├── include/             # fa_common.h(FLOP 模型 + CPU 参考注意力) · fa_cc.h
    ├── vendor/tfa/          # 内联的 TinyFA CuTe 前向（性能路径 = 终态 kernel）
    └── baselines/           # run_all.sh 在空闲 GPU 上回填
```

每个算子目录都能**独立 `make` / `bench` / `verify`**，不依赖其它算子的构建产物；顶层 `Makefile` 与 `run_all.sh` 只是把三者串起来。

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
 │ Ampere 张量核   │──基元──▶ │ online softmax  │──基元──▶ │                          │
 │ mma.sync+       │  (tc_06) │ 一遍流式        │ (sc_05)  │  新增的唯一东西 = FUSION   │
 │ ldmatrix+cp.async│         │ running max/sum │          │                          │
 └────────────────┘          └────────────────┘          └──────────────────────────┘
```

- **GEMM 交付两把基元**：① CUDA-core 的分块 / 向量化 / 双缓冲**快矩阵乘**；② tensor_core 的 **`mma.sync.m16n8k16` + `ldmatrix` + 多级 `cp.async`（tc_06）**——Ampere 原生的张量核 matmul。
- **softmax 交付**：**快归约**（block / warp shuffle reduce）+ **online-softmax** 一遍流式技巧（`sc_05`）——"边扫边维护 running (max, sum) 并用校正因子 rescale"。这正是 FA 内层需要的 softmax 形态。
- **FA = 复用上面两把基元 + 融合**：把 attention 的两次 matmul（QKᵀ 与 PV）交给 **GEMM 的 mma.sync 张量核**，把中间的 softmax 换成 **softmax 的 online 技巧**，并把三者**融合在一个 kernel、全程留在 SRAM**（不物化 S=QKᵀ、不落回 HBM）。FA **不引入新的计算基元**，只引入"融合"这一件事。

> **硬依赖**：`gemm/` 的 tensor_core（尤其 **tc_06 mma.sync**）是 `flash_attn/` 的 tensor_core 的**前置**；`softmax/` 的 **sc_05 online** 是 FA online-softmax 的算法前置。逐级依赖与"每个技术在哪一级定型"见 [`docs/CURRICULUM.md`](docs/CURRICULUM.md)。

---

## 构建与运行

### 1. 一键（推荐）

```bash
bash run_all.sh                # 自动选最空闲 GPU、锁 1410MHz、build+verify+bench，快照到 result/<时间戳>/
bash run_all.sh --gpu 3        # 强制用 GPU 3
bash run_all.sh --ncu          # 额外采集 ncu profile（需 sudo）
bash run_all.sh --no-lock      # 不锁频（无 sudo 时）
```

`run_all.sh` 做四件事：**选空闲 GPU**（按 `memory.used` 最小，busy>4GB 会告警）→ **锁 SM 时钟 1410MHz** → `make all` → **先 verify 再 bench**（可选 ncu）→ 全部落盘到 `result/<ts>/`（`00_build.log` / `01_verify.log` / `02_bench.log` / `03_ncu.log`），并软链 `result/latest`。

### 2. 手动构建

```bash
make all          # 三个算子全建（gemm 含 FP32+BF16+tensor_core；softmax；flash_attn）
make gemm         # 只建 gemm（FP32 + BF16 + tc）
make softmax
make flash_attn
make clean
```

各算子 `Makefile` 默认 `ARCH=-arch=sm_80`；`gemm/` 默认 `TC_HOPPER=0`（A800 只编译 WMMA 三级 + `tc_06`，用 `-DNO_HOPPER` 把 Hopper 独占的 `tc_04`/`tc_05` 从派发表剔除）。产物 out-of-source 落在各算子的 `build/`。

### 3. 逐算子 bench / verify（按 id）

| 算子 / 引擎 | 可执行 | 用法 | 合法 id |
| --- | --- | --- | --- |
| GEMM FP32 cuda_core | `gemm/build/cuda_core/{bench,verify}` | `<id> M N K` | `0`=cublas_ref，`1..12` |
| GEMM BF16 cuda_core | `gemm/build/cuda_core/{bench_bf16,verify_bf16}` | `<id> M N K` | 同上 |
| GEMM tensor_core | `gemm/build/tensor_core/{bench,verify}` | `<id> M N K` | A800：`1,2,3,6` |
| softmax cuda_core | `softmax/build/cuda_core/{bench,verify}` | `<id> M N` | `1..5` |
| FA tensor_core (TinyFA) | `flash_attn/build/tensor_core/{bench,verify}` | `<fp16\|bf16> B H S D causal` | dtype 选择 |
| FA cuda_core 脚手架 | `flash_attn/build/cuda_core/{bench,verify}` | `<id> B H S D causal` | `1,2` |

> **GEMM cuda_core 的"文件编号"≠"运行 id"**：源码文件是 `01_naive..10_doublebuffer`，但运行 id 另有一套（id 0=cuBLAS，6/7/8 是 `06_autotuning.cuh` 的 tile 自动调参实例）。对照表见 [复现章节](#gemm-cuda_core-id-对照)。不带参数运行任一 `bench`/`verify` 会打印该引擎的 id 清单。

例子：

```bash
# GEMM：先证对，再测速
gemm/build/cuda_core/verify 5 2048 2048 2048      # id5 vectorized 对拍 cuBLAS
gemm/build/cuda_core/bench  5 4096 4096 4096       # id5 vectorized 4096³ GFLOPS
gemm/build/tensor_core/bench 6 4096 4096 4096      # tc_06 mma.sync 张量核
# softmax：bench 报有效 HBM 带宽
softmax/build/cuda_core/verify 5 1024 2048         # sc_05 online 对拍 CPU double
softmax/build/cuda_core/bench  5 8192 8192         # eff_BW GB/s（÷2039）
# FA：dtype + 形状
flash_attn/build/tensor_core/bench fp16 2 32 4096 128 0   # TinyFA 前向 TFLOPS
flash_attn/build/cuda_core/verify 2 2 4 256 64 0          # fa_cc_02 tiled 对拍 CPU 注意力
```

---

## 验证方法论

四条铁律，贯穿所有算子（`run_all.sh` 强制执行）：

1. **先验证，再基准（verify-before-bench）**：`run_all.sh` 先跑完 `01_verify` 全绿，才进 `02_bench`。**没证明正确的 kernel 不看性能**——快而错是零分。
2. **allclose 对拍权威参考**：
   - **GEMM** → 对拍 **cuBLAS**（`|a−b| ≤ atol + rtol·|b|`，atol=rtol=1e-2）；tensor_core 与 cuBLAS BF16 常做到 bit 级一致（`max_abs=0`）。
   - **softmax** → 对拍 **CPU double 精度**安全 softmax（`softmax.h` 里的 `softmax_cpu_ref`）。
   - **FlashAttention** → 对拍 **fp32 CPU 参考注意力**（`fa_common.h` 的 `fa_cpu_ref`，double 累加，即 **PyTorch SDPA 的数学定义**；支持 MHA/GQA、causal）。
3. **锁频基准（locked-clock bench）**：`bench` 前 `nvidia-smi -lgc 1410,1410` 把 SM 钉在额定 1410MHz。A800 默认自适应 boost 在重张量负载下只维持 ~1140–1290MHz（`dmon` 实测），不锁频则同一 kernel 会随预热在 214→262T 之间漂移 20%。canonical 数据一律锁频、avg-of-10（FA/softmax 取 median）。
4. **逐级 ncu（ncu per rung）**：`--ncu` 对每一级抓 profile（落 `baselines/*.details.txt`），看 **Compute(SM)% / L1-TEX% / 占用率 / cyc-per-issue / stall**——用计数器**解释每一级的瓶颈是什么、下一级凭什么更快**（例：WMMA 三级被 L1/TEX 打满 → tc_06 用 `ldmatrix` 把 L1/TEX 从 93.8% 降到 44.5%）。ncu 锁 base clock 采计数器，故只看与时钟无关的 % / 占用率 / stall。

---

## 复现（Reproduction）

### GEMM 地面真值（GROUNDTRUTH，A800 sm_80，锁频 1410MHz，4096³）

下面这张表是 **CUDA_GEMM 在本机 A800 上的锁频实测**（见 `gemm/baselines/GROUNDTRUTH_cuda_gemm_a800.txt`，2026-07-06 于一块**瞬时空闲**的 GPU 上采得；本机平时被 8 卡 vLLM 共享）。本仓库 `gemm/` 的 kernel 与 CUDA_GEMM **逐字节相同**，故在一块干净的 A800 上**按构造复现同一组数字**。

分母：**FP32 非张量峰值 ≈ 19.49 TFLOPS**；**BF16 张量核峰值 = 312 TFLOPS**（GA100 规格，= H20 的 2.1×）。

**① FP32 cuda_core（`cuda_core/bench <id>`）** — 快矩阵乘阶梯

| 文件 | 运行 id | kernel | GFLOPS | %峰值(19.5T) | %cuBLAS |
| --- | ---: | --- | ---: | ---: | ---: |
| — | 0 | cublas_ref | 15688 | 80.5% | 100% |
| `01_naive` | 1 | naive | 2935 | 15.1% | 18.7% |
| `02_smem` | 2 | smem | 4143 | 21.3% | 26.4% |
| `03_blocktiling` | 3 | blocktiling (1D) | 10289 | 52.8% | 65.6% |
| `04_2Dblocktiling` | 4 | 2D blocktiling | 11510 | 59.1% | 73.4% |
| `05_vectorized` | 5 | vectorized (float4) | 13988 | 71.8% | 89.2% |
| `07_warptile` | 9 | warptile | 10239 | 52.5% | 65.3% |
| `08_warptile_vec` | 10 | **warptile_vec** | **17363** | **89.1%** | 110.7% |
| `09_bankconflict` | 11 | bank_conflict | 16617 | 85.3% | 105.9% |
| `10_doublebuffer` | 12 | double_buffer | 13829 | 71.0% | 88.1% |

> 手写最佳 `warptile_vec` = **17.4 TFLOPS ≈ FP32 峰值的 89%**。此快照里 `warptile_vec`/`bank_conflict` 数值反超 `cublas_ref`(15688)，是因为这次瞬时抓取的 cuBLAS 读数偏低；充分预热后 cuBLAS SGEMM 可达 ~19.0T（详见 `gemm/docs/a800/`）。绝对值低于 H20 纯因 A800 每 SM 只有 64 个 FP32 核（Hopper 128）。<a id="gemm-cuda_core-id-对照"></a>id 6/7/8 是 `06_autotuning.cuh` 的 tile 自动调参实例（不在此阶梯的主线上）。

**② tensor_core（`tensor_core/bench <id>`）** — 张量核 matmul 阶梯

| 级 | id | 用例 | GFLOPS | TFLOPS | %峰值(312T) |
| --- | ---: | --- | ---: | ---: | ---: |
| ① | 1 | tc_01 wmma_naive | 15570 | 15.6 | 5.0% |
| ② | 2 | tc_02 wmma_smem | 23388 | 23.4 | 7.5% |
| ③ | 3 | tc_03 wmma_pipe (cp.async) | 34601 | 34.6 | 11.1% |
| ⑥ | 6 | **tc_06 mma_pipe** (mma.sync+ldmatrix+cp.async) | **110535** | **110.5** | **35.4%** |

> WMMA 三级（`load_matrix_sync`/`mma_sync`）被 L1/TEX 喂数打满，在 A800 止步 ~11% 峰；**tc_06** 换成 Ampere 原生 `mma.sync.m16n8k16` + `ldmatrix`（消 bank 冲突）+ 多级 `cp.async`，把手写拉到 **110.5 TFLOPS = 312T 的 35.4%**（= WMMA `tc_03` 的 3.2×）。**tc_06 就是 FA tensor_core 复用的那把张量核基元。** 
> 口径说明：`bench` 二进制打印的内建 `util(vs148T)` 列沿用 H20 教程写死的 148T 分母（故显示 10.5%/15.8%/23.4%…）；A800 的真实 MFU 请以上表的 **%峰值(312T)** 为准。tc_06 的深度剖析（ncu 证据、为何纯 CUDA C++ 上限约 35–48%）见 `gemm/docs/a800/Ampere mma.sync 张量核.md`。Hopper 独占的 `tc_04`(WGMMA+TMA)/`tc_05`(FP8) 在 A800 无对应指令、不编译。

### softmax / FlashAttention 性能数字：待 `run_all.sh` 在空闲 GPU 回填

`softmax/baselines/` 与 `flash_attn/baselines/` 目前**为空**——这是**有意为之**：

- **本开发机在编写期间被一套 8 卡 vLLM 服务长期占用**，`bench` 的计时会被算力争用严重污染（GPU busy>4GB 时 `run_all.sh` 会打 `WARNING`）。因此**所有 timing 运行都门控在"GPU 空闲"这一前提上**：在一块空闲卡上 `bash run_all.sh`，`02_bench.log` 会自动填入 softmax 的 **有效 HBM 带宽（GB/s，÷2039 得屋顶线占比）** 与 FA 的 **TFLOPS**，并快照到 `result/<ts>/`。
- **正确性与算力争用无关（contention-immune），且已经 PASS**：`verify` 对拍的是 cuBLAS / CPU double / CPU 参考注意力，结果不受同机其它进程影响。无论 GPU 是否被 vLLM 占着，`01_verify.log` 都稳定全绿。
- **GEMM 数字之所以已在表内**，是因为 CUDA_GEMM 的 GROUNDTRUTH 恰好在一段**瞬时空闲**窗口锁频采得；`gemm/` 用相同 kernel，干净卡上按构造复现。softmax/FA 只等下一个空闲窗口跑一遍 `run_all.sh` 即补全。

> 一句话：**正确性现在就能全绿（与争用无关）；性能数字只差一块空闲的 A800 + 一条 `bash run_all.sh`。**

---

## 延伸阅读

- [`docs/CURRICULUM.md`](docs/CURRICULUM.md) — 跨三算子的**完整学习阶梯**（逐级教什么、依赖谁、每个技术在哪一级定型）。
- `gemm/docs/00_GPU硬件前置知识.md` · `01_性能分析方法论.md` — GPU 硬件前置 & ncu 三板斧。
- `gemm/docs/a800/` — A800 GEMM 复现总结、**Ampere mma.sync 张量核（tc_06 深剖）**、ncu 分析。
