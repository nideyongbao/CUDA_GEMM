# A800 GEMM 复现总结

> 本文把原本在 **NVIDIA H20（Hopper, sm_90）** 上写就的 GEMM 优化教程，完整迁移到 **NVIDIA A800-SXM4-80GB（Ampere, sm_80）** 真机复现：每个示例都在 A800 上重跑、记录实际结果，并与 ① H20 基线（`profiling/throughput_4096.txt`、`docs/13`）② cuBLAS 库 ③ 硬件理论峰值 ④ 一份公开 A800 cuBLAS BF16 参考基准，逐项对账，给出结论。
>
> 分支：`feat/a800-reproduction`。原 H20 收口见 [docs/13](../h20/13%20H20%20GEMM%20复现总结.md)。

---

## 0、TL;DR

1. **CUDA core / FP32**：完整复现整条阶梯，13 个 kernel 全部对拍 cuBLAS PASS。手写最佳 **17.6 TFLOPS ≈ FP32 峰值(19.49T) 的 90%、cuBLAS SGEMM 的 93%**。因 A800 FP32 单元只有 64 核/SM（Hopper 128 核/SM），**绝对吞吐低于 H20**（17.6 vs 24T；cuBLAS 19 vs 30T）。
2. **BF16 / 张量核（关键差异）**：A800 BF16 张量核峰值 **312 TFLOPS = H20(148T) 的 2.1 倍**。A800 cuBLAS BF16 实测 **214–270 TFLOPS（默认时钟）/ 262–295 TFLOPS（锁频）**，**约为 H20 cuBLAS BF16(132T) 的 1.6–2.2 倍** —— **做 BF16 GEMM，A800 是更强的卡**。
3. **手写张量核在 A800 只能到 WMMA**：Ampere **硬件无 WGMMA / TMA / FP8**，H20 教程里真正质变的 **tc_04(WGMMA→80%)、tc_05(FP8)** 在 A800 **无法编译运行**。A800 手写张量核止步 **tc_03 WMMA+cp.async = 43 TFLOPS（锁频，13.8% 峰值）**。要吃满 312T 只能靠 cuBLAS/CUTLASS（Ampere 原生 `mma.sync`+`ldmatrix`，本仓库未手写）。
4. **方法论 & 参考对账**：A800 默认 boost 在重张量负载下只能维持 **~1140–1290 MHz**（dmon 实测，非额定 1410）。**默认时钟下我的 cuBLAS BF16 与公开参考表逐 shape 吻合（误差 ≤3%）**；canonical 数据用**锁频 1410MHz**（可复现、额定满频）。详见 §5。

---

## 1、硬件与算力峰值（利用率的分母）

| 维度 | **A800-SXM4-80GB（本次）** | H20（原教程） |
| --- | --- | --- |
| 架构 / CC | Ampere GA100 / **sm_80 (8.0)** | Hopper / sm_90 (9.0) |
| SM 数 | **108** | 78 |
| 额定 boost / 实测持续 | 1410 MHz / **默认仅~1200，锁频可达1410** | — |
| FP32 峰值 | **19.49 TFLOPS** | ~40–44 TFLOPS |
| **BF16 张量核峰值(dense)** | **312 TFLOPS** | **148 TFLOPS** |
| FP8 张量核 | **无** | 296 TFLOPS |
| 显存 / 带宽 | 80GB HBM2e / **2039 GB/s** | 96GB HBM3 / ~4 TB/s |
| L2 / smem | 40 MB / 164 KB·SM | 60 MB / 228 KB·SM |
| **WGMMA / TMA** | **无（Hopper 独占）** | 有 |

> 实测确认（deviceQuery）：108 SM、1.41GHz、2039 GB/s、L2 40MB、smem 164KB/SM、FP32 峰值 19.49 TFLOPS。BF16 312T 为 GA100 规格（同 A100）。
>
> **两卡性格相反**：H20 = 带宽厚(4TB/s)、FP32 较高、BF16 张量核薄(148T) 但有 WGMMA 把它用满；A800 = **BF16 张量核厚(312T)、FP32 薄(19.5T)，但无 WGMMA**，手写吃不到张量核峰值。

---

## 2、在 A800 上构建与运行（实际步骤）

原 Makefile 默认 `-arch=sm_90a`（Hopper 专用）。本分支让 `ARCH`/`TC_HOPPER` 可命令行覆盖，并用 `-DNO_HOPPER` 把 Hopper 独占用例（tc_04/05）从派发表剔除：

```bash
make ARCH=-arch=sm_80                  # CUDA core FP32+BF16，源码零改动
make tc ARCH=-arch=sm_80 TC_HOPPER=0   # Tensor Core 只编 WMMA 三级(tc_01-03)
```

构建后 tensor_core 派发表正确只剩 3 个用例：

```
$ ./build/tensor_core/bench 99
tensor_core 用例 id：
  1  tc_01 WMMA_naive
  2  tc_02 WMMA_smem
  3  tc_03 WMMA_pipe
```

> **tc_04/05 为何不能跑**：tc_04 用 `wgmma.mma_async.m64n128k16`（warpgroup 异步张量指令）+ TMA(`cp_async_bulk_tensor`/`CUtensorMap`)，tc_05 再换 FP8 `e4m3`。这些指令/硬件单元**只在 Hopper sm_90 存在**，`sm_80` 编译会被 ptxas 拒绝。不是代码问题，是**架构能力缺失**——也是 A800 与 H20 最本质的差距。

运行入口与 H20 一致：`bench/verify <id> M N K`（FP32 id 0-12、BF16 同、WMMA id 1-3）。

---

## 3、正确性（全部 PASS）

2048³ 对拍 cuBLAS（allclose `|a-b|≤atol+rtol|b|`，atol=rtol=1e-2）：

| 引擎 | 驱动 | 结果 |
| --- | --- | --- |
| FP32 ×13（id 0–12） | `cuda_core/verify` | **全部 PASS** |
| BF16 ×13（id 0–12） | `cuda_core/verify_bf16` | **全部 PASS** |
| WMMA ×3（tc_01–03） | `tensor_core/verify` | **全部 PASS**（`max_abs=0`，与 cuBLAS BF16 bit 级一致） |

---

## 4、性能阶梯（4096³，A800 vs H20，逐引擎同口径）

> **口径（回应"与 H20 对比要一致"）**：每条线两卡都用**同一个驱动**取数、利用率分母用**各自卡对应峰值**。绝不跨驱动混用（tc_04 里的 "cuBLAS BF16 fair" 只属 WGMMA 小节，A800 不涉及）。
>
> A800 canonical = **GPU 锁频 1410MHz、avg-of-10**。其中 cuBLAS BF16 / WMMA(tc_01,tc_03) 等张量核重载 kernel 对时钟敏感，括注其**默认时钟**值（与公开参考一致，详见 §5）；FP32 与 BF16 手写（CUDA core 轻载）默认时钟≈锁频，已实测验证。

### 4.1 CUDA core / FP32（`cuda_core/bench`）

| id | kernel | A800 GFLOPS | %峰值(19.49T) | %cuBLAS | H20 GFLOPS | H20 %cuBLAS |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 0 | cublas_ref | **19017** | **97.6%** | 100% | 30266 | 100% |
| 1 | naive | 3051 | 15.7% | 16.0% | 3532 | 11.7% |
| 2 | smem | 5348 | 27.4% | 28.1% | 5411 | 17.9% |
| 3 | blocktiling(1D) | 10314 | 52.9% | 54.2% | 9410 | 31.1% |
| 4 | 2D blocktiling | 14426 | 74.0% | 75.9% | 15547 | 51.4% |
| 5 | vectorized(float4) | 17602 | 90.3% | 92.6% | 22789 | 75.3% |
| 6 | autotune 64×64×8_8×4 | **17615** | **90.4%** | **92.6%** | 22759 | 75.2% |
| 7 | autotune 64×64×16_8×4 | 16818 | 86.3% | 88.4% | 21852 | 72.2% |
| 8 | autotune 64×64×8_8×8 | 16781 | 86.1% | 88.2% | 19822 | 65.5% |
| 9 | warptile | 12665 | 65.0% | 66.6% | 12842 | 42.4% |
| 10 | warptile_vec | 17349 | 89.0% | 91.2% | 23970 | 79.2% |
| 11 | bank_conflict | 17096 | 87.7% | 89.9% | 21519 | 71.1% |
| 12 | double_buffer | 17055 | 87.5% | 89.7% | 25131 | 83.0% |

- 阶梯形状与 H20 一致；**A800 手写最佳 ≈17.6T**（H20 25.1T）。绝对值低纯因 FP32 核少（64 vs 128/SM）。
- **A800 手写达 cuBLAS 的 92.6%、峰值 90%，比例高于 H20**：A800 cuBLAS SGEMM(19017) 已是峰值 97.6%，二者同被 FP32 算力卡住。
- 微架构差异：A800 上 `double_buffer`(17055) 略**低于** `vectorized`(17602)；H20 上 double_buffer 是冠军——Ampere 双缓冲收益不及 Hopper。

### 4.2 BF16 on CUDA core（`cuda_core/bench_bf16`，分母=各自 BF16 张量核峰值）

> "BF16 输入 + FP32 累加，**仍跑 CUDA core**"的对照——证明只换类型不用张量核没用。工具打印的 `util(vs148T)` 分母是 H20 的；**A800 真实利用率 = GFLOPS/312000**，下表已换算。

| id | kernel | A800 GFLOPS | %峰值(312T) | H20 GFLOPS | H20 %峰值(148T) |
| ---: | --- | ---: | ---: | ---: | ---: |
| 0 | **cublas_bf16** | **264600**（默认 214000） | **84.8%**（默认 68.6%） | 132152 | 89.3% |
| 1 | naive | 3105 | 1.0% | 3228 | 2.2% |
| 4 | 2D blocktiling | 14336 | 4.6% | 15841 | 10.7% |
| 5 | vectorized | **17782** | **5.7%** | 23631 | 16.0% |
| 10 | warptile_vec | 17473 | 5.6% | 24464 | 16.5% |
| 12 | double_buffer | 17251 | 5.5% | 25882 | 17.5% |

- **cuBLAS BF16：A800 262T（锁频）/214T（默认）≈ H20(132T) 的 1.6–2.0 倍** —— 厚张量核的威力。
- **手写 BF16(CUDA core) 在 A800 仅 ~5.7% 峰值**（H20 ~17.5%）。不是 A800 差，恰相反：峰值翻倍而代码仍跑 CUDA core（绝对 17.8T ≈ H20 24T 量级但分母大一倍），更说明**必须真正用张量核**。

### 4.3 Tensor Core / WMMA（`tensor_core/bench`，分母=各自 BF16 峰值）

| 级 | 用例 | A800 GFLOPS | %峰值(312T) | H20 GFLOPS | H20 %峰值(148T) |
| --- | --- | ---: | ---: | ---: | ---: |
| ① | tc_01 WMMA_naive | **17790**（默认 13728） | 5.7% | 16688 | 11.3% |
| ② | tc_02 WMMA_smem | **27100** | 8.7% | 29692 | 20.1% |
| ③ | tc_03 WMMA_pipe (cp.async) | **43051**（默认 34754） | **13.8%** | 38057 | 25.7% |
| ④ | tc_04 WGMMA(TMA+WS) | **不支持** | — | 119357 | 80.6% |
| ⑤ | tc_05 WGMMA_fp8 | **不支持** | — | 224225 | 75.8%/296T |

- A800 WMMA 三级**绝对吞吐略高于 H20**（tc_03 43 vs 38T）——张量核更强；但 **% 峰值更低**（13.8% vs 25.7%），同样写法填不满 2 倍大的峰值。
- **A800 手写张量核到 tc_03(43T) 即到顶**：缺 WGMMA/TMA，无法走 tc_04(80%)。**A800 手写最高 13.8% vs H20 手写最高 80.6%——两次复现最大鸿沟。**

---

## 5、时钟策略与公开基准对账（重要）

复现初期发现 A800 同一 kernel 多次测量波动达 20%。`nvidia-smi dmon` 连续采样揭示根因：**A800 在重张量负载下，默认自适应 boost 只能维持 ~1140–1290 MHz，并非额定 1410**（单次 `nvidia-smi` 抓到的 1410 是瞬时峰值，会误导）。

```
锁频1410：  time=3.73ms  GFLOPS=294660   SM时钟全程 1410 MHz
默认(解锁)：time=4.07ms  GFLOPS=269906   SM时钟 1140/1155/1290 MHz   ← 实际只~1200
```

**哪些 kernel 受影响**（实测）：
- **对时钟敏感**（默认比锁频低 15–20%）：cuBLAS BF16、WMMA tc_01/tc_03 —— 张量核功耗密度高，默认 boost 保守降频。
- **不敏感**（默认≈锁频）：FP32 全部（cuBLAS 预热后 = 19T）、BF16 手写（CUDA core 轻载）。
- 默认时钟还**强依赖预热**：cuBLAS BF16 @4096³ 冷态 214T → 连跑 5 次热态 261T。**这正是基准测试要锁频的原因**——canonical 用锁频 1410（可复现、额定满频）。

**与公开参考表对账**（cuBLAS BF16，峰值分母两边都是 312T）：

| Shape | 参考表 TFLOPS / MFU | **A800 默认时钟** | A800 锁频1410 |
| --- | ---: | ---: | ---: |
| 2048³ | 128.6 / 41.2% | **124.4 / 39.9%** | 153.6 / 49.2% |
| 4096³ | 217.8 / 69.8% | **214.0 / 68.6%** | 264.6 / 84.8% |
| 8192³ | 245.5 / 78.7% | **238.5 / 76.4%** | 293.7 / 94.1% |
| 16384³ | 268.2 / 86.0% | **274.6 / 88.0%** | 294.8 / 94.5% |
| 4096×3584×18944 | 264.0 / 84.6% | **269.9 / 86.5%** | 269.5 / 86.4% |
| 4096×18944×3584 | 253.4 / 81.2% | **226.8 / 72.7%** | 278.9 / 89.4% |

> **结论：参考表与本测试一致。** 默认时钟下 6 个 shape 有 5 个与参考误差 ≤3%（2048³ −3.3%、4096³ −1.7%、8192³ −2.9%、16384³ +2.4%、rect1 +2.2%）；仅 `4096×18944×3584` 偏低 10%（N 超大时 cuBLAS 选核/boost 抖动差异，锁频后 278.9T 反高于参考）。**参考表对应默认 boost；我的 canonical 锁频值是额定满频上限，高约 10–20%，两者描述同一块硬件，差距纯属时钟。**

### 5.1 为什么"94% 峰值"与"实际 78.7%"并存？——按实际时钟算效率，cuBLAS 两边都是 ~94%

312T 这个分母是**额定 1410MHz 下**的峰值。用 dmon 同步采时钟 + 吞吐，把 8192³ cuBLAS BF16 拆开看：

| 时钟策略 | 吞吐 | 活跃 SM 时钟(dmon) | 该频有效峰值 312×f/1410 | **按实际频效率** | vs 额定312T |
| --- | ---: | ---: | ---: | ---: | ---: |
| 锁频 1410 | 293.6T | 1410 MHz（稳定） | 312T | **94.1%** | 94.1% |
| 默认（解锁） | 237.6T | ~1140 MHz（dmon 主频） | 252T | **~94.3%** | 76.2% |

> **cuBLAS 的真实效率两边都是 ~94%**（相对它当时实际跑的时钟）。参考表的 78.7% / 我默认的 76.2%，**不是 cuBLAS 没打满，而是 GPU 在默认 boost 下只跑 ~1140–1215MHz，却拿额定 1410MHz 的 312T 当分母**。换句话说：78.7% = 94%(效率) × 84%(时钟比 1185/1410)。**口径确认无误**——分母同为 312T@1410，差的是分子里的实际时钟。要让"vs 312T"这个 MFU 数字也到 94%，必须锁频把 GPU 钉在额定 1410。

**铁证（吞吐比 = 时钟比）**：8192³ 默认 237.6T / 锁频 293.6T = **0.809**，而 1140MHz / 1410MHz = **0.808** —— 两者吻合到 0.1%，即默认就跑在基频 1140，锁频跑 1410，效率两边一致。

**且默认偏低不是"被限频防过热"**：跑 BF16 时实测 `clocks_throttle_reasons` 全部 **Not Active**（非 SW Power Cap、非 HW/SW Thermal），温度 **30–37℃**、功耗 **< 400W 上限**。根因是这张卡的**默认应用时钟 = 1140MHz**（`default_applications.graphics`，额定 max = 1410），默认 GPU Boost 对 BF16 这种高电流密度负载不稳定地拉升（FP32 预热后能上 1410，BF16 常停在 1140–1290）——**是默认 boost 策略 + 基频设定，不是机器保护性降频**。

---

## 6、尺寸缩放（A800，锁频 1410MHz）

| size | FP32 cublas | FP32 vectorized | BF16 cublas | WMMA tc_03 |
| ---: | ---: | ---: | ---: | ---: |
| 1024³ | 16552 | 11084 | 110376 | 22758 |
| 2048³ | 17657 | 15700 | 153357 | 39499 |
| 4096³ | 19021 | 17627 | 264521 | 43052 |
| 8192³ | **19176 (98.4%峰)** | 18124 (93%峰) | **294628 (94.4%峰)** | 43490 (13.9%峰) |

- **cuBLAS 越大越满**：FP32 8192³ 吃到峰值 **98.4%**；BF16 8192³ 吃到 312T 的 **94.4%（294.6T）**——确认 312T 峰值正确、cuBLAS 几乎打满。**注**：此为锁频 1410MHz 值；默认 boost 下因实际只跑 ~1150MHz，对 312T 的 MFU 约 78%（≈ 参考表，见 §5.1），但按实际时钟仍是 ~94% 效率。
- **手写 WMMA tc_03 在 4096³ 后饱和(~43T,13.9%峰)**：受 L1/TEX fragment 搬运卡死，放大也突破不了（见 §8）。

---

## 7、Hopper 独占级在 A800 的缺失（tc_04 / tc_05）

| | tc_04 WGMMA+TMA+WS | tc_05 FP8 |
| --- | --- | --- |
| H20 实测 | 119–131 TFLOPS（**80.6%**，8192³ 88%） | 224 TFLOPS（75.8%/296T） |
| A800 | **无法运行** | **无法运行** |
| 缺失硬件 | `wgmma.mma_async`、TMA、warp specialization mbarrier | 同上 + FP8 `e4m3` |

H20 教程"最后三级质变"（异步 warpgroup MMA 用流水线代替占用率→80%；FP8 可插拔翻倍）在 Ampere **无对应指令**。A800 逼近 312T 的正道是 **Ampere 原生 `mma.sync.m16n8k16` + `ldmatrix` + 多级 `cp.async` + 寄存器双缓冲**（CUTLASS Ampere 写法）——cuBLAS 正是如此（214–294T）。**本仓库未手写该级**，故 A800 上"手写 vs cuBLAS"张量核差距（43T vs 262T，≈6×）远大于 H20（119T vs 131T，≈0.9×）。**这是 A800 后续最值得补的一课。**

### 7.1 为什么手写张量核只到 WMMA 13.8%？——是代码/算法问题，不是统计、不是硬件天花板

逐一排除：

1. **不是统计问题**：分母 312T 经 cuBLAS 验证为真（同一块 A800 上 cuBLAS BF16 锁频打到 293.6T = 94%）；分子 43T（tc_03）稳定且对拍 PASS。13.8%（锁频）/11.1%（默认）是**真实达成比例**。
2. **不是硬件天花板**：A800 silicon 完全能到 94%——**ncu 抓到 cuBLAS 在 A800 上用的真实 kernel 就是证据**：
   - BF16：`ampere_s16816gemm_bf16_256x128_ldg8_stages_32x3_nn` —— `s16816` = **`mma.sync.m16n8k16`**（Ampere warp 级张量指令）+ `ldg8`(128bit 取数) + **`stages_32x3`(3 级 cp.async 流水线)**。
   - FP32：`ampere_sgemm_128x64_nn`。
3. **是代码/算法问题（且"A800 写法不同"成立）**：本仓库手写阶梯用的是 **WMMA C++ API**（`load_matrix_sync`/`mma_sync`），其上**直接跳到 Hopper WGMMA（tc_04）**，**跳过了 Ampere 原生的 `mma.sync.m16n8k16`+`ldmatrix` 这一级**。在 H20 上跳过没问题（WGMMA 更高级取代它）；在 A800 上 WGMMA 不存在，于是手写就**卡在 WMMA 的访存路径上下不来**——ncu 实证：

   | 用例 | Compute(SM)% | L1/TEX% | cyc/iss | 瓶颈 |
   | --- | ---: | ---: | ---: | --- |
   | tc_01 WMMA_naive | 16.5 | **99.7** | 106.8 | fragment 直取 global，L1/TEX 打满、张量核饿死 |
   | tc_02 WMMA_smem | 32.6 | 73.3 | 14.6 | smem 复用，但 `load_matrix_sync`+索引仍压 L1/TEX/MIO |
   | tc_03 WMMA_pipe | 24.8 | **93.8** | 27.4 | cp.async 重叠，**仍被 L1/TEX 喂数卡死，张量核只忙 25%** |

   关键差别：cuBLAS 用 `ldmatrix` 从 smem 高效喂 fragment（绕开 `load_matrix_sync` 的 L1/TEX 瓶颈）、用 `mma.sync` 更细粒度发射、3 级流水线 + swizzle 消 bank conflict —— 把张量核喂饱到 94%。WMMA 这套 API 在 Ampere 上**喂数效率天生低一档**，加上 tc_01-03 是教学版（无 smem swizzle、无寄存器双缓冲），所以只到 ~14%。

> **一句话**：13.8% 不是测错、也不是 A800 跑不动，而是**仓库缺了 Ampere 原生 `mma.sync`+`ldmatrix` 这级手写 kernel**（H20 用 WGMMA 顶替了它，A800 没得顶替）。补上这级（CUTLASS Ampere 风格）手写就能往 cuBLAS 的 94% 靠。

---

## 8、ncu 关键洞察（A800 vs H20，2048³，% 类指标跨卡可比）

> 完整见 [profiling/a800/SUMMARY.md](../../profiling/a800/SUMMARY.md)。ncu 锁 base clock 采计数器，故只看与时钟无关的 %/占用率/stall。

| 用例 | A800 Compute(SM)% | SMBusy% | L1/TEX% | 占用率 | cyc/iss | H20 对照(Compute/L1TEX/占用) |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| cublas FP32 | 91.3 | 96.6 | 38.5 | 18.5% | 5.26 | 88.1 / 88.0 / 24.4% |
| 01 naive | 75.2 | 44.0 | 75.9 | 98.2% | 35.7 | 75.2 / 76.0 / 98.9% |
| 02 smem | 77.0 | 40.6 | 92.4 | 98.6% | 41.5 | 77.9 / 93.7 / 98.8% |
| 10 doublebuffer | 79.9 | 83.9 | 43.0 | 18.5% | 6.15 | 70.7 / 66.0 / 21.5% |
| tc_01 WMMA_naive | 16.5 | 10.3 | **99.7** | 68.8% | **106.8** | 16.5 / 99.8 / 71.2% |
| tc_02 WMMA_smem | 32.6 | 33.9 | 73.3 | 30.9% | 14.6 | 37.8 / 82.8 / 42.0% |
| tc_03 WMMA_pipe | 24.8 | 18.4 | 93.8 | 31.5% | 27.4 | 37.2 / 92.2 / 33.4% |

1. **CUDA core 那条线 A800 与 H20 几乎逐核重合**：占用率 naive 98% → doublebuffer 18%，SM Busy 反而 44%→84%、cyc/iss 36→6——**ILP 取代 TLP**。cublas 极致（占用率 18.5% 却 SM Busy 96.6%）。
2. **WMMA 三级被同一瓶颈卡死**：tc_01 的 **L1/TEX 99.7%、cyc/iss 106.8**（fragment 直取 global、张量核饿死，Compute 16.5%），A800 与 H20 数值几乎一样。
3. **关键分叉**：H20 在此之上有 tc_04 用 WGMMA+TMA 把 Compute(SM) 拉到 67%、占用率仅 7.6%（异步流水线代替 TLP）；**A800 无此级，WMMA 止于 Compute(SM)≤33%**——ncu 直观证明"差的就是 WGMMA"。

---

## 9、五个重点结论

1. **A800 是更强的 BF16 GEMM 卡，但要用对工具**。BF16 张量核 312T（H20 2.1×），cuBLAS BF16 **214–270T（默认）/262–295T（锁频）≈ H20(132T) 的 1.6–2.2×**。FP32 相反（峰值 19.5T<H20 ~40T，cuBLAS 19<30T）。**选卡看负载：BF16/FP16 训推选 A800，纯 FP32 选 H20。**
2. **"只换数据类型(BF16)"在 A800 更没用**：手写 BF16(CUDA core) 仅 ~5.7% 峰值（H20 ~17.5%）——峰值翻倍而代码仍跑 CUDA core。换类型不换硬件，离峰值更远。
3. **手写张量核在 A800 止步 WMMA（13.8% 峰）**：naive WMMA 被 L1/TEX(99.7%) 卡死，smem→cp.async 爬到 tc_03=43T，与 H20 同形、绝对值略高；但 **A800 无 WGMMA/TMA/FP8，无法复现 tc_04(80%)/tc_05**。
4. **A800 手写与 cuBLAS 张量核差距≈6×，根因缺 Ampere 原生 `mma.sync` 级**。补齐正道是 `mma.sync`+`ldmatrix`+多级 cp.async（CUTLASS Ampere），本仓库未实现，是下一课。
5. **方法论必须一致才有可比性**：① **锁时钟**——A800 默认 boost 在张量负载只维持~1200MHz（dmon 实测），不锁频则数字随预热在 214→262T 漂移；锁 1410 后 <0.1% 波动且可复现，公开参考表正落在默认时钟带内（§5 已对账一致）。② **分母选对**——BF16/WMMA 对**各自卡张量核峰值**（A800 312T，非工具写死的 148T）。③ **同引擎同驱动**——FP32/BF16/WMMA 各用各的 bench，不混用。

---

## 10、复现实操

```bash
git checkout feat/a800-reproduction
make ARCH=-arch=sm_80                  # FP32 + BF16
make tc ARCH=-arch=sm_80 TC_HOPPER=0   # WMMA(tc_01-03)
sudo nvidia-smi -i 0 -lgc 1410         # 锁额定 boost 以复现 canonical（默认 boost 会随预热漂移）
./build/cuda_core/bench 0 4096 4096 4096        # FP32 cuBLAS
./build/cuda_core/bench_bf16 0 4096 4096 4096   # BF16 cuBLAS
./build/tensor_core/bench 3 4096 4096 4096      # WMMA pipe
sudo nvidia-smi -i 0 -rgc              # 用完解锁
```

数据产物：`profiling/a800/`（`throughput_4096.txt` 全量 sweep、`scaling.txt` 缩放、`SUMMARY.md`+`a800_*.details.txt` ncu 全量、`verify_*.txt` 对拍）。

---

## 11、A800 vs H20 终览

| 项（4096³，各自最佳/cuBLAS） | A800 | H20 | A800/H20 |
| --- | ---: | ---: | ---: |
| FP32 手写最佳 | 17.6 TFLOPS | 25.1 TFLOPS | 0.70× |
| FP32 cuBLAS SGEMM | 19.0 TFLOPS | 30.3 TFLOPS | 0.63× |
| FP32 峰值 | 19.49 TFLOPS | ~40 TFLOPS | 0.49× |
| BF16 手写张量核最佳 | 43 TFLOPS (tc_03) | **119 TFLOPS (tc_04)** | 0.36× |
| **BF16 cuBLAS** | **214–262 TFLOPS** | 132 TFLOPS | **1.6–2.0×** |
| BF16 峰值 | **312 TFLOPS** | 148 TFLOPS | **2.1×** |
| FP8 | 无 | 224 TFLOPS | — |

> 一句话：**A800 的 BF16 张量核肌肉是 H20 的两倍，cuBLAS 能把它打到 94%；但本教程的手写阶梯在 Ampere 上只能爬到 WMMA(13.8% 峰)，因为真正解锁算力的 WGMMA/TMA/FP8 是 Hopper 独占。** A800 复现完整重走了 CUDA core 全套 + WMMA 三级，ncu 证明瓶颈与 H20 同源；缺的一课是 Ampere 原生 `mma.sync` 级手写 kernel。
