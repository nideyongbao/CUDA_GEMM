# result/ — 跨机型执行快照 & 结果总表

本目录存放 `run_all.sh` 每次运行的**全部瞬时产物**（逐示例日志、遥测、profiling、汇总），
一次运行一个 `<时间戳>/` 目录，用于**跨轮次 / 跨机器对比**。本 README 同时是**跨机型结果总表**。

---

## 0. 跨机型结果总表（headline @ 4096³）

> 口径统一由 [`scripts/gpu_specs.py`](../scripts/gpu_specs.py) 提供：BF16 dense 峰值手工维护，
> **FP8 峰值 = 2×BF16、FP4 = 4×BF16 按计算能力(CC)推导**；张量核每个用例对**自己精度的峰值**算利用率。
> A800/H20 用 `--lock` 锁额定频（可复现、公平）；A10G/L4 由 Modal 云端跑**默认时钟**（云容器不能锁频）。

| GPU | 架构 / CC | 时钟策略 | FP32峰 / cuBLAS | BF16峰 / cuBLAS | 手写TC · BF16 路径 | FP8峰 / 手写TC · FP8 |
| --- | --- | --- | --- | --- | --- | --- |
| **A800-SXM** | Ampere sm_80 | lock@1410 | 19.5T / **19.0T** (98%) | 312T / **264.7T** (85%) | **150.3T** (48%, tc_06 mma.sync=3.5×tc_03) | — (Ampere 无 FP8) |
| **H20** | Hopper sm_90 | lock@1980 (~1830实测) | 39.5T / **28.0T** (71%) | 148T / **134.0T** (91%) | **120.6T** (82%, WGMMA) | 296T / **226.1T** (76%) |
| **A10G** | Ampere sm_86 | Modal 默认¹ | 31.2T / **14.1T** (45%) | 125T / **92.7T** (74%) | **25.1T** (20%, WMMA) | — (Ampere 无 FP8) |
| **L4** | Ada sm_89 | Modal 默认¹ | 30.3T / **13.3T** (44%) | 121T / **83.5T** (69%) | **25.4T** (21%, WMMA) | 242T / **未跑**² |

¹ 云容器无法锁频，只有默认一轮；A10(150W)/L4(72W) 是**小功耗卡**，默认时钟下大尺寸会撞功耗/温度墙降频——
  L4 尺寸缩放里 BF16 从 2048³ 的 80.5T 掉到 8192³ 的 66.7T 就是证据（与 platform 仓库 A10/L4 的 open→sustained MFU 跌幅一致）。
  故 A10G/L4 的 %峰值低于 A800/H20 有一半是**降频**而非效率差；要公平须锁频，云上做不到。
  （另注：Modal 的 "A10G" 实例驱动上报名为 `NVIDIA A10`、CC 8.6，故按 A10 规格 72 SM 记。）
² **L4(Ada) 有 FP8 硬件（峰值 242T=2×121T），但本仓库的 FP8 kernel 走 Hopper 独占的 WGMMA 指令**，
  Ada 上跑不了；要在 L4 测 FP8 需另写 `mma.sync` 版本（见 [跨代际适配设计 §5](../docs/跨代际适配设计.md)）。

**读表的四条主线：**
1. **Ampere 峰值高、补上 mma.sync 后手写能吃到近一半**：A800 BF16 峰值 312T 是 H20（148T）的 2.1×。手写阶梯**原止步 WMMA 只到 43T（14%峰）**，本轮补上 Ampere 原生 **tc_06（`mma.sync`+`ldmatrix`+多级 `cp.async`）→ 150.3T（48%峰）= 3.49×tc_03**，绝对值已**反超** H20 手写最佳（120.6T）。ncu 证据：WMMA 的 `load_matrix_sync` 有 6 路 bank 冲突把 L1/SMEM 管线打满(93.8%)、张量核挨饿；ldmatrix 消除冲突后瓶颈变成寄存器限占用率(延迟墙)。Ampere 没有 WGMMA/TMA，天花板本就靠 TLP/ILP，cuBLAS 能到 85%。详见 [docs/a800/Ampere mma.sync 张量核](../docs/a800/Ampere%20mma.sync%20张量核.md)。
2. **Hopper 峰值小、但手写打得满**：H20 靠 **WGMMA+TMA+warp specialization** 的异步多级流水线，手写 BF16 到 **82% 峰**、FP8 到 **76% 的 FP8 峰（226T）**——单卡手写就超过 A800 手写最佳 5×。
3. **FP8 让同一条流水线吞吐翻倍**：H20 FP8 226T ≈ BF16 手写 120T 的 1.9×，因为 FP8 张量核吞吐是 BF16 的 2×（296T vs 148T）。**这也是为什么 FP8 的利用率必须对 296T 算（76%），而不是对 148T 算（会得出不可能的 153%）**——详见下方「FP8 口径修正」。
4. **入门卡（A10/L4）：手写止步 WMMA，且小功耗卡默认时钟会降频**：A10G/L4 与 A800 同属「无 WGMMA」阵营，手写只到 WMMA 20–21% 峰；且 72–150W 的功耗墙让默认时钟下 cuBLAS 也只 69–74% 峰（大尺寸更低）。它们验证了「跨代际能跑通、口径正确、精度门控对（无 FP8 假象）」——正是本轮适配的目标。

各机完整报告：
- A800-SXM：[`20260701_093537_both/locked/00_summary.md`](20260701_093537_both/locked/00_summary.md)（含新增 tc_06 mma.sync；旧 `20260701_065155_both` 为补 tc_06 前基线）
- H20：[`20260701_150344_both/locked/00_summary.md`](20260701_150344_both/locked/00_summary.md)（软链 `latest`）
- A10G（Ampere sm_86）：[`A10_20260701_074243_modal/00_summary.md`](A10_20260701_074243_modal/00_summary.md)（Modal 实测）
- L4（Ada sm_89）：[`L4_20260701_073953_modal/00_summary.md`](L4_20260701_073953_modal/00_summary.md)（Modal 实测）

### FP8 口径修正（本轮重点）
H20 旧汇总把 FP8 吞吐(226T)对 **BF16** 峰值(148T)算利用率，得出「**152.7% 峰值**」这种物理上不可能的数。
根因是 `gemm_summary.py` 对所有张量核行一律套 BF16 峰值——而 kernel 原始日志本就同时打印了
`util(vs296T)=76.4%`（对 FP8 峰值，正确）与 `util(vs148T)=152.7%`（对 BF16 峰值，选错了）。
已修正为**按精度选峰值**，FP8 = 76.4% of 296T。完整分析见
[docs/FP8 利用率口径修正与 H20 结果分析](../docs/h20/FP8-利用率口径修正与H20结果分析.md)。

---

## 1. 仓库的四层分离（约定）

| 层 | 位置 | 内容 | 随运行变化 |
| --- | --- | --- | --- |
| **代码** | `kernels/` `include/` `scripts/` `Makefile` `run_all.sh` `modal/` | 只放代码/构建/工具/云入口 | 否（固化） |
| **算力事实** | `scripts/gpu_specs.py` | 各卡峰值/SM/精度支持的唯一真源 | 否（人工更新，加卡改一行） |
| **文档** | `docs/<机型>/` + `docs/跨代际适配设计.md` | 分析与结论（永久） | 否 |
| **参考基线** | `baselines/{cuda_core,tensor_core,a800}/` | 各机型策展好的 **ncu 参考快照** | 否（人工更新） |
| **瞬时快照** | **`result/<时间戳>/`** | 每次执行的逐示例日志 + 遥测 + profiling + 汇总 | **是（每跑一次多一份）** |

> 一句话：**代码目录只放代码；每次跑出来的日志/profiling 都进 `result/<时间戳>/`**，永久物与瞬时物互不污染。

## 2. 每个 `<时间戳>/` 里有什么

```
00_summary.md / .txt      汇总报告（算力/峰值/正确性/三张性能表/结论；峰值口径来自 gpu_specs.py）
00_timings.tsv            每步耗时 + 状态
00_clock_policy.txt       本轮时钟策略（default / locked@…）
telemetry_dmon.txt        全程遥测：每秒 SM时钟/显存时钟/功耗/温度/throttle（判断降频的证据）
01_fingerprint.txt        平台指纹（GPU/CC/CUDA/CPU/torch）
02_build.log              编译输出（用的 -arch / TC_HOPPER）
verify/
  cuda_core_fp32/<id>_<name>.log   ← 每个 FP32 kernel 单独的对拍日志（id 0–12）
  cuda_core_bf16/<id>_<name>.log   ← 每个 BF16 kernel
  tensor_core/<id>_<name>.log      ← 每个张量核用例（WMMA / Hopper 上含 WGMMA/FP8）
bench/
  cuda_core_fp32/<id>_<name>_<size>.log   ← 每个 kernel 单独的性能日志（含尺寸）
  cuda_core_bf16/<id>_<name>_<size>.log
  tensor_core/<id>_<name>_<size>.log
09_scaling.log            尺寸缩放（1024→8192³）
10_autotune.log           autotune 配置扫描
profiling/                （仅 --ncu）本轮新采的 ncu：<kernel>.ncu-rep + .details.txt
```

## 3. `--both`：归档基线（默认+锁频一次出对比）

`bash run_all.sh --both` 一次跑两轮，产 `result/<时间戳>_both/`：

```
<ts>_both/
  default/    完整一轮(默认时钟，真实开箱值)——结构同上
  locked/     完整一轮(锁额定 boost，可复现满频上限)——结构同上
  00_compare.md   逐 kernel 默认 vs 锁频 差异表 + 时钟 + 结论(scripts/gemm_compare.py 生成)
```

这是**推荐的归档方式**：一份快照同时给「真实开箱值(default)」与「可复现满频上限(locked)」，并直接量化 boost 抖动对每个 kernel 的影响（H20 上默认轮个别 kernel 被欠采 5–8%）。**跨机型公平对比 MFU 请用 locked。**
（云端 Modal 无法锁频，只有默认一轮——对比时以遥测实测时钟为准。）

## 4. 怎么对比

```bash
# --both 的现成对比：
cat result/<ts>_both/00_compare.md

# 手动对比两轮(同机不同时钟 / 跨机)：
python3 scripts/gemm_compare.py result/<A> result/<B>

# 重生成某轮汇总(改了 gpu_specs/summary 后，无需 GPU，纯解析日志)：
python3 scripts/gemm_summary.py result/<dir> "<GPU名>" "<CC>" "<时钟策略>" "<额定MHz>"

# 跨机型横排：
for d in result/*/; do [ -f "$d/00_summary.txt" ] && { echo "== $d =="; cat "$d/00_summary.txt"; }; done
```

## 5. 在别的卡上跑（本地 or 云端）

- **本地**：`bash run_all.sh --both`（自动按 CC 选 arch / 张量核用例，见 [docs/跨代际适配设计](../docs/跨代际适配设计.md)）。
- **云端（你没有的卡）**：`GPU_TYPE=L4 uv run modal run modal/run_gemm.py`（见 [modal/README](../modal/README.md)），
  跑完 `modal volume get cuda-matmul-gemm-results <tag>` 拉回，放进 `result/` 即可并入上面的总表。

## 6. git 约定

- 入库：`.log / .txt / .md / .tsv / telemetry / profiling/*.details.txt`（文本，便于 diff 与留档）。
- 忽略：`result/latest`（移动软链）、`result/**/*.ncu-rep`（大二进制，本地生成）、`__pycache__/`。见根 `.gitignore`。
