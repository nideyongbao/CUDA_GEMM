# result/ — 每次执行的瞬时快照

本目录存放 `run_all.sh` 每次运行的**全部瞬时产物**（逐示例日志、遥测、profiling、汇总），一次运行一个 `<时间戳>/` 目录，方便**跨轮次 / 跨机器对比**。

## 仓库的三层分离（约定）

| 层 | 位置 | 内容 | 是否随运行变化 |
| --- | --- | --- | --- |
| **代码** | `kernels/` `include/` `scripts/` `Makefile` `run_all.sh` | 只放代码/构建/工具，**不掺任何日志或结果** | 否（固化） |
| **文档** | `docs/<机型>/` | 分析与结论（永久） | 否 |
| **参考基线** | `baselines/{cuda_core,tensor_core,a800}/` | 各机型策展好的 **ncu 参考快照**（文档引用的"golden"证据） | 否（人工更新） |
| **瞬时快照** | **`result/<时间戳>/`** | 每次执行的逐示例日志 + 遥测 + profiling + 汇总 | **是（每跑一次多一份）** |

> 一句话：**代码目录只放代码；每次跑出来的日志/profiling 都进 `result/<时间戳>/`**，永久物（代码/文档/参考基线）与瞬时物（每轮日志）互不污染。

## 每个 `<时间戳>/` 里有什么

```
00_summary.md / .txt      汇总报告（算力/峰值/正确性/三张性能表/结论）
00_timings.tsv            每步耗时 + 状态
00_clock_policy.txt       本轮时钟策略（default / locked@…）
telemetry_dmon.txt        全程遥测：每秒 SM时钟/显存时钟/功耗/温度/throttle（判断降频的证据）
01_fingerprint.txt        平台指纹（GPU/CC/CUDA/CPU/torch）
02_build.log              编译输出（用的 -arch / TC_HOPPER）
verify/
  cuda_core_fp32/<id>_<name>.log      ← 每个 FP32 kernel 单独的对拍日志（id 0–12）
  cuda_core_bf16/<id>_<name>.log      ← 每个 BF16 kernel
  tensor_core/<id>_<name>.log         ← 每个张量核用例（WMMA / Hopper 上含 WGMMA/FP8）
bench/
  cuda_core_fp32/<id>_<name>_<size>.log   ← 每个 kernel 单独的性能日志（含尺寸）
  cuda_core_bf16/<id>_<name>_<size>.log
  tensor_core/<id>_<name>_<size>.log
scaling.log               尺寸缩放（1024→8192³）
autotune.log              autotune 配置扫描
profiling/                （仅 --ncu）本轮新采的 ncu：<kernel>.ncu-rep + .details.txt
```

## `--both`：归档基线（默认+锁频一次出对比）

`bash run_all.sh --both` 一次跑两轮，产 `result/<时间戳>_both/`：

```
<ts>_both/
  default/    完整一轮(默认时钟)——结构同上
  locked/     完整一轮(锁频1410)——结构同上
  00_compare.md   逐 kernel 默认 vs 锁频 差异表 + 时钟 + 结论(scripts/gemm_compare.py 生成)
```

这是**推荐的归档方式**：一份快照同时给"真实开箱值(default)"与"可复现满频上限(locked)"，并直接量化 boost 抖动对每个 kernel 的影响。

## 怎么对比

```bash
# --both 的现成对比：
cat result/<ts>_both/00_compare.md

# 手动对比两轮(同机不同时钟 / 跨机)：
python3 scripts/gemm_compare.py result/<A> result/<B>
diff result/<A>/bench/cuda_core_fp32/00_cublas_ref_4096.log \
     result/<B>/bench/cuda_core_fp32/00_cublas_ref_4096.log

# 汇总横排：
for d in result/*/; do [ -f "$d/00_summary.txt" ] && { echo "== $d =="; cat "$d/00_summary.txt"; }; done
```

## git 约定

- 入库：`.log / .txt / .md / .tsv / telemetry / profiling/*.details.txt`（文本，便于 diff 与留档）。
- 忽略：`result/latest`（移动软链）、`result/**/*.ncu-rep`（大二进制，本地生成）。见根 `.gitignore`。
