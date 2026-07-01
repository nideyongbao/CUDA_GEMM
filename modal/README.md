# Modal 云端入口 — 在任意代际 GPU 上跑本仓库 GEMM 全量测试

`run_gemm.py` 把本仓库的 `run_all.sh` 搬到 [Modal](https://modal.com) 上，一条命令就能在
**你本地没有的卡**（A10G / L4 / A100 / H100 / B200 …）上跑「编译 + 正确性 + 性能 + 遥测 + 汇总」，
结果落到 Modal Volume，再拉回本地对比。

镜像**刻意很轻**：纯 CUDA C++ 只需 `nvcc + cuBLAS + make + nvidia-smi`，**不装 torch/triton**。
架构由 `run_all.sh` 按 `compute_cap` 自适应；SM 数在 torch 缺席时由 `scripts/gpu_specs.py` 兜底表提供。

## 用法

```bash
cd /shard_data/brooksli/workspace/0625/CUDA_MATMUAL/modal

# GPU 类型通过 env GPU_TYPE 在 import 时绑定给 @app.function(gpu=...)：
GPU_TYPE=A10G uv run modal run run_gemm.py      # Ampere sm_86  (A10/A10G)
GPU_TYPE=L4   uv run modal run run_gemm.py      # Ada    sm_89
GPU_TYPE=A100 uv run modal run run_gemm.py      # Ampere sm_80
GPU_TYPE=H100 uv run modal run run_gemm.py      # Hopper sm_90  (全量含 WGMMA/FP8)

# 快速版（跳过尺寸缩放 + autotune，省时省钱）：
GPU_TYPE=L4 BENCH_QUICK=1 uv run modal run run_gemm.py
```

Modal 支持的 GPU 串：`T4 / L4 / A10G / A100 / A100-80GB / L40S / H100 / H200 / B200`。
（注意 Modal 提供的是 **A10G**（Ampere sm_86），没有裸 A10；L4 是 Ada sm_89。）

## 取结果

跑完控制台会打印 `vol_tag`（形如 `L4_20260701_xxms_modal`）与下载命令：

```bash
modal volume get cuda-matmul-gemm-results <vol_tag>            # 拉整个结果目录
modal volume ls  cuda-matmul-gemm-results                       # 列出所有历史run
```

拉下来的目录结构与本地 `result/<时间戳>/` 完全一致（`00_summary.md`、逐示例 `bench/`·`verify/`、
`telemetry_dmon.txt` 等），可直接喂给 `scripts/gemm_compare.py` 跨机对比。

## 与本地 `--both` 的差别

- 云容器通常**无法锁频**（驱动禁止 `nvidia-smi -lgc`），故 modal 跑的是**默认时钟**一轮，
  不做 `--both`。跨机对比 MFU 时以 `telemetry_dmon.txt` 记录的**实测计算期时钟**为准。
- 云卡常有**功耗/温度墙**（如 A10 150W、L4 72W），默认时钟下大尺寸会降频——这正是
  遥测要一起看的原因（对比 platform 仓库 A10/L4 的 open→sustained MFU 跌幅）。

## 设计说明

见 [../docs/跨代际适配设计.md](../docs/跨代际适配设计.md)：GPU 选择、镜像分层、Volume 落盘、
以及「换代际只改 `gpu_specs.py` + `run_all.sh` 矩阵、kernel 零改动」的整体思路。本入口对齐
`platform/modal/run_platform.py` 的模式，便于两个仓库横向对照。
