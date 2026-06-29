# CUDA_MATMUAL

一个从 naive CUDA SGEMM 一步步优化到 shared memory、register tiling、float4、warp tiling、bank conflict 分析和 double buffering 的学习型矩阵乘法项目。

这个仓库的重点不是只给一个最快 kernel，而是把每一步优化为什么有效、什么时候无效、怎么用 Nsight Compute 判断瓶颈都记录下来。代码负责复现实验，`docs/` 负责沉淀分析过程。

## 项目内容

按"计算引擎"拆分：

- `kernels/cuda_core/`: 跑在 CUDA core 上的 kernel —— FP32 阶梯（`01_naive`…`10_doublebuffer`）+ 同一批 kernel 的 BF16 版（`bf16_cudacore.cu`，datatype 实验）。**驱动源码也在此**：`benchmark.cu`/`verify.cu`（FP32）、`bench_bf16.cu`/`verify_bf16.cu`（BF16）；`make` 后 `bench`/`verify`/`bench_bf16`/`verify_bf16` 可执行文件也输出在这里
- `kernels/tensor_core/`: 5 个 Tensor Core 用例（`tc_01_wmma_naive` / `tc_02_wmma_smem` / `tc_03_wmma_pipe` cp.async / `tc_04_wgmma_tma_ws` WGMMA+TMA / `tc_05_wgmma_fp8` FP8），各自保留专属 harness、编成 kernel-only 对象，由统一的 `bench`/`verify` 按 id(1–5) 派发运行（`bench.cu`/`verify.cu` + 派发表 `tc_cases.h`）
- `include/`: 公共宏、FP32/BF16 声明、autotuning 模板、tensor core 用例公共脚手架 `tc_common.cuh`
- `profiling/`: 性能分析，按路线分目录 `cuda_core/` + `tensor_core/`，各含 `run_ncu.sh` + `SUMMARY.md` + `ncu` 报告（以及下一步优化思路）
- `docs/`: GPU 硬件知识、性能分析方法论、每个 kernel 的 ncu 分析（00–07 CUDA core，08–12 Tensor core，13 总结）

矩阵计算接口统一为：

```cpp
C = alpha * A * B + beta * C
```

其中 A、B、C 都按 row-major 存储。cuBLAS reference 通过交换 A/B 和维度参数适配 row-major 结果。

## 环境要求

- NVIDIA GPU
- CUDA Toolkit，包含 `nvcc`
- cuBLAS
- GNU Make

本项目目标硬件是 **NVIDIA H20（Hopper，compute capability 9.0）**，Makefile 默认编译架构为：

```makefile
ARCH := -arch=sm_90a
```

`sm_90a` 是 Hopper 架构专用目标（architecture-specific），后续若扩展 WGMMA / TMA 等 Hopper 张量指令也需要它。

> 注意：旧版本默认是 `sm_75`（Turing）。在 H20 上用 `sm_75` 编译出的二进制其实**也能跑**——因为 fatbin 里带了 `compute_75` 的 PTX，驱动会在加载时 JIT 成 sm_90 SASS——但会有首次启动 JIT 开销、且不是 Hopper 原生调优代码，性能数字不可信。务必用 `sm_90a`。

换其他显卡时按设备改 `ARCH`：Ada 系列 `sm_89`、A100 `sm_80`、Turing `sm_75`。改完头文件或架构后执行 `make clean && make`。

## 编译

```bash
make        # CUDA core：FP32 bench/verify + BF16 bench_bf16/verify_bf16
make tc     # Tensor Core：统一 bench/verify 驱动（按 id 1–5 派发到 tc_01..tc_05）
```

产物按计算引擎分目录：`make` 生成 `kernels/cuda_core/` 下的 `bench`/`verify`（FP32）与 `bench_bf16`/`verify_bf16`（BF16）；`make tc` 生成 `kernels/tensor_core/bench` 和 `kernels/tensor_core/verify`。

清理生成文件：

```bash
make clean
```

## 正确性验证

默认验证 `id=0`，矩阵尺寸为 `1024 x 1024 x 1024`：

```bash
./kernels/cuda_core/verify
```

指定 kernel 和矩阵尺寸：

```bash
./kernels/cuda_core/verify <kernel_id> <M> <N> <K>
```

例子：

```bash
./kernels/cuda_core/verify 10 1024 1024 1024
```

`verify` 会跑当前 kernel 和 cuBLAS reference，然后输出：

- `max_abs_err`
- `max_rel_err`
- `bad_count`
- `PASS` / `FAIL`

当前误差阈值在 [kernels/cuda_core/verify.cu](kernels/cuda_core/verify.cu) 中设置为 `1e-2` 量级。

## 性能测试

默认 benchmark：

```bash
./kernels/cuda_core/bench
```

指定 kernel 和矩阵尺寸：

```bash
./kernels/cuda_core/bench <kernel_id> <M> <N> <K>
```

例子：

```bash
./kernels/cuda_core/bench 10 4096 4096 4096
```

输出格式类似：

```text
warptile_vec_kernel: M=4096 N=4096 K=4096, time=..., GFLOPS=...
```

autotuning 扫描入口：

```bash
./kernels/cuda_core/bench autotune 4096 4096 4096
```

## BF16（CUDA core 对照）

`make` 默认还会生成 `bench_bf16` / `verify_bf16`：把 CUDA core 那批 kernel 改成 BF16 输入 / FP32 累加（**仍跑 CUDA core**），用来证明"光换数据类型不碰张量核没用"。

```bash
./kernels/cuda_core/bench_bf16 <id> M N K     # 输出 GFLOPS 及对 148 TFLOPS BF16 峰值的利用率
./kernels/cuda_core/verify_bf16 <id> M N K    # 与 cuBLAS BF16 对拍
```
id 0=cuBLAS BF16，1–12 与 FP32 同名。结论：全量天花板只有 **~17%**（compute bound 在 FP32 单元上，BF16 只省了一半访存）。

## Tensor Core 系列（统一 bench/verify + ncu）

`make tc` 把 5 个用例编成 kernel-only 对象，链接成两个统一驱动（`tc_04/tc_05` 用 TMA，需 `-lcuda`），按 id(1–5) 派发：

```bash
make tc
# verify：对拍参考，打印 PASS/FAIL（tc_05 FP8 无简单 cuBLAS 路径，用 CPU double，自动在 ≤512³ 对拍）
./kernels/tensor_core/verify <id> [M N K]
# bench：纯计时 + 利用率
./kernels/tensor_core/bench  <id> [M N K]

# 例：
./kernels/tensor_core/bench  4 4096 4096 4096   # tc_04 WGMMA（不带参数默认 id=4 @ 4096³）
./kernels/tensor_core/verify 5 512 512 512      # tc_05 FP8 对拍 CPU double
```

手写 tensor core 完整阶梯（bench 4096³，对 148T；FP8 对 296T）：

| id | 用例 | 技术 | util |
| --- | --- | --- | ---: |
| 1 | tc_01 WMMA_naive | warp 级，global 直取 | 11.3% |
| 2 | tc_02 WMMA_smem | + shared memory 复用 | 20.1% |
| 3 | tc_03 WMMA_pipe | + cp.async 多级流水线 | 25.7% |
| **4** | **tc_04 WGMMA** | warpgroup 异步 + TMA + warp specialization | **80.6%**（8192³ 88%）|
| 5 | tc_05 WGMMA_fp8 | 同上换 FP8 | 75.8% /296T（224 TFLOPS）|
| — | 参考 cuBLAS BF16 (fair) | — | 89.1% |

ncu 剖析见 `profiling/tensor_core/SUMMARY.md`（CUDA core 那半见 `profiling/cuda_core/SUMMARY.md`）；逐例分析见 docs 08–12。核心结论：**warp 级 WMMA 靠高占用率藏延迟（张量核饿死，≤26%）；warpgroup 级 WGMMA 靠异步流水线，占用率仅 7.6% 却 SM Busy 83%**——这才是逼近 148T 的路。

> 注：cuBLAS 基线务必让 handle 复用——`cublasCreate/Destroy` 约 0.33ms/次，放进计时循环会把 cuBLAS 严重低估（4096³ 89%→68%），制造"手写反超 cuBLAS"假象。本项目已修复。详见 docs/13 §5。

## Kernel Id（CUDA core / FP32）

`kernels/cuda_core/bench` / `verify` 的注册表（Tensor Core 的 id 1–5 见上面「Tensor Core 系列」）：

| id | Kernel | 说明 |
| --- | --- | --- |
| 0 | cuBLAS reference | row-major 适配的 cuBLAS SGEMM 基线 |
| 1 | naive | 一个 thread 计算 C 的一个元素 |
| 2 | smem | A/B 分块搬入 shared memory |
| 3 | block tiling | 1D register tiling，一个 thread 计算多个结果 |
| 4 | 2D block tiling | TM/TN 二维 register tiling |
| 5 | vectorized | float4 向量化访存 |
| 6 | autotuning 64x64x8_8x4 | `launch_at<64,64,8,8,4>` |
| 7 | autotuning 64x64x16_8x4 | `launch_at<64,64,16,8,4>` |
| 8 | autotuning 64x64x8_8x8 | `launch_at<64,64,8,8,8>` |
| 9 | warp tile | warp-level tiling 基线 |
| 10 | warp tile vec | warp tiling + float4 版本 |
| 11 | bank conflict | 搬运映射与 shared memory bank conflict 分析版本 |
| 12 | double buffer | double buffering 版本 |

如果传入非法 id，程序会打印当前注册表。

## 文档导航

建议按这个顺序读：

1. [GPU 前置硬件知识](docs/00%20GPU前置硬件知识.md)
2. [性能分析方法论](docs/01%20性能分析方法论.md)
3. [naive kernel 性能分析](docs/02%20naive%20kernel性能分析.md)
4. [smem kernel 性能分析](docs/03%20smem%20kernel性能分析.md)
5. [register tiling](docs/04%20register%20tiling.md)
6. [vectorizer](docs/05%20vectorizer.md)
7. [warp tile](docs/06%20warp%20tile.md)
8. [bank conflict](docs/07%20blank%20conflict.md)

Tensor Core 系列（每个用例一篇，含 ncu 分析）：

9. [WMMA naive](docs/08%20tensor%20core%20-%20WMMA%20naive.md)
10. [WMMA smem](docs/09%20tensor%20core%20-%20WMMA%20smem.md)
11. [WMMA cp.async pipeline](docs/10%20tensor%20core%20-%20WMMA%20cp.async%20pipeline.md)
12. [WGMMA + TMA + warp specialization](docs/11%20tensor%20core%20-%20WGMMA%20TMA%20warp%20specialization.md)
13. [FP8 WGMMA](docs/12%20tensor%20core%20-%20FP8%20WGMMA.md)

**最终总结**（重点收敛）：[docs/13 H20 GEMM 复现总结.md](docs/13%20H20%20GEMM%20复现总结.md)。早期逐步复现详录见 [docs/H20复现结论.md](docs/H20复现结论.md)。

`docs/6.1.md` 到 `docs/6.5.md` 是阶段性交接文档，记录了每轮实验结论、性能对账、踩坑和下一步计划。想快速了解项目演进，可以从最新的 [docs/6.5.md](docs/6.5.md) 开始。

## Nsight Compute

常用 profiling 命令：

```bash
ncu --set full ./kernels/cuda_core/bench 10 4096 4096 4096
```

项目文档里主要关注这些指标：

- kernel duration / elapsed cycles
- Compute(SM) Throughput
- Memory Throughput
- DRAM / L1/TEX / L2 Throughput
- achieved occupancy
- registers per thread
- global memory coalescing
- shared memory bank conflict
- warp stall reason

一个核心经验是：ncu 上某个 throughput 很高，不等于它就是 binding bottleneck。真正要看的是优化它之后 elapsed cycles 是否下降。

## 当前状态

这个项目已经覆盖了 SGEMM 优化中的多条主线：

- 从全局内存 naive 访问开始
- 引入 shared memory 降低重复 global load
- 用 register tiling 提高单线程计算密度
- 用 float4 改善 global memory coalescing
- 用 warp tiling 改变 shared/register 复用结构
- 用 ncu 实证分析 bank conflict 是否真的卡在关键路径上
- 开始引入 double buffering 做 latency hiding

后续可以继续做的方向：

- 固化更多尺寸下的 benchmark 表
- 针对不同 GPU 架构维护独立配置
- 继续完善 double buffering 和更系统的 autotuning 搜索
