对应用例：`kernels/tensor_core/tc_02_wmma_smem.cu`（独立可执行 `tc_02_wmma_smem`）。

## 0、为什么要加 shared memory
[08 WMMA naive](08%20tensor%20core%20-%20WMMA%20naive.md) 的结论是：张量核被"从 global 反复取 fragment"饿死（L1/TEX 99.8%、warp cyc/issue 109）。解法和 CUDA core 那条线一模一样——**先把 A/B tile 搬进 shared memory，再让多个 warp 反复复用**，把 global 的重复读取换成 smem 内部复用。

本用例的布局：block tile 64×64，block=128 线程=4 个 warp，排成 2×2；每个 warp 负责 32×32 = **2×2 个 16×16 wmma fragment**。每个 K 步先 128 线程协作把 A(64×16)、B(16×64) 搬进 smem，再从 smem 取 fragment 做 4 次 `mma_sync`。smem 直接用 bf16（wmma fragment 要 bf16）。

## 1、观大局
```
sudo /usr/local/cuda/bin/ncu --set full -k regex:"wmma_smem_kernel" -s 1 -c 1 ./build/tensor_core/bench 2 2048 2048 2048
```

```
    Metric Name             Metric Unit Metric Value
    ----------------------- ----------- ------------
    Duration                         us       682.08
    Compute (SM) Throughput           %        37.77
    Memory Throughput                 %        76.37
    L1/TEX Cache Throughput           %        82.76
    L2 Cache Throughput               %        10.26
    DRAM Throughput                   %         0.67
    Achieved Occupancy                %        42.01
    Registers Per Thread  register/thread        64
```
对比 naive：Duration 1.18ms → 682µs，`Compute (SM) Throughput` 16.5% → 37.8%。L1/TEX 还是高（82.76%），因为 smem 的读写也走 L1/TEX 这块 SRAM——但**性质变了**：之前是反复读 global，现在是 smem 内部复用。

## 2、看延迟
```
sudo /usr/local/cuda/bin/ncu --section WarpStateStats \
  --metrics smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio \
  -k regex:"wmma_smem_kernel" -s 1 -c 1 ./build/tensor_core/bench 2 2048 2048 2048
```

```
    Warp Cycles Per Issued Instruction                                          cycle        16.38
    --------------------------------------------------------------------------- ----------- ------------
    smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio           inst         5.59
    smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio              inst         2.24
    smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio          inst         2.22
    smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio                   inst         0.82
    --------------------------------------------------------------------------- ----------- ------------
```
关键变化：**warp cyc/issue 从 109 砍到 16.4**（`Warp Cycles Per Issued = 16.38`）—— smem 复用一下子把"等访存"的时间打下来了。各项 stall 用 ratio÷cyc/issue 换算占比：头号是 **long_scoreboard 5.59 → 34.1%**（注意这里是 *long* scoreboard，等的是 global load 回来，不是 smem 的 short scoreboard——short 只有 2.22 → 13.6%），mio_throttle 2.24 → 13.7%，barrier 0.82 → 5.0%。同时 ncu 的 ComputeWorkloadAnalysis 提示 **ALU 是最高管线（32.4%）**：现在的瓶颈变成了从 smem 把 fragment 装进寄存器（`load_matrix_sync`）以及地址计算这类 ALU/MIO 指令，张量核还是没完全喂饱（SM Busy 40.9%）。

## 3、占用率的代价
占用率从 naive 的 71% 掉到 42%（每线程寄存器 40→64，smem 也吃资源）。但因为 cyc/issue 大幅下降，整体反而快了——这说明 **occupancy 不是越高越好**，关键是"每个 warp 等得久不久"。这个 trade-off 后面 WGMMA 会推到极致（占用率 7%、却最快）。

## 总结
smem staging 是 tensor core 阶梯上最划算的一步：warp cyc/issue 109→16，`Compute (SM)` 16.5%→37.8%，4096³ 实测 16685→**29733 GFLOPS（11.3%→20.1%）**，一举超过最好的 CUDA core kernel。

但还没到头：load_matrix(smem→寄存器) + 同步 `__syncthreads` 让张量核仍有空档。下一步用 **cp.async 把搬运和计算重叠**（见 [10 tensor core - WMMA cp.async pipeline](10%20tensor%20core%20-%20WMMA%20cp.async%20pipeline.md)）。

> 本文 ncu 原始分项输出见 `baselines/tensor_core/doc_raw/tc_02_wmma_smem.txt`（与 cuda_core 的 doc_raw 对称，可由 `baselines/tensor_core/collect_doc_ncu.sh` 复跑）。
