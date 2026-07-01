对应用例：`kernels/tensor_core/tc_03_wmma_pipe.cu`（独立可执行 `tc_03_wmma_pipe`）。

## 0、为什么要异步流水线
[09 WMMA smem](09%20tensor%20core%20-%20WMMA%20smem.md) 把"等访存"从 109 cyc/issue 砍到 16，但 SM Busy 还只有 41%——因为它仍是"搬一块 → `__syncthreads` → 算一块 → 再搬"的**同步**节奏：搬 tile 的时候张量核闲着，算的时候搬运单元闲着。两件事用的是不同硬件，本可以重叠。

本用例做三件事：① 更大 block tile（128×64，提高复用）；② shared memory 改成 **NSTAGES=3 级 ring buffer**；③ 用 **`cp.async`**（`__pipeline_memcpy_async` + `__pipeline_commit` + `__pipeline_wait_prior`）异步预取后续 K-tile，让"搬下一块"和"算当前块"重叠。这就是"异步搬运"第一次出现。

> 它正是 Hopper TMA + mbarrier 多级流水线的同步原语版前身：cp.async ↔ TMA，commit/wait_prior ↔ mbarrier 的 phase 等待。结构（ring buffer + 预取 + 按 stage 等待）和 WGMMA 那版一模一样。

## 1、观大局
```
sudo /usr/local/cuda/bin/ncu --set full -k regex:"wmma_pipe_kernel" -s 1 -c 1 ./build/tensor_core/bench 3 2048 2048 2048
```

```
    Metric Name             Metric Unit Metric Value
    ----------------------- ----------- ------------
    Duration                         us       520.77
    Compute (SM) Throughput           %        37.20
    Memory Throughput                 %        86.08
    L1/TEX Cache Throughput           %        92.24
    L2 Cache Throughput               %         7.05
    DRAM Throughput                   %         0.90
    Achieved Occupancy                %        33.42
    Registers Per Thread  register/thread        66
```

## 2、看延迟
```
sudo /usr/local/cuda/bin/ncu --section WarpStateStats \
  --metrics smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio \
  -k regex:"wmma_pipe_kernel" -s 1 -c 1 ./build/tensor_core/bench 3 2048 2048 2048
```

```
    Warp Cycles Per Issued Instruction                                          cycle        27.72
    --------------------------------------------------------------------------- ----------- ------------
    smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio              inst         9.31
    smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio          inst         4.04
    smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio                   inst         3.15
    smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio           inst         1.21
    --------------------------------------------------------------------------- ----------- ------------
```
按 ratio÷cyc/issue（27.72）换算占比：头号 stall 从 smem 版的 scoreboard 变成了 **mio_throttle 9.31 → 33.6%**——cp.async 把"等数据回来"(long_scoreboard 已掉到 1.21 → 4.4%) 这件事异步化了，剩下的瓶颈是发射这些访存/cp.async 指令本身的 MIO 队列压力。其余 short_scoreboard 4.04 → 14.6%、barrier 3.15 → 11.4%（多级 ring buffer 的 stage 等待）。

## 3、为什么 2048³ 看着没比 smem 快多少
注意一个反直觉点：**2048³ 上 tc_03(37.2%) 和 tc_02(37.8%) 的 Compute 几乎一样**，占用率反而更低（33% vs 42%）。原因是 2048 矩阵小、每个 block 干的活少，同步版靠较高占用率（TLP）已经把延迟藏得差不多了，异步流水线的额外收益被淹没。

但放大到 **4096³**，更大 tile 的复用 + 异步重叠才显出来：tc_02 20.1% → **tc_03 25.7%**（38063 GFLOPS），+35%。这也是教学点：**异步/流水线的价值在占用率低、延迟难藏时才充分体现**——而那正是 WGMMA 的常态。

## 总结
cp.async 多级流水线把搬运/计算重叠，4096³ 下把手写 WMMA 从 20.1% 推到 **25.7%**，并把 stall 从"等数据"变成"发指令排队"。但它仍是 **warp 级** WMMA + 靠占用率隐藏延迟，到 ~26% 基本见顶。

要再上一个数量级，必须换 Hopper 原生指令：warpgroup 级、操作数直接吃 smem、彻底异步的 **WGMMA + TMA**（见 [11 tensor core - WGMMA TMA warp specialization](11%20tensor%20core%20-%20WGMMA%20TMA%20warp%20specialization.md)）。

> 本文 ncu 原始分项输出见 `profiling/tensor_core/doc_raw/tc_03_wmma_pipe.txt`（与 cuda_core 的 doc_raw 对称，可由 `profiling/tensor_core/collect_doc_ncu.sh` 复跑）。
