对应用例：`kernels/tensor_core/tc_01_wmma_naive.cu`（独立可执行 `tc_01_wmma_naive`）。

## 0、为什么从 WMMA 开始
前面 CUDA core 那条线（naive→…→double_buffer）的天花板是 FP32 计算单元，最高也就到 H20 FP32 峰值（~44 TFLOPS）的 57%、BF16 峰值（148 TFLOPS）的 ~17%。要往上走必须换引擎——用 **Tensor Core**。

WMMA(warp-level matrix multiply-accumulate) 是 Tensor Core 最入门的接口：不再是"一个 thread 算一个 C 元素"，而是 **一个 warp(32 线程) 协作**，把数据装进 `wmma::fragment`，由 Tensor Core 一条 `mma_sync` 吃掉一个 16×16×16 的小矩阵乘。本用例每个 warp 负责一个 16×16 的 C tile。

"朴素"在于：A/B 的 fragment **直接从 global memory 取**（`load_matrix_sync` 读 global），没有 shared memory 复用。它是这条 tensor core 阶梯的第 ① 级，专门用来暴露"光会用张量核还不够"。

## 1、观大局
```
sudo /usr/local/cuda/bin/ncu --set full -k regex:"wmma_naive_kernel" -s 1 -c 1 ./kernels/tensor_core/bench 1 2048 2048 2048
```

```
    Metric Name             Metric Unit Metric Value
    ----------------------- ----------- ------------
    Duration                         ms         1.18
    Compute (SM) Throughput           %        16.45
    Memory Throughput                 %        93.21
    L1/TEX Cache Throughput           %        99.78
    L2 Cache Throughput               %         8.85
    DRAM Throughput                   %         0.40
    Achieved Occupancy                %        71.17
    Registers Per Thread  register/thread        40
```
`L1/TEX Cache Throughput` 几乎打满（99.78%），而 `Compute (SM) Throughput` 只有 16.45% —— 张量核基本在干等。注意 `DRAM Throughput` 只有 0.4%：2048³ 的矩阵（每个 8MB）整个进了 60MB 的 L2，所以瓶颈不在 DRAM 带宽，而在 **L1/TEX 这条路**——也就是把 fragment 从 global（命中 L1）反复取的访存指令把 L1/TEX 打满了。

## 2、看延迟
```
sudo /usr/local/cuda/bin/ncu --section WarpStateStats \
  --metrics smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio \
  -k regex:"wmma_naive_kernel" -s 1 -c 1 ./kernels/tensor_core/bench 1 2048 2048 2048
```

```
    Warp Cycles Per Issued Instruction                                          cycle       109.25
    --------------------------------------------------------------------------- ----------- ------------
    smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio           inst        42.17
    smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio              inst        38.77
    smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio          inst        10.64
    smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio                   inst            0
    --------------------------------------------------------------------------- ----------- ------------
```
每发射一条指令平均要等 **109 个周期**（`Warp Cycles Per Issued = 109.25`）。把每项 stall 的 ratio 除以这个 cyc/issue 就是它占的周期比，头两块各占三分之一强：
- **long_scoreboard 42.17 → 38.6%**：warp 在等 `load_matrix_sync` 的 global 访存回来（数据没到，FMA/MMA 算不了）；
- **mio_throttle 38.77 → 35.5%**：访存指令太多，MIO 发射队列堵死，下一条 `LDG` 发不进去；
- short_scoreboard 10.64 → 9.7%，barrier 0（没用到 block 同步）。

头两项 **long_scoreboard + mio_throttle 加起来占 74%**（搬 fragment 等 global + 访存指令排队），就是结论：**kernel 完全卡在"搬 fragment"上，张量核被饿死。** 占用率虽然有 71%（靠很多 warp 在跑），但每个 warp 大部分时间都在等访存，TLP 也救不回来。

## 3、为什么张量核救不了它
WMMA 的 `mma_sync` 本身很快（一条指令 16×16×16），但它要求操作数先在寄存器 fragment 里。本版每个 K 步、每个 warp 都要从 global 取 A、B 两个 16×16 fragment，**算 1 次只搬 1 次、毫无复用**——和 CUDA core 的 naive 一模一样的病：算术强度太低，访存喂不上。SM Busy 仅 17.5% 就是铁证。

## 总结
WMMA 让我们第一次用上 Tensor Core，但朴素地从 global 取 fragment 会被 **L1/TEX + MIO** 卡死（warp cyc/issue 高达 109，long_scoreboard + mio_throttle 占 74%），张量核利用率（SM Busy）只有 17.5%，4096³ 实测 16685 GFLOPS = 148T 的 **11.3%**，甚至比最好的 CUDA core kernel（double_buffer 17.5%）还低。

这正好引出和 CUDA core 完全相同的下一步——**把 tile 搬进 shared memory 复用**（见 [09 tensor core - WMMA smem](09%20tensor%20core%20-%20WMMA%20smem.md)）：用 Tensor Core 是必要的，但远不充分。

> 本文 ncu 原始分项输出见 `profiling/tensor_core/doc_raw/tc_01_wmma_naive.txt`（与 cuda_core 的 doc_raw 对称，可由 `profiling/tensor_core/collect_doc_ncu.sh` 复跑）。
