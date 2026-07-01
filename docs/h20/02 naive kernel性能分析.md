对应用例：`kernels/cuda_core/01_naive.cu`（`naive_kernel`，bench id 1）。下文所有 ncu 数据均为 **H20（CC 9.0，78 SM）实测**，尺寸 **4096³**，`--launch-count 1`（剖析第一次 launch）。本机采计数器需 root + ncu 全路径（见 [01 性能分析方法论](01%20性能分析方法论.md) 开头）。

## 1.看大局
```
sudo /usr/local/cuda/bin/ncu --set basic \
    -k regex:"naive" \
    --launch-count 1 \
    ./build/cuda_core/bench 1 4096 4096 4096
```

```
  naive_kernel(...) (128, 128, 1)x(32, 32, 1), Context 1, Stream 7, Device 0, CC 9.0
    Section: GPU Speed Of Light Throughput
    ----------------------- ----------- -------------
    Metric Name             Metric Unit  Metric Value
    ----------------------- ----------- -------------
    DRAM Frequency                  Ghz          2.62
    SM Frequency                    Ghz          1.83
    Elapsed Cycles                cycle    75,032,607
    Memory Throughput                 %         73.51
    DRAM Throughput                   %          4.40
    Duration                         ms         41.00
    L1/TEX Cache Throughput           %         73.93
    L2 Cache Throughput               %          5.89
    SM Active Cycles              cycle 74,533,602.19
    Compute (SM) Throughput           %         73.49
    ----------------------- ----------- -------------
```

判读：Compute 73.49% 和 Memory 73.51% 几乎完全相等，且都约等于 L1/TEX(73.93%)。这不是"算力和带宽双满"，而是 **同一条 L1/TEX LSU 管线同时顶满了两个口径**——LSU 既进 SM 算力口径，它发出的访存又进 Memory 口径。真正的 DRAM 才 4.40%（H20 4TB/s HBM3 + 60MB L2 把 4096³ 的重复读取大量吸进片上），外部带宽几乎闲着。4096³ 实测 **3532 GFLOPS = FP32 峰值(~44T)的 8%、cuBLAS SGEMM(30264) 的 11.7%**——光"能算对"远不够。

## 2.看延迟

```
sudo /usr/local/cuda/bin/ncu --section WarpStateStats \
    --metrics \
smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio \
-k regex:"naive" --launch-count 1 ./build/cuda_core/bench 1 4096 4096 4096
```

```
    Warp Cycles Per Issued Instruction             cycle        37.78
    --------------------------------------------------------------------------- ----------- ------------
    smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio           inst        13.95
    smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio        inst         0.37
    smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio              inst         0.01
    smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio          inst         0.01
    --------------------------------------------------------------------------- ----------- ------------
```

`Warp Cycles Per Issued = 37.78` 是个很大的数值：每发射一条指令平均要熬 38 个周期。分解看四项 stall，**`long_scoreboard` 13.95 一项独大**（占 37.78 的 ~37%，ncu 的 OPT 提示也点名 "spends 14.0 cycles ... waiting for a scoreboard dependency on a L1TEX ... operation, about 36.9%"），其余三项几乎为 0。

`long_scoreboard` 专指 warp 在等**远距离内存操作**返回结果：

- global memory load（L2/DRAM）
- local memory load（寄存器溢出）

**不包括** shared memory（那是 `short_scoreboard`，这里 0.01 → naive 根本没用 smem）；也几乎没有 `mio_throttle`（naive 没有 LDS，MIO 队列不堵）。结论很干净：**naive 唯一的病是等全局内存 load 回来**。

```
 sudo /usr/local/cuda/bin/ncu --section SchedulerStats -k regex:"naive" --launch-count 1 ./build/cuda_core/bench 1 4096 4096 4096
```

```
    ---------------------------- ----------- ------------
    Metric Name                  Metric Unit Metric Value
    ---------------------------- ----------- ------------
    One or More Eligible                   %        42.22
    Issued Warp Per Scheduler                        0.42
    No Eligible                            %        57.78
    Active Warps Per Scheduler          warp        15.97
    Eligible Warps Per Scheduler        warp         3.14
    ---------------------------- ----------- ------------
```

H20 一个 SM 有 64 个 warp（4 个 scheduler，每个上限 **16** warp，注意不是老 Turing 的 8）。这里 `Active = 15.97` → **占用率顶满**（Achieved Occupancy 99.78%）；但 `Eligible = 3.14` → 平均每周期只有约 3 个 warp 是 ready 的，**57.78% 的周期一个能发的 warp 都没有**（No Eligible），scheduler 空转。即使把占用率拉满到 16 warp/scheduler，大家还是几乎同时卡在等 global load 上，谁也补不了谁的空窗。

## 3.看访存合并
```
sudo /usr/local/cuda/bin/ncu --metrics \
l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio,\
l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_st.ratio \
-k regex:"naive" --launch-count 1 ./build/cuda_core/bench 1 4096 4096 4096
```

```
    l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio      sector         2.50
    l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_st.ratio      sector            4
```

访存合并：ld 单位是 sector（1 sector=32B），32 个线程连续读 float 是 4×32=128B=4 sector，所以一次 ld 指令理想就是 4。这里 **ld=2.5**：读 B 时一个 warp 连续 → 4 sector，读 A 时一个 warp 同行广播 → 实际只碰 1 sector，两者一平均 ≈ 2.5。写 C **st=4**（完美合并）。**naive 的线程映射天然就是合并的**——所以不需要 siboehm 那个 "GMEM coalescing" 单独一级。

```
sudo /usr/local/cuda/bin/ncu --metrics \
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum \
-k regex:"naive" --launch-count 1 ./build/cuda_core/bench 1 4096 4096 4096
```

```
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum                        0
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum                        0
```

bank conflict ld/st 全 0——因为 naive 根本没用 shared memory。

## 结论

- `long_scoreboard` 13.95 碾压其它 stall → 病灶是**等全局内存返回**（L1TEX scoreboard 依赖）。
- `Warp Cycles/Issued = 37.78` → 每发一条指令要熬 38 周期，这种量级的等待只可能来自全局访存延迟，和 long_scoreboard 互相印证。
- `Active 15.97 / Eligible 3.14` → occupancy 顶满（16 warp/scheduler 是 H20 上限），但平均只有 3 个 warp ready，57.78% 周期无 warp 可发。**64 个 warp 几乎同时卡在等内存上，谁也补不了谁的位。** 访存合并(ld 2.5/st 4)和 bank(0/0)都没问题，瓶颈纯粹是延迟。

所以我们要解决的问题就是 load 时间过长：**把反复读的数据搬到读写更快的地方（shared memory），用复用摊薄 global 访问** → 见 [03 smem kernel性能分析](03%20smem%20kernel性能分析.md)。
