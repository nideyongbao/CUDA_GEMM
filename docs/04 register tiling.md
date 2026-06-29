对应用例：`kernels/cuda_core/04_2Dblocktiling.cu`（`Dblocktiling_kernel`，2D register tiling，bench id 4）。下文所有 ncu 数据均为 **H20（CC 9.0，78 SM）实测**，尺寸 **4096³**，`--launch-count 1`（剖析第一次 launch）。本机采计数器需 root + ncu 全路径（见 [01 性能分析方法论](01%20性能分析方法论.md) 开头）。

这一版的核心改动：在 smem 的基础上让**一个线程负责一个 TM×TN（=8×4）的 C 小块**。内循环里先把 `As` 的一列预取进 `regA[TM]`、`Bs` 的一行预取进 `regB[TN]`，再做 TM×TN 个 FMA。这样一条 LDS 读进来的数据能喂很多次 FMA，**LDS:FMA 的比例被压下来**——这正是 smem 那一版 MIO 堵死的病根。block 也从 1024 线程缩到 128 线程（16×8）。

## 1、看大局
```
sudo /usr/local/cuda/bin/ncu --set basic \
    -k regex:"D" \
    --launch-count 1 \
    ./kernels/cuda_core/bench 4 4096 4096 4096
```

```
  Dblocktiling_kernel(...) (64, 64, 1)x(16, 8, 1), Context 1, Stream 7, Device 0, CC 9.0
    Section: GPU Speed Of Light Throughput
    ----------------------- ----------- -------------
    Metric Name             Metric Unit  Metric Value
    ----------------------- ----------- -------------
    DRAM Frequency                  Ghz          2.62
    SM Frequency                    Ghz          1.83
    Elapsed Cycles                cycle    17,321,671
    Memory Throughput                 %         78.71
    DRAM Throughput                   %          2.49
    Duration                         ms          9.47
    L1/TEX Cache Throughput           %         78.95
    L2 Cache Throughput               %         10.60
    SM Active Cycles              cycle 17,250,430.55
    Compute (SM) Throughput           %         59.01
    ----------------------- ----------- -------------
```

判读：**Duration 从 smem 的 27.27 ms 直接掉到 9.47 ms**，一大步。但有意思的是 `L1/TEX Cache Throughput` 反而**降**了（90.65% → 78.95%），`Compute (SM) Throughput` 也**降**了（76% → 59%）。

这里要专门纠正一个老 Turing（GTX-1650S，CC 7.5）文档里的误判：在 Turing 上 register tile 之后 L1/TEX 会冲到 98%，被解读成"撞上了 SRAM 墙"。**在 H20 上完全不是这样——register tile 反而把 L1/TEX 压下去了。** 原因是 H20 的 L2/SRAM 余量大得多（60MB L2 + 4TB/s HBM3），LDS 指令一旦变少，L1/TEX 这条管线立刻松开，根本没有所谓的 SRAM 墙。

`Compute (SM)` 降到 59% 也不用慌：它不再是"throughput 顶满"的状态了。kernel 已经不卡在管线吞吐上，而是转成了**延迟/占用率受限**——后面看延迟和占用率会非常清楚。

性能上，4096³ 实测 **15548 GFLOPS ≈ cuBLAS SGEMM(30264) 的 51.5%、FP32 峰值的 35.4%**。一步就把 naive→smem 之后的曲线又抬了一大截。

## 2.看延迟
```
sudo /usr/local/cuda/bin/ncu --section WarpStateStats \
    --metrics \
smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio \
-k regex:"D" --launch-count 1 ./kernels/cuda_core/bench 4 4096 4096 4096
```

```
    --------------------------------------------------------------------------- ----------- ------------
    Metric Name                                                                 Metric Unit Metric Value
    --------------------------------------------------------------------------- ----------- ------------
    smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio           inst         1.19
    smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio        inst         0.11
    smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio              inst         0.22
    smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio          inst         1.82
    --------------------------------------------------------------------------- ----------- ------------

    Section: Warp State Statistics
    ---------------------------------------- ----------- ------------
    Metric Name                              Metric Unit Metric Value
    ---------------------------------------- ----------- ------------
    Warp Cycles Per Issued Instruction             cycle         8.18
    Warp Cycles Per Executed Instruction           cycle         8.18
    Avg. Active Threads Per Warp                                   32
    Avg. Not Predicated Off Threads Per Warp                    31.92
    ---------------------------------------- ----------- ------------
```

这是这一版最关键的一组数：**`Warp Cycles Per Issued` 从 smem 的 43.66 暴跌到 8.18**。每发一条指令要熬的周期数砍掉了 80%。这就是 register tile 的核心收益——一条 LDS 读进 `regA/regB` 后跟着一大串 TM×TN 个 FMA，**LDS:FMA 比例被彻底压平，一次 LDS 喂很多次 FMA**，于是发射不再被访存指令卡着等。

最直接的印证是 **`mio_throttle` 从 21.74 掉到 0.22**——smem 那一版 MIO 队列被 LDS 塞爆的拥堵**彻底消失了**。剩下四项 stall（long_scoreboard 1.19 / short_scoreboard 1.82 / mio 0.22 / math 0.11）都很小且分布平均，没有哪一项再独大。

```
 sudo /usr/local/cuda/bin/ncu --section SchedulerStats -k regex:"D" --launch-count 1 ./kernels/cuda_core/bench 4 4096 4096 4096
```

```
    ---------------------------- ----------- ------------
    Metric Name                  Metric Unit Metric Value
    ---------------------------- ----------- ------------
    One or More Eligible                   %        59.37
    Issued Warp Per Scheduler                        0.59
    No Eligible                            %        40.63
    Active Warps Per Scheduler          warp         4.85
    Eligible Warps Per Scheduler        warp         1.76
    ---------------------------- ----------- ------------
```

这里要把"占用率"和"延迟隐藏方式"一起看，是这一版最重要的教学点。

先看占用率：寄存器从 32 飙到 **96/线程**，block 又从 1024 缩到 128 线程，于是每个 SM 能放的 warp 数大幅减少，**Achieved Occupancy 从 99.66% 掉到 30.34%**（block (16,8,1)=128 线程，grid (64,64,1)=4096，Registers/Thread=96）。

按老思路，占用率掉这么多应该很慌。但调度器数据说不慌：`Active = 4.85`、`Eligible = 1.76`，而 **`No Eligible` 只有 40.63%**——比 smem 那一版的 63% 还低。也就是说，warp 数量虽然少了一大截，但"无 warp 可发"的空窗反而更少了。

这正是这一版的范式转变：**不再靠堆 warp 数量（TLP）来藏延迟，而是改靠单线程内部的 ILP 来藏延迟。** 每个线程那 TM×TN 个互相独立的 FMA，本身就构成一长串可以乱序填充流水线的独立指令，少量 warp 也足以把延迟盖住。这是从 smem 到 register tile 在"怎么藏延迟"这件事上的根本切换。

## 3.看访存合并

```
sudo /usr/local/cuda/bin/ncu --metrics \
l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio,\
l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_st.ratio \
-k regex:"D" --launch-count 1 ./kernels/cuda_core/bench 4 4096 4096 4096
```

```
    -------------------------------------------------------------------- ----------- ------------
    Metric Name                                                          Metric Unit Metric Value
    -------------------------------------------------------------------- ----------- ------------
    l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio      sector         4.09
    l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_st.ratio      sector           16
    -------------------------------------------------------------------- ----------- ------------
```

下图是 AS 的搬运过程，可以看到 BK 设计得很好，刚好是 8 的倍数，访问一行刚好是一个 sector，所以一个 warp 访问的 sector 刚好就是 4，同样 B 也是 4，所以**读的访存合并（ld=4.09）是十分好的了**。
![](img/Pasted%20image%2020260606133301.png)


读没问题，问题出在写 C 上：**st = 16**（严重不合并）。原因很明显——register tile 之后一个线程负责 TM×TN 个 C 元素，写回时逐个 `C[global_row][global_col] = ...`，这些地址在显存里是跳着走的：
- 原因：一个线程负责 TM×TN 个 C 元素，直接逐个写 → 地址跳跃 → 不合并 → sec/req 被拉到 16。
- 解法：**写回时也向量化**——把 `acc` 的一行用 `float4`（`reinterpret_cast<float4*>`）一次写 128 bit，sec/req 能从 16 拉回到 4。这和读侧的 LDS.128 是配套的。

这就是下一篇 [05 vectorizer](05%20vectorizer.md) 要做的第一件事。
**![](img/Pasted%20image%2020260606134019.png)**



```
sudo /usr/local/cuda/bin/ncu --metrics \
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum \
-k regex:"D" --launch-count 1 ./kernels/cuda_core/bench 4 4096 4096 4096
```

```
    -------------------------------------------------------- ----------- ------------
    Metric Name                                              Metric Unit Metric Value
    -------------------------------------------------------- ----------- ------------
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum              268,799,510
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum                8,456,644
    -------------------------------------------------------- ----------- ------------
```

读 shared 时有一大堆 bank conflict：**op_ld 高达 268,799,510**（op_st 也有 8,456,644，但小得多）。这是从转置过的 `As` 里把一列读进 `regA` 时产生的——多个 lane 同时打到同一个 bank 的不同元素。看下面这个图很容易看出来：在 `As → regA` 搬运的时候发生了 bank conflict（多个线程访问同一个 bank 的不同元素）；而 `Bs → regB` 没有这个问题，因为 `Bs` 的布局正好是 `As` 的转置方向，刚好把 bank 错开了。

![](img/Pasted%20image%2020260605192018.png)


## 总结

register tile 这一步收益巨大：Duration 27.27 → 9.47 ms，**Warp Cycles Per Issued 43.66 → 8.18**，`mio_throttle` 21.74 → 0.22，MIO 拥堵彻底消失。同时它把延迟隐藏的范式从"堆 warp（TLP）"切换成了"单线程 ILP"——占用率掉到 30.34% 但 No-Eligible 反而更低（40.63%）。在 H20 上 L1/TEX 不升反降（90.65→78.95%），**没有老 Turing 文档说的那道 SRAM 墙**。

但还留了两个明确指向下一步的问题：

- **写 C 不合并（st sectors = 16）**：一个线程写 TM×TN 个分散的 C 元素 → 地址跳跃。解法是**写回向量化（float4）**。
- **shared op_ld bank conflict = 268.8M（巨大）**：从转置过的 `As` 读 `regA` 时多 lane 撞同一 bank。

这两点正是 [05 vectorizer](05%20vectorizer.md) 要解决的：**用 float4 把读侧 LDS.128 和写侧 STG.128 一起向量化**，顺带缓解 AS 读的 bank conflict。
