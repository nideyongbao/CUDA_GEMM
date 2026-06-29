对应用例：`kernels/cuda_core/08_warptile_vec.cu`（`warptile_vec_kernel`，bench id 10，float4 版 warp tile）。下文所有 ncu 数据均为 **H20（CC 9.0，78 SM）实测**，尺寸 **4096³**，`--launch-count 1`（剖析第一次 launch）。本机采计数器需 root + ncu 全路径（见 [01 性能分析方法论](01%20性能分析方法论.md) 开头）。文中还会把标量 warp tile（`07_warptile.cu`，`warptile_kernel`，bench id 9）拉出来当反面教材。

## 0、为什么要进行warp tile
所谓warp tile，就是规定一个warp负责算C中的那个区域的元素，其实之前已经有了隐式的warp tile 就是在一个block内按线程id行优先排列，下图之前一个warp负责的C的区域，其实按照之前的分析我们的thread tile已经尽可能的减少的bank conflict和增加访存合并，但只是针对于当前的分块，要是换一个分法，那么我们就要重新设计。
![](img/Pasted%20image%2020260606135707.png)其实之前的分法我们是设计过的，TN=4 是为了让每个 lane 用一个 **float4**(16 字节 = 半个 sector)一次访问连续 4 列。warp 的一行是 16 个 lane**(`laneIdx%8`),16 × float4 = 64 个 float = 256 字节 = 8个连续 sector**,每 2 个 lane 填满 1 个 sector,8 个 sector 全部满载有效数据**,无浪费 → 完美合并。整个 warp 的 4 行各占 4 个满载 sector,跨行只是把一条 store 拆成 4 段,每段独立满载，但是你要是换了一个block 的大小就没有这么凑巧了，一旦 block 尺寸不对齐(如 BN=68),warp 分区会切在非 sector 边界,边缘 warp 落进"只占一半的 sector",sector 里掺无效数据 → 合并崩。 **warp tile 的作用**:用显式 `WM/WN`(选成 sector 的倍数)把每个 warp 的区域对齐到 sector 边界,消除映射人为制造的非对齐。但矩阵尺寸本身的非对齐(N=68 的尾巴)warp tile 救不了,要靠 padding 或边界处理。

先说一句结论：warp tile 这步只有 **float4 版（id 10）** 才是赢的；如果你像很多教程那样写**标量 warp tile（id 9）**，在 H20 上它是一次**负优化**——比上一篇的普通向量化 kernel 还慢。原因后面用真数据摆出来。

## 1、观大局
```
sudo /usr/local/cuda/bin/ncu --set basic \
    -k regex:"w" \
    --launch-count 1 \
    ./kernels/cuda_core/bench 10 4096 4096 4096
```

```
  warptile_vec_kernel(...) (32, 64, 1)x(128, 1, 1), Context 1, Stream 7, Device 0, CC 9.0
    Section: GPU Speed Of Light Throughput
    ----------------------- ----------- -------------
    Metric Name             Metric Unit  Metric Value
    ----------------------- ----------- -------------
    DRAM Frequency                  Ghz          2.62
    SM Frequency                    Ghz          1.83
    Elapsed Cycles                cycle    11,306,944
    Memory Throughput                 %         66.36
    DRAM Throughput                   %          3.54
    Duration                         ms          6.18
    L1/TEX Cache Throughput           %         67.48
    L2 Cache Throughput               %         10.46
    SM Active Cycles              cycle 11,104,854.14
    Compute (SM) Throughput           %         69.54
    ----------------------- ----------- -------------
```

判读：先和上一篇 vectorized（doc 05，id 5）比。Duration **6.51 → 6.18 ms**，4096³ 吞吐 **22790 → 23971 GFLOPS（+5%，到了 cuBLAS SGEMM 30264 的 79.2%）**——这是目前最快的 CUDA core kernel。但真正的看点不是这 5%，而是 **L1/TEX 89.84% → 67.48%**：warp tile 终于把上一篇顶到快 90% 的 L1/TEX 压力**卸下来了**。道理就是 warp tile 引入了 warp 级的复用——同一个 warp 内 lane 之间共享了从 smem 取出的 regA/regB，per-thread 的 shared 访问次数降下来，L1/TEX 这条 LSU 管线就没那么挤了。Compute 67.64% → 69.54% 略升（不是 compute bound，访存口径一松，算力口径自然能多吃一点）。

Memory 66.36% 和 Compute 69.54% 大致持平，ncu 也直接给了 "Compute and Memory are well-balanced"。DRAM 才 3.54%（H20 的 4TB/s HBM3 + 大 L2 把 4096³ 的重复读几乎全吸进片上），外部带宽依旧闲着，瓶颈仍在片上访存这一侧。

顺手把**标量 warp tile（id 9）**也跑一遍当对照：

```
sudo /usr/local/cuda/bin/ncu --set basic \
    -k regex:"w" \
    --launch-count 1 \
    ./kernels/cuda_core/bench 9 4096 4096 4096
```

```
  warptile_kernel(...) (32, 32, 1)x(128, 1, 1), Context 1, Stream 7, Device 0, CC 9.0
    Section: GPU Speed Of Light Throughput
    ----------------------- ----------- -------------
    Metric Name             Metric Unit  Metric Value
    ----------------------- ----------- -------------
    Duration                         ms         11.57
    Memory Throughput                 %         80.82
    L1/TEX Cache Throughput           %         86.42
    Compute (SM) Throughput           %         45.26
    ----------------------- ----------- -------------
    Section: Launch Statistics
    Registers Per Thread             register/thread             211
    Section: Occupancy
    Achieved Occupancy                        %         12.46
```

看出问题了吗：标量版 Duration **11.57 ms**，4096³ 吞吐只有 **12843 GFLOPS**——比上一篇那个连 warp tile 都没有的普通向量化 kernel（22790）还**慢了一截**。这是一次**实打实的负优化**。罪魁是 **Registers Per Thread = 211**：标量 warp tile 把每个 lane 要算的一大片 acc 全摊成标量，外加把 regA/regB 一个个标量地从 smem 取，寄存器一下爆到 211，于是 `Block Limit Registers = 2`，**Achieved Occupancy 砸到 12.46%**（理论占用都只有 12.5%）。占用一塌，scheduler 没几个 warp 可调度，Compute 反而只剩 45.26%。

对照 float4 版（id 10）：同样的 warp tile 思路，**Registers 105、Achieved Occupancy 24.07%**（block (128,1,1)=128，grid (32,64,1)=2048，Theoretical Occupancy 25% 被寄存器卡住）。同样是被寄存器限占用，但 float4 把每个 lane 的访存合并成 128-bit、acc 也排得更紧，寄存器只用一半、占用翻倍，立刻就回到正轨并反超。**一句话：warp tile 本身不保证赢，是 float4 把寄存器压回来才让它赢。**

这里也顺便点一个贯穿全项目的主题：id 10 的占用率才 **24.07%**，却是目前最快的 CUDA core kernel——**低占用 + 高 ILP 照样能打赢高占用**。别一看见占用低就急着调 block。往下分析延迟，看它低占用为什么还能喂饱流水线。

## 2、看延迟
```
sudo /usr/local/cuda/bin/ncu --section WarpStateStats \
    --metrics \
smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio \
-k regex:"w" --launch-count 1 ./kernels/cuda_core/bench 10 4096 4096 4096
```

```
    --------------------------------------------------------------------------- ----------- ------------
    Metric Name                                                                 Metric Unit Metric Value
    --------------------------------------------------------------------------- ----------- ------------
    smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio           inst         0.74
    smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio        inst         0.04
    smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio              inst         0.00
    smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio          inst         1.19
    --------------------------------------------------------------------------- ----------- ------------

    Section: Warp State Statistics
    ---------------------------------------- ----------- ------------
    Metric Name                              Metric Unit Metric Value
    ---------------------------------------- ----------- ------------
    Warp Cycles Per Issued Instruction             cycle         5.45
    Warp Cycles Per Executed Instruction           cycle         5.45
    Avg. Active Threads Per Warp                                   32
    Avg. Not Predicated Off Threads Per Warp                    32.00
    ---------------------------------------- ----------- ------------
```

这组数据非常健康。`Warp Cycles Per Issued = 5.45`，对比上一篇 vectorized 的 **11.21** 直接砍了一半——每发一条指令平均只熬 5.45 个周期。四项 stall 也全压下去了：`short_scoreboard 1.19` 最大（等 smem/LDS），`long_scoreboard 0.74`（等 global），`mio_throttle` 基本 **0.00**（MIO 队列不再堵了，对照 vectorized 那篇 mio 一度冲到 9.59），`math_pipe_throttle 0.04`。**没有任何一项 stall 独大**——这正是低占用却跑得快的原因：每个 warp 的 ILP 足够高，等待短、补位快，不需要堆很多 warp 来互相填空窗。

继续看 scheduler：

```
sudo /usr/local/cuda/bin/ncu --section SchedulerStats \
    -k regex:"w" --launch-count 1 ./kernels/cuda_core/bench 10 4096 4096 4096
```

```
    ---------------------------- ----------- ------------
    Metric Name                  Metric Unit Metric Value
    ---------------------------- ----------- ------------
    One or More Eligible                   %        70.75
    Issued Warp Per Scheduler                        0.71
    No Eligible                            %        29.25
    Active Warps Per Scheduler          warp         3.85
    Eligible Warps Per Scheduler        warp         1.74
    ---------------------------- ----------- ------------
```

`Active = 3.85`（占用低，每个 scheduler 平均才不到 4 个 warp，和 24% 占用对得上），但 `One or More Eligible = 70.75%`、`No Eligible` 只剩 **29.25%**——对比 naive 那篇 57.78% 的空转、vectorized 那篇 62.08%，这里 scheduler 七成周期都有 warp 能发。`Eligible = 1.74` 也比之前高。**warp 不多，但个个 ready 得勤**，issue slot 利用率高，这就是低占用打赢高占用的微观证据。

## 3、看访存合并和blank conflict
```
sudo /usr/local/cuda/bin/ncu --metrics \
l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio,\
l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_st.ratio \
-k regex:"w" --launch-count 1 ./kernels/cuda_core/bench 10 4096 4096 4096
```

```
    -------------------------------------------------------------------- ----------- ------------
    Metric Name                                                          Metric Unit Metric Value
    -------------------------------------------------------------------- ----------- ------------
    l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio      sector        16.00
    l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_st.ratio      sector           16
    -------------------------------------------------------------------- ----------- ------------
```

全局 ld/st 都是 **16 sec/req**，和前面几篇差不多——这数和分块/向量化的搬运布局有关，warp tile 这一步没去动它（它本来就不是为了修这个）。真正变好的是上面 L1/TEX 压力卸下来、延迟砍半，所以速度才上来。接着看 bank conflict：

```
sudo /usr/local/cuda/bin/ncu --metrics \
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum \
-k regex:"w" --launch-count 1 ./kernels/cuda_core/bench 10 4096 4096 4096
```

```
    -------------------------------------------------------- ----------- ------------
    Metric Name                                              Metric Unit Metric Value
    -------------------------------------------------------- ----------- ------------
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum                  151,451
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum               21,938,185
    -------------------------------------------------------- ----------- ------------
```

读 conflict 只有 15 万，基本可忽略；问题集中在 **写 conflict 仍有 2190 万（op_st，写 As 时撞的）**。也就是说——warp tile 修好了 L1/TEX 压力和延迟，但**写 As 的 bank conflict 还没动**。

为什么只有 As 撞、Bs 不撞？看代码：Bs 是用 **128-bit float4 整体写**（`reinterpret_cast<float4*>(&Bs[bs_r][bs_c])[0] = t;`）。硬件把一个 warp 的 128-bit shared 访问**按每 8 个 lane 一个 phase 分批仲裁**——每 phase 8 lane 各占 4 bank、正好铺满 32 bank、phase 内不撞，所以 Bs 自始至终无冲突。而 As 因为要做转置存放（`As[as_c][as_r] = t.x; As[as_c+1][as_r] = t.y; ...`），那条 float4 被**拆成了 4 条标量写**，退回 32-bit 路径、32 lane 一起仲裁，相邻线程的目标 bank 撞在一起 → 这 2190 万就是这么来的。

（对照标量 warp tile id 9 会更直观：它读 As 也是标量，于是连 **op_ld 都撞到 8 亿多**、op_st 285 万，bank conflict 全面爆炸，这也是它 short_scoreboard 高、跑那么慢的一部分原因。float4 版至少把读这一侧的 conflict 压没了，只剩写 As 这一处。）

## 总结
目前其实除了AS的black conflict没有解决其他的都挺好的：

- vs 上一篇 vectorized（id 5）：Duration 6.51 → 6.18 ms，吞吐 22790 → 23971 GFLOPS（cuBLAS 的 79.2%）。**最大的功劳是 L1/TEX 89.84% → 67.48%**——warp tile 用 warp 级复用把顶满的 L1/TEX 压力卸了下来，顺带 cyc/issue 11.21 → 5.45、mio_throttle 归零、No Eligible 降到 29.25%。
- 占用率只有 24.07%（Registers 105），却是目前最快的 CUDA core kernel——**低占用 + 高 ILP > 高占用**，这是本项目反复出现的主题。
- 反例标量 warp tile（id 9）：Registers 211、占用 12.46%、Compute 45.26%、Duration 11.57 ms、吞吐 12843 GFLOPS，比没 warp tile 的普通向量化还慢。warp tile 想赢，**必须靠 float4 把寄存器压回来**。
- 唯一没解决的：写 As 的 bank conflict 还有 **2190 万**（op_st），全局 ld/st 仍是 16 sec/req。warp tile 治好了 L1/TEX 压力，但 As-write 的 bank conflict 没动——这正是下一篇 [07 blank conflict](07%20blank%20conflict.md) 要收拾的最后一块。
