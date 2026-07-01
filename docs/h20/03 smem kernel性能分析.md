对应用例：`kernels/cuda_core/02_smem.cu`（`smem_kernel`，bench id 2）。下文所有 ncu 数据均为 **H20（CC 9.0，78 SM）实测**，尺寸 **4096³**，`--launch-count 1`（剖析第一次 launch）。本机采计数器需 root + ncu 全路径（见 [01 性能分析方法论](01%20性能分析方法论.md) 开头）。

## 1.看大局
```
sudo /usr/local/cuda/bin/ncu --set basic \
    -k regex:"smem" \
    --launch-count 1 \
    ./build/cuda_core/bench 2 4096 4096 4096
```

```
  smem_kernel(...) (128, 128, 1)x(32, 32, 1), Context 1, Stream 7, Device 0, CC 9.0
    Section: GPU Speed Of Light Throughput
    ----------------------- ----------- -------------
    Metric Name             Metric Unit  Metric Value
    ----------------------- ----------- -------------
    DRAM Frequency                  Ghz          2.62
    SM Frequency                    Ghz          1.83
    Elapsed Cycles                cycle    49,896,306
    Memory Throughput                 %         90.34
    DRAM Throughput                   %          7.02
    Duration                         ms         27.27
    L1/TEX Cache Throughput           %         90.65
    L2 Cache Throughput               %          8.83
    SM Active Cycles              cycle 49,676,957.88
    Compute (SM) Throughput           %         76.00
    ----------------------- ----------- -------------
```

先看一眼总账：Duration 从 naive 的 **41 ms 降到 27.27 ms**，确实变快了。这一步把反复读的 A/B tile 搬进了 shared memory，方向是对的。

但有个反直觉的地方：`L1/TEX Cache Throughput` 不但没降，反而从 naive 的 **73.93% 涨到了 90.65%**。说好的"减负"呢？关键在于——**shared memory 和 L1 是同一块物理 SRAM**，smem 的 load/store（`LDS`/`STS`）也走 L1/TEX 这条管线、也算进 L1/TEX 口径。我们只是把数据从"反复读 global"换成了"在 smem 里复用"，**访问的总次数并没有变少，只是搬到了片上**，所以这条 SRAM 管线反而被压得更满（90.65%）。`Memory Throughput 90.34%` 几乎完全顶在 L1/TEX 上，而真正的 `DRAM Throughput` 只有 7.02%（全局内存现在只在加载 tile 时碰一次，外部带宽基本闲着）。

`Compute (SM) Throughput 76.00%` 看着不算低，但 Memory(L1/TEX) 已经 90%+，谁是瓶颈得继续往下挖。先记一个结论：4096³ 实测 **5412 GFLOPS = 公平口径 cuBLAS SGEMM(30264) 的 17.9%、FP32 峰值(~44T) 的 12.3%**——比 naive 的 11.7% 好了一点，但离上限还很远。

## 2.看延迟
```
sudo /usr/local/cuda/bin/ncu --section WarpStateStats \
    --metrics \
smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio \
-k regex:"smem" --launch-count 1 ./build/cuda_core/bench 2 4096 4096 4096
```

```
    Warp Cycles Per Issued Instruction             cycle        43.66
    --------------------------------------------------------------------------- ----------- ------------
    smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio           inst         7.07
    smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio        inst         0.06
    smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio              inst        21.74
    smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio          inst         0.63
    --------------------------------------------------------------------------- ----------- ------------
```

`Warp Cycles Per Issued = 43.66`，每发一条指令平均要熬 44 个周期。分解四项 stall，病灶换了：

- **`mio_throttle` 21.74 一项独大**（占 43.66 的 ~50%，ncu OPT 也点名 "spends 21.7 cycles ... waiting for the MIO instruction queue to be not full ... about 49.8%"）。
- `long_scoreboard` 从 naive 的 **13.95 掉到了 7.07**——把数据搬进 smem 之后，等全局内存返回的时间确实砍掉了一半。
- `short_scoreboard 0.63`、`math_pipe_throttle 0.06`，仍然很小。

`mio_throttle` 指的是 **MIO(Memory Input/Output) 指令队列满了**——MIO 管线负责 shared memory 访问(`LDS`/`STS`)、LSU 指令以及部分特殊数学指令的发射。队列满，意味着 warp 想发一条 `LDS` 去读 smem，但发射口被前面排队的 MIO 指令堵死，发不进去。**naive 的病是 long_scoreboard（等 global），smem 把它换成了 MIO 拥塞**——典型的"按下葫芦浮起瓢"：global 的坑填上了，smem 这条管线又被 LDS 打满了。

这里有个值得停下来想的细节：**cyc/issue 反而从 naive 的 37.78 涨到了 43.66，可整体却更快了（41→27.27 ms）。** 看似矛盾，其实不矛盾——smem 版本因为数据被复用，**总指令条数少了非常多**。每条发射虽然更"贵"（要排 MIO 的队），但要发的指令本来就少，"少而每条更慢"仍然干得过"多而每条略快"。所以单看 cyc/issue 这一个数会被带偏，得结合总指令量一起看。

```
sudo /usr/local/cuda/bin/ncu --section SchedulerStats -k regex:"smem" --launch-count 1 ./build/cuda_core/bench 2 4096 4096 4096
```

```
    ---------------------------- ----------- ------------
    Metric Name                  Metric Unit Metric Value
    ---------------------------- ----------- ------------
    One or More Eligible                   %        36.53
    Issued Warp Per Scheduler                        0.37
    No Eligible                            %        63.47
    Active Warps Per Scheduler          warp        15.94
    Eligible Warps Per Scheduler        warp         2.62
    ---------------------------- ----------- ------------
```

H20 一个 scheduler 上限 16 warp。这里 `Active = 15.94`——占用率依旧顶满（Achieved Occupancy **99.66%**，和 naive 一样满）。但 `Eligible 只有 2.62`，`No Eligible 高达 63.47%`：**绝大多数周期一个能发的 warp 都没有，scheduler 空转。** 占用率拉满也救不了——因为这些 warp 的下一条指令几乎都是 `LDS`，而 MIO 队列满了发不进去，于是它们全部 not eligible，谁也补不了谁的空窗。一条完整的因果链：

```
LDS 指令密度过高（每个 FMA 都要先来一条 LDS）
   ↓
MIO 队列被塞满 → mio_throttle = 21.74（占 cyc/issue 的一半）
   ↓
绝大多数 warp 的下一条指令是 LDS，但 MIO 满了发不进去
   ↓
这些 warp 全部 not eligible
   ↓
Active 15.94/16 占满，但 Eligible 只剩 2.62 → No Eligible 63.47%
   ↓
scheduler 大量空转 → SM 算力口径(76%)上不去
```

## 3.看访存合并
```
sudo /usr/local/cuda/bin/ncu --metrics \
l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio,\
l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_st.ratio \
-k regex:"smem" --launch-count 1 ./build/cuda_core/bench 2 4096 4096 4096
```

```
    l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio      sector            4
    l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_st.ratio      sector            4
```

访存合并：`ld=4`、`st=4`，全部干净。naive 时 ld 还是 2.5（读 A 同行广播只碰 1 sector 拖低了平均），现在全局内存只剩**把 tile 从 DRAM 搬进 shared memory** 这一种访问，而这个搬运你写成了完美连续 → 32 线程连续读 128B → 干净的 4 sector，写也是连续的 4 sector。

**注意**：ld 从 2.5 变成 4，不是因为"算得多了"，而是因为现在全局内存只在 tile 加载时被碰一次、且这一次写得很规整；复用全发生在 shared memory 那一侧，和全局 sec/req 是两回事。

```
sudo /usr/local/cuda/bin/ncu --metrics \
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum \
-k regex:"smem" --launch-count 1 ./build/cuda_core/bench 2 4096 4096 4096
```

```
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum                  573,162
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum              12,801,929
```

bank conflict：smem store 这边 `op_st = 12.8M` 确实存在冲突（写进 smem 的布局有 bank conflict），`op_ld = 573,162` 量级小一些。但这俩**和真正的瓶颈(MIO 21.74)比起来都是小事**——就算把 bank conflict 全消干净，也只是少占一点 MIO 管线，根上的问题还是 LDS 指令太多。这里先记一笔，后面再收。

全局访存合并已经十分完美，但性能还是上不去，可以判断为 **MIO bound**——当前从 smem 取数据的速度跟不上计算了。本质是"一次 load 一次 compute"的节奏不行了，最好改成"**一次 load，多次 compute**"。换 shared memory 这一步已经把 DRAM 上的数据放到了读写更快的地方，但你花了大把时间把数据搬进 smem，结果只读写一次，这不血亏吗？得让搬进来的每个数被复用更多次，才能把这趟搬运的成本赚回来。

## 总结
- `mio_throttle 21.74`（占 cyc/issue 43.66 的一半）碾压其它 stall → 病灶是 **MIO 队列被 LDS 打满**。naive 的 long_scoreboard(13.95)已降到 7.07，等 global 的坑填上了，却换来了 MIO 拥塞。
- `cyc/issue 37.78→43.66` 反而涨了，但整体更快(41→27.27 ms)——因为 smem 复用后**总指令条数大幅减少**，"少而每条更慢"赢过"多而每条略快"，不能只盯单条指令的代价。
- `Active 15.94 / Eligible 2.62 / No Eligible 63.47%` → occupancy 顶满(99.66%)，但绝大多数周期下一条指令是被 MIO 卡住的 LDS，没 warp 可发。
- 访存合并 ld=4/st=4 已完美；smem `op_st` 有 12.8M bank conflict，但和 MIO 比是小事。

根因是 **LDS 指令密度过高**（每条 FMA 都得先来一条 LDS）。解决方向是**降低 LDS:FMA 比例**，让搬进来的每个数喂更多次计算：

- **register tiling**：把 smem 数据攒进寄存器再复用，一条 LDS 喂多个 FMA（根治）
- **float4 / LDS.128 向量化**：一条指令读 128 bit，LDS 指令数直接降到 1/4（缓解 MIO）

→ 见 [04 register tiling](04%20register%20tiling.md)。
