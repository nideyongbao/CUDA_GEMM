对应用例：`kernels/cuda_core/05_vectorized.cu`（`vectorized_kernel`，用 float4 向量化访存，bench id 5）。下文所有 ncu 数据均为 **H20（CC 9.0，78 SM）实测**，尺寸 **4096³**，`--launch-count 1`（剖析第一次 launch）。本机采计数器需 root + ncu 全路径（见 [01 性能分析方法论](01%20性能分析方法论.md) 开头）。

## 1.看大局
```
sudo /usr/local/cuda/bin/ncu --set basic \
    -k regex:"v" \
    --launch-count 1 \
    ./kernels/cuda_core/bench 5 4096 4096 4096
```

```
  vectorized_kernel(...) (64, 64, 1)x(16, 8, 1), Context 1, Stream 7, Device 0, CC 9.0
    Section: GPU Speed Of Light Throughput
    ----------------------- ----------- -------------
    Metric Name             Metric Unit  Metric Value
    ----------------------- ----------- -------------
    DRAM Frequency                  Ghz          2.62
    SM Frequency                    Ghz          1.83
    Elapsed Cycles                cycle    11,914,803
    Memory Throughput                 %         88.52
    DRAM Throughput                   %         11.90
    Duration                         ms          6.51
    L1/TEX Cache Throughput           %         89.84
    L2 Cache Throughput               %         18.08
    SM Active Cycles              cycle 11,737,868.91
    Compute (SM) Throughput           %         67.64
    ----------------------- ----------- -------------
```

```
    Section: Launch Statistics
    Block Size                                    128   ← block (16,8,1)
    Grid Size                                   4,096   ← grid (64,64,1)
    Registers Per Thread             register/thread             63
    Achieved Occupancy                        %        48.10
```

先和上一版（[04 register tiling](04%20register%20tiling.md)，2D 寄存器分块）对比，这才是这一步真正的看点：

- **Duration 9.47 → 6.51 ms**，快了约 1.45 倍。
- **Registers/Thread 96 → 63**：float4 让编译器把寄存器打得更紧——一条 `LD.128` 顶四条 `LD.32`，中间临时量少了，寄存器压力直接降下来。
- **Achieved Occupancy 30.34% → 48.10%（升！）**：寄存器是 H20 上限制 occupancy 的那一项（ncu 的 OPT 也点名 "theoretical occupancy 50.0% is limited by the number of required registers"），寄存器一降，每个 SM 能塞下的 block 就多了，占用率跟着上去。
- **Compute (SM) 59 → 67.64%**。

这里要把结论说清楚，因为它和直觉相反：你一开始用 float4，多半是想"减少访存指令数、把 L1/TEX 打下来"。但 **L1/TEX 还是 89.84%，根本没降**（04 也是 ~90%）。float4 在这一步真正的红利不是省 L1/TEX，而是 **寄存器红利 → 更高的占用率**——少了一半多的寄存器，让更多 warp 同时在飞，靠并发把延迟摊薄，速度就上来了。这和整个项目复现下来的发现一致。

DRAM 才 11.90%，外部带宽依旧大量闲着（H20 4TB/s HBM3 + 60MB L2 把 4096³ 的重复读取吸进了片上），瓶颈还是在片内 L1/TEX 这条 LSU 管线上。

## 2.看延迟
```
sudo /usr/local/cuda/bin/ncu --section WarpStateStats \
    --metrics \
smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio \
-k regex:"v" --launch-count 1 ./kernels/cuda_core/bench 5 4096 4096 4096
```

```
    --------------------------------------------------------------------------- ----------- ------------
    Metric Name                                                                 Metric Unit Metric Value
    --------------------------------------------------------------------------- ----------- ------------
    smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio           inst         2.16
    smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio        inst         0.06
    smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio              inst         1.09
    smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio          inst         1.81
    --------------------------------------------------------------------------- ----------- ------------

    Section: Warp State Statistics
    ---------------------------------------- ----------- ------------
    Metric Name                              Metric Unit Metric Value
    ---------------------------------------- ----------- ------------
    Warp Cycles Per Issued Instruction             cycle        11.21
    ---------------------------------------- ----------- ------------
```

`Warp Cycles Per Issued = 11.21`（04 是 8.18，略微升了一点点），但更关键的是 **stall 已经被打散了**：long_scoreboard 2.16、short_scoreboard 1.81、mio_throttle 1.09、math_pipe 0.06——再没有一项独大。对比 naive 阶段 long_scoreboard 一项 13.95 碾压全场，现在是"哪一项都不算高、谁也不主导"的健康状态：等全局内存（long 2.16）、等 shared（short 1.81）、MIO 队列（mio 1.09）三者势均力敌。这说明前几步该治的延迟都治得差不多了，cyc/issue 这点小幅上升只是并发结构变化的副产物，不是新病灶。

```
 sudo /usr/local/cuda/bin/ncu --section SchedulerStats -k regex:"v" --launch-count 1 ./kernels/cuda_core/bench 5 4096 4096 4096
```

```
    ---------------------------- ----------- ------------
    Metric Name                  Metric Unit Metric Value
    ---------------------------- ----------- ------------
    One or More Eligible                   %        68.64
    Issued Warp Per Scheduler                        0.69
    No Eligible                            %        31.36
    Active Warps Per Scheduler          warp         7.70
    Eligible Warps Per Scheduler        warp         2.64
    ---------------------------- ----------- ------------
```

`No Eligible` 降到 **31.36%**——这是到目前为止最好的一次（naive 57.78%）。`One or More Eligible 68.64%` 意味着近七成的周期 scheduler 手里至少有一个能发的 warp，`Eligible 2.64` 也比之前更宽裕。占用率上去（48.10%）+ stall 打散，二者合力让 scheduler 不再频繁空转。`Active 7.70` 看着不高，是因为这一版 block 只有 128 线程（4 warp）、占用率理论上限 50%，但配合低 stall 已经够把流水喂得相当饱了。

## 3.访存合并

```
sudo /usr/local/cuda/bin/ncu --metrics \
l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio,\
l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_st.ratio \
-k regex:"v" --launch-count 1 ./kernels/cuda_core/bench 5 4096 4096 4096
```

```
    -------------------------------------------------------------------- ----------- ------------
    Metric Name                                                          Metric Unit Metric Value
    -------------------------------------------------------------------- ----------- ------------
    l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio      sector           16
    l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_st.ratio      sector           16
    -------------------------------------------------------------------- ----------- ------------
```

**ld = 16，访存合并很差**——理想下一个 warp 连续读 float4 应该只碰几个 sector，这里却要碰 16 个。问题出在读 A：

A 是行主序，`A[a_row*K + a_col]` 的地址主要由 **`a_row = blockIdx.y*BM + as_r`（而 `as_r = (tid*4)/BK = tid/2`）** 决定。看一个 warp(tid 0~31)：

```
tid:    0    1    2    3    4    5   ...
a_row:  0    0    1    1    2    2   ...   (相邻两线程同行)
a_col:  0    4    0    4    0    4   ...
```

所以 warp 内地址是：

```
t0: A[0*K + 0]     t1: A[0*K + 4]      ← 同一行,连续(这两个其实挨着)
t2: A[1*K + 0]     t3: A[1*K + 4]      ← 跳到下一行,地址 +K (=4096) 个 float!
t4: A[2*K + 0]     ...                 ← 又跳一行
```

**每两个线程就跳一整行(+K×4 字节 = +16KB)。** 32 个线程跨了 16 个不同的行，地址散布在 16 段相距 16KB 的位置 → 一次 LD 请求要碰十几个 cache sector → **sec/req = 16**。这就是读合并差的原因。换成 float4 只是把"每次读 4 个 float"打包了，并没有改掉"warp 里线程按行散开"这个根本的映射问题，所以照样不合并。写 C **st 也是 16**，同理是 `tid` 映射把一个 warp 撒到了多行上。



```
sudo /usr/local/cuda/bin/ncu --metrics \
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum \
-k regex:"v" --launch-count 1 ./kernels/cuda_core/bench 5 4096 4096 4096
```
```
    -------------------------------------------------------- ----------- ------------
    Metric Name                                              Metric Unit Metric Value
    -------------------------------------------------------- ----------- ------------
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum                  144,270
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum               36,777,844
    -------------------------------------------------------- ----------- ------------
```
写 shared 仍然有大量 bank conflict：**op_st = 36.8M**（很高），op_ld 只有 144k（很低）。主要还是把 A 写进转置布局的 `As` 时，每两个线程就会撞 bank——float4 把单次写变成了 `STS.128`，但落点的 bank 模式没变，照样冲突。下图就是这个 AS 写冲突：
![](img/Pasted%20image%2020260605201651.png)

## 总结

float4 这一步用 **寄存器红利 → 占用率 30.34%→48.10%** 换来了速度（Duration 9.47→6.51 ms），但它同时把两个被掩盖的布局问题暴露了出来：

- **全局读没合并（ld = 16）**：`a_row = tid/2` 让一个 warp 跨 16 行，地址散成 16 段。
- **shared 写 bank conflict（st = 36.8M）**：写转置 `As` 时每两个线程撞 bank。

这两件事同根同源——都是 `tid/16`、`tid%16`、`tid/2` 这种"一级 tid→地址"的随手映射，把 warp 内 32 个线程撒得到处都是，导致读 A、写 C 跨行不合并，写 As 撞 bank。

下一步就是引入 **warp tile**——"warp 这一中间层"：先决定每个 warp 管哪块连续的 C 区域，再决定 warp 内 32 个 lane 怎么排，目的就是让同一个 warp 的访存地址连续（合并）、LDS 地址错开 32 个 bank（无 conflict）、并能用上 broadcast。换句话说：register tile 解决"一个线程算多少、复用多少"（降 LDS 指令数），warp tile 解决"32 个线程怎么协同访存才高效"（合并 + 无 bank conflict + 广播）。你这一版测出来的那一堆访存问题，正是缺了 warp tile 这一层。

> 性能坐标：4096³ 实测 **22790 GFLOPS = cuBLAS SGEMM(30264) 的 75.2% = FP32 峰值的 51.8%**。已经很能打，但访存合并和 bank conflict 这两块还没修，后面 warp tile 接着治。

```
naive            → memory bound
+ block tile/smem → DRAM 复用,但 LDS 太多 → MIO bound
+ register tile   → 降 LDS:FMA,缓解 MIO
+ vectorize(float4)→ 寄存器红利→占用率↑,提速     ← 你在这,但访存映射乱:读不合并 + 写 As 撞 bank
+ warp tile       → 修好访存合并、bank conflict、广播  ← 下一步,治你现在的病
+ double buffer   → 预取隐藏延迟
```

→ 见 [06 warp tile](06%20warp%20tile.md)。
