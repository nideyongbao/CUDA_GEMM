对应用例：`kernels/tensor_core/tc_04_wgmma_tma_ws.cu`（独立可执行 `tc_04_wgmma_tma_ws`，需 `-lcuda`）。

## 0、为什么 cp.async+WMMA 到头了，要换 WGMMA
WMMA 是 **warp 级**：32 线程先把 fragment `load_matrix_sync` 装进寄存器，再 `mma_sync`。这个"先搬进寄存器"本身就是开销，且必须靠高占用率(很多 warp)来藏访存延迟。Hopper 给了一套全新的原生指令，把这两点都改了：

- **WGMMA**(`wgmma.mma_async`)：**warpgroup 级**（4 warp = 128 线程一起发一条），**异步**（fence/commit/wait_group），且**操作数可直接从 shared memory 取**（用 64-bit matrix descriptor 描述 smem 的 128B-swizzle 布局，不再过寄存器）。
- **TMA**(`cp.async.bulk.tensor`)：descriptor 化的批量异步拷贝，一个线程发起一整块 global→smem 搬运，自带 swizzle。
- **mbarrier**(`cuda::barrier` + `expect_tx` 字节计数)：做 TMA→WGMMA 的完成信号与 stage 复用。
- **warp specialization**：1 个生产者 warpgroup 专跑 TMA、1 个消费者 warpgroup 专跑 WGMMA，3 级流水线。

> 关键认识：**WGMMA 没法和 TMA/swizzle 拆开单独"最小化"**——它的 smem 操作数必须是 TMA 产出的 128B-swizzle core-matrix 布局，描述符魔数也和这套布局绑死。所以"最小 WGMMA"实际就是 TMA+WGMMA 这一对，这本身就是 Hopper 异步硬件协同设计的体现。本 kernel 移植自已验证正确的 LeetCUDA `hgemm_wgmma_fp32acc_stages_tn`，改 bf16 + 适配本项目 row-major（内部把 B 转置成 N×K 一次，不计时）。

## 1、观大局
```
sudo /usr/local/cuda/bin/ncu --set full -k regex:"wgmma_kernel" -s 1 -c 1 ./build/tensor_core/bench 4 2048 2048 2048
```

```
    Metric Name             Metric Unit Metric Value
    ----------------------- ----------- ------------
    Duration                         us       190.88
    Compute (SM) Throughput           %        67.03
    Memory Throughput                 %        16.78
    L1/TEX Cache Throughput           %        20.73
    L2 Cache Throughput               %        11.48
    DRAM Throughput                   %         2.47
    Achieved Occupancy                %         7.58
    Registers Per Thread  register/thread       154
    SM Busy                           %        82.81
```
和前面三版彻底反过来了：**L1/TEX 从 ~90% 掉到 20.7%**（TMA 走专用路径，不再用一堆 LDS/LDG 把 L1/TEX 打满），而 **SM Busy 飙到 82.81%**。ncu 直接点名：
> *Shared is the highest-utilized pipeline (20.7%) ... dominated by its **Tensor (FP)** sub-pipeline. It is well-utilized.*

也就是说，**张量核（Tensor FP 管线）终于成了最高利用的管线**——这是前面 WMMA 三版从没做到的。

## 2、看延迟 + 占用率的颠覆
```
sudo /usr/local/cuda/bin/ncu --section WarpStateStats \
  --metrics smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio \
  -k regex:"wgmma_kernel" -s 1 -c 1 ./build/tensor_core/bench 4 2048 2048 2048
```

```
    Warp Cycles Per Issued Instruction                                          cycle        25.09
    --------------------------------------------------------------------------- ----------- ------------
    smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio                   inst        13.46
    smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio           inst         4.75
    smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio              inst         1.36
    smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio          inst         0.39
    --------------------------------------------------------------------------- ----------- ------------
```
（Achieved Occupancy = 7.58%，见 §1 的 `--set full`。）按 ratio÷cyc/issue（25.09）换算占比，最反直觉的一点：**占用率只有 7.58%**（每线程 154 寄存器 + 96KB smem，每 SM 只挂得下 1~2 个 block），却跑出了 82.8% 的 SM Busy。

这正是 Hopper 路线和 WMMA 的本质差别：WMMA 靠**高占用率(TLP)**——很多 warp 轮流跑来盖住访存延迟；WGMMA 靠**深度异步流水线**——TMA 异步搬、WGMMA 异步算、mbarrier 衔接，一个 warpgroup 自己就把延迟流水起来，不需要很多 warp。头号 stall 是 **CTA barrier 13.46 → 53.6%**（生产者/消费者 mbarrier 交接），即流水线结构的固有同步、且大部分被重叠掉了；其余 long_scoreboard 4.75 → 18.9%，mio_throttle/short_scoreboard 都很小。

## 3、性能与 cuBLAS 对照
4096³ 实测：

```
WGMMA(TMA+WS)      time=1.1519 ms  GFLOPS=119311  util(vs148T)=80.6%
cuBLAS BF16 (fair) time=1.0426 ms  GFLOPS=131820  util(vs148T)=89.1%
```
- 25.6%(cp.async+WMMA) → **80.6%(WGMMA)**，×3.1，是整条手写阶梯最大的一跳。
- 规模越大越高：2048³ 65.6% / 4096³ 80.6% / **8192³ 88.1%**。
- 达到 fair cuBLAS BF16 的约 **90%**（cuBLAS 仍领先；注意对照 cuBLAS 必须 handle 复用，否则会被 0.33ms/次的 create 开销假性拉低，详见 H20复现结论 §5）。

## 总结
换上 Hopper 原生 **WGMMA + TMA + warp specialization** 是质变：张量核第一次成为最高利用管线（SM Busy 82.8%），占用率仅 7.6% 却最快——**用异步流水线代替高占用率隐藏延迟**。手写到此达到 cuBLAS 的九成。

同样这套流水线换成 FP8 只需改极少代码、吞吐翻倍（见 [12 tensor core - FP8 WGMMA](12%20tensor%20core%20-%20FP8%20WGMMA.md)）。再往上（追平/超过 cuBLAS）要的是 persistent kernel + tile scheduler、cluster/DSMEM、自写 swizzle 等。

> 本文 ncu 原始分项输出见 `baselines/tensor_core/doc_raw/tc_04_wgmma_tma_ws.txt`（与 cuda_core 的 doc_raw 对称，可由 `baselines/tensor_core/collect_doc_ncu.sh` 复跑）。
