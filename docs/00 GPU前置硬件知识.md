## 0、本项目的目标硬件：NVIDIA H20 (Hopper)

下面的概念分析是架构无关的，但具体的带宽/容量数字以本项目实测的 H20 为准（`cudaGetDeviceProperties` + `nvidia-smi`）：

| 部件 | H20 实测值 |
| --- | --- |
| 架构 / 计算能力 | Hopper / **sm_90 (CC 9.0)** |
| SM 数量 | **78** |
| 显存 | 96 GB **HBM3**，带宽 **~4023 GB/s** |
| L2 Cache | **60 MB**（全 SM 共享） |
| Shared Memory | **228 KB/SM**，单 block 可 opt-in 至 227 KB |
| 寄存器 | 65536 / SM |
| SM 时钟 | ~1.98 GHz（boost ~2.2），功耗墙 500 W |
| FP32 (CUDA core) 峰值 | **~40–44 TFLOPS** |
| FP16/BF16 Tensor Core 峰值 | **~148 TFLOPS**（TF32 ~74，FP8 ~296） |

注意：本项目所有 kernel 都是 **FP32 + CUDA core**，从不使用 Tensor Core，因此天花板是 ~40 TFLOPS 那一列，而不是 ~148 TFLOPS。这也是它能作为"Hopper GEMM 前置课"、但还不是"Hopper GEMM 教程"的根本原因。

## 1、全局图
SM 是 GPU 里的"计算引擎集群"，是 GPU 并行执行的基本硬件单元。一块 GPU 由几十到上百个 SM 并列组成，每个 SM 内部包含若干 CUDA Core（或 Tensor Core）、寄存器堆、shared memory、warp 调度器等完整的执行资源。**理解 GPU 硬件的核心就是理解一个 SM 的内部结构**，因为 CUDA 的所有性能优化（coalescing、bank conflict、occupancy）本质上都是在分析对 SM 内部资源的使用效率。在一个sm内部，按数据流向分四块

``` text
[显存 DRAM] --(H20: HBM3, ~4 TB/s)--> [L2] --> [L1/TEX + SMEM 共享的SRAM] 
                                                      |
                                          [LSU 加载存储单元]
                                                      |
[寄存器堆 Register File] <--> [FP32 计算单元 (CUDA cores)]
                                                      ^
                                          [Warp Scheduler 发射]
```


## 2、DRAM
DRAM（Dynamic Random Access Memory） 一般指显存的物理内存，离 SM 最远,延迟几百周期,带宽是这块卡的硬上限


## 3、L2 / L1/TEX
L2 是全 SM 共享的缓存;L1/TEX 是每个 SM 私有的。它们缓冲 DRAM 的数据。
L1/TEX 和 shared memory **在物理上是同一块 SRAM**(在 Volta（sm_70）及之后的架构中)。这点极其重要——它解释了为什么你访问 shared memory 会顶高 L1/TEX throughput,也解释了 bank conflict 为什么发生在这块 SRAM 上。

## 4、LSU 和 MIO
**LSU(Load/Store Unit)** 是执行访存指令的硬件单元。**MIO(Memory Input/Output)** 是 LSU 等单元前面的**指令发射队列**。区别是:

- LSU 忙不忙 = 数据搬得多不多。
- MIO 堵不堵 = 访存**指令条数**多不多(指令排队等着进 LSU)。
- **对应指标**:`mio_throttle`(MIO 队列满,指令发不进去)、`inst_executed_pipe_lsu.sum`(LSU 执行的指令绝对数)

### 4.1 MIO

MIO（Memory Input/Output pipeline）是 SM 内部的**访存指令派发队列**，凡是走"慢路径"的访存指令都要经过它：
- `LDS` / `STS`——shared memory 读写
- `LD.GLOBAL` / `ST.GLOBAL`——全局内存（DRAM/L2/L1 cache line fill）
- `LD.LOCAL`——local memory（寄存器溢出到 DRAM）
- 部分特殊指令，如 `ATOM`（原子操作）、`RED`

**L1 cache 命中走不走 MIO**

这里有一个容易混淆的地方：

|访问类型|走 MIO 吗|
|---|---|
|shared memory（`LDS`）|✅ 走 MIO|
|global load，L1 **miss**，需要去 L2/DRAM|✅ 走 MIO|
|global load，L1 **hit**|✅ 仍然走 MIO（指令本身还是要排队发射）|
|寄存器读写|❌ 直接进 ALU pipeline|

所以 L1 hit 并不能绕开 MIO。MIO throttle 的本质是**指令条数太多把队列塞满**，不管这些指令最终在 L1 就命中还是要去 DRAM，指令本身都要占队列位置。

## 5、Warp Scheduler 

每个 SM 有 4 个 warp scheduler。每个周期,每个 scheduler 挑一个"ready"的 warp,发射它的一条指令。


## 6、寄存器堆 + FP32 单元
寄存器是最快的存储,FP32 单元(CUDA cores)做实际乘加。一个 SM 的寄存器堆是固定大小(65536 个)。寄存器是**所有 active warp 平分**的稀缺资源。每个线程用得越多,能同时挂的 warp 越少(occupancy 越低)。


## 总结

| SM 部件          | 它的瓶颈指标                      | 哪一代撞上                              | 怎么治                 |
| -------------- | --------------------------- | ---------------------------------- | ------------------- |
| DRAM 带宽        | DRAM Throughput             | naive(21%,没用上)→ vectorized(57%,逼近) | tiling 提高算术强度       |
| L1/TEX SRAM    | L1/TEX Throughput           | 全程高(含义随上下文变)                       | —                   |
| MIO 发射队列       | mio_throttle, LSU指令数        | smem(27.5,LDS塞爆)                   | 向量化/warptile 减指令条数  |
| Warp Scheduler | Eligible, No-Eligible       | naive(同时等DRAM)、2D(63%没藏住)          | 双缓冲(load/compute重叠) |
| 寄存器堆           | Reg/Thread, Block Limit Reg | 2D(96,occ降62%)                     | 主动权衡,换ILP           |
| FP32 单元        | math_pipe_throttle          | 逐代缓升(接近算力墙是好天花板)                   | 接近此处=优化到头           |
