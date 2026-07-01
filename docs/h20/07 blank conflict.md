对应用例：`kernels/cuda_core/09_bankconflict.cu`（`bank_conflict_kernel`，bench id 11）与收官的 `kernels/cuda_core/10_doublebuffer.cu`（`double_buffer_kernel`，id 12）。下文 ncu 均为 **H20（CC 9.0）实测，4096³，`--launch-count 1`**。

## 0、为什么是 +4 padding
[06 warp tile](06%20warp%20tile.md) 收尾时唯一没解决的就是写 As 的 shared bank conflict（op_st 量级 2×10⁷）。这里专门治它，主要是 st 部分。

没加 padding 之前，写 As 出现了 **8-way bank conflict**；给 As 的每行加 4 个 float 把它错开。为什么是 +4 而不是教科书惯用的 +1：bank 编号是 `(行号 × (BM+p) + 列号) % 32`，因为 `BM % 32 = 0`，化简成 `(列号 × p + 行号) % 32`。要让同一拍 32 个 lane（按"快变维 8 组 × 慢变维宽度 4"排布）落到 32 个不同 bank，需要 `(BM+p) % 32 = 4` → **p=4** 才完全错开；p=1 只会留下斜条纹、削一半。

![](img/Pasted%20image%2020260606154619.png)
![](img/Pasted%20image%2020260606154656.png)

## 1、真实数据：padding 前后
```
sudo /usr/local/cuda/bin/ncu --metrics \
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum,\
l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio \
-k regex:"bank" --launch-count 1 ./build/cuda_core/bench 11 4096 4096 4096
```

| 指标 | 06 warp_tile_vec(id 10，padding 前) | 09 bank_conflict(id 11，+4 padding 后) |
| --- | ---: | ---: |
| shared op_st bank conflict | 21,938,185 | **5,632,486** |
| shared op_ld bank conflict | 151,451 | 174,038 |
| global ld sec/req | 16 | **8.16**（合并改善） |
| L1/TEX Throughput | 67.48% | 60.58% |
| Compute(SM) | 69.54% | 71.54% |
| Duration | 6.18 ms | 6.89 ms |
| GFLOPS@4096³ | **23971**（79.2% cuBLAS） | **21519**（71.0% cuBLAS） |

padding + 搬运映射改造把写 As 的 bank conflict 砍了约 4×（21.9M→5.6M），顺带把 global 读合并从 16 sec/req 拉到 8.16，L1/TEX 也降了几个点。

## 2、关键认知：ncu 显眼 ≠ binding bottleneck
反直觉的地方：bank conflict 明明降了 4×、合并也改善了，**throughput 却没涨反而略降**（23971 → 21519 GFLOPS，Duration 6.18→6.89ms）。

这说明在 H20 这个尺寸上，**bank conflict 是"ncu 上显眼、但不在关键路径上"的瓶颈**——它占满了 L1 流水线的某个计数器，却没卡在决定总时间的依赖链上。看 warp stall 就清楚：bank_conflict 版的 `short_scoreboard`（等 shared 的那项）只有 0.93、`cyc/issue` 5.18，shared 等待早就垫底了；真正还在拖的是零散的 long_scoreboard / 算术依赖，bank conflict 消不消都不在这条链上。

> **判断一个瓶颈值不值得打，看的是它消除后 Duration/Elapsed Cycles 降不降，而不是它在 ncu 上的 throughput% 或绝对计数有多吓人。** 这一步工程上是对的（代码更干净、global 合并更好、换到真撞墙的卡上有用），但在 H20 这张"带宽厚、片上余量大"的卡上不加速。负结果，价值高。

## 3、收官：double buffer
```
sudo /usr/local/cuda/bin/ncu --set basic -k regex:"double" --launch-count 1 ./build/cuda_core/bench 12 4096 4096 4096
```

```
    Duration                         ms          5.93
    L1/TEX Cache Throughput           %         68.89
    Compute (SM) Throughput           %         75.56
    Memory Throughput                 %         66.59
    Registers Per Thread  register/thread        127
    Achieved Occupancy                %         24.23
    Warp Cycles Per Issued Instruction cycle      4.96
```
double buffer 用双缓冲（`As[2]`/`Bs[2]`）让"搬下一块 K-tile"和"算当前块"在不同硬件上重叠，盖住 global→shared 的加载延迟。它把 shared op_st bank conflict 也顺手降到 665,496，是整条手写阶梯的最高点：**5.93 ms / 25132 GFLOPS = cuBLAS SGEMM 的 83.0%、FP32 峰值的 57.1%**，`Compute(SM)` 75.56% 为手写 CUDA core 最高、`cyc/issue` 4.96 最低。

注意它的占用率只有 **24%**（127 寄存器）——又一次印证整条线的主旋律：**靠单线程 ILP（独立的 TM×TN 乘加 + 双缓冲预取）而不是高占用率隐藏延迟**。

## 总结
- **+4 padding 的量是推出来的**：`(BM+p)%32 = 慢变维宽度`，不是无脑 +1。先给 conflict 归类（连续 stride 撞 bank / 离散值组错位），再选 padding 量。
- **bank conflict 在 H20 上是非 binding 瓶颈**：消除它（21.9M→5.6M）对 throughput 没有正收益（略降），因为它不在关键依赖链上。区分"指标高"和"该优化"是关键判断力。
- **真正的收官增益来自 double buffer**：load/compute 重叠把手写 CUDA core 推到 **25132 GFLOPS（cuBLAS 的 83%）**，占用率仅 24% 却最快。
- CUDA core 这条 FP32 阶梯到此见顶（~57% FP32 峰值）。要再上一个数量级必须换引擎——Tensor Core，见 [08 tensor core - WMMA naive](08%20tensor%20core%20-%20WMMA%20naive.md)。完整阶梯对照见 [H20复现结论](H20%E5%A4%8D%E7%8E%B0%E7%BB%93%E8%AE%BA.md)。
