# Tensor Core 用例 ncu 剖析汇总

- 工具：`ncu --set full`（需 GPU 计数器权限，用 `sudo /usr/local/cuda/bin/ncu`），脚本 `profiling/run_ncu.sh`。
- 剖析尺寸：**2048³**（`-s 1 -c 1`，跳过 verify、剖析 bench 尺寸的一次 launch）。
- 每个用例的完整报告：`profiling/<name>.ncu-rep`（可用 Nsight Compute UI 打开）+ `profiling/<name>.details.txt`（文本全量）。
- 注意：表中是 2048³ 的剖析值；headline GFLOPS 用 4096³（见 docs/H20复现结论，规模越大利用率越高）。

## Speed-of-Light + 占用率（@2048³）

| 用例 | Duration | Compute(SM)% | L1/TEX% | DRAM% | L2% | 占用率 | Reg/thread | SM Busy% |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| tc_01 WMMA_naive | 1.18 ms | 16.5 | **99.8** | 0.4 | 8.9 | 71.2% | 40 | 17.5 |
| tc_02 WMMA_smem | 682 µs | 37.8 | 82.8 | 0.7 | 10.3 | 42.0% | 64 | 40.9 |
| tc_03 WMMA_pipe | 521 µs | 37.2 | 92.2 | 0.9 | 7.1 | 33.4% | 66 | 39.9 |
| tc_04 WGMMA(TMA+WS) | 191 µs | **67.0** | 20.7 | 2.5 | 11.5 | **7.6%** | 154 | **82.8** |
| tc_05 WGMMA_fp8 | 109 µs | 58.9 | 22.9 | 1.9 | 15.9 | 7.4% | 154 | 73.1 |

## 主要 warp stall（每条发射间隔的平均阻塞周期）

| 用例 | Warp cyc/issue | 头号 stall | 解读 |
| --- | ---: | --- | --- |
| tc_01 | **109.2** | long_scoreboard 38.7% + mio_throttle 35.4% | fragment 直接从 global 取，L1/TEX 99.8% 打满、访存指令排队 → 张量核饿死 |
| tc_02 | 16.4 | (smem)scoreboard 34%；ALU 是最高管线 32.5% | smem 复用把 cyc/issue 从 109 砍到 16；但 load_matrix/索引算的 ALU/MIO 仍是天花板 |
| tc_03 | 27.7 | mio_throttle 33.6% | cp.async 让搬运/计算重叠；占用率降到 33% 但靠流水线补偿（4096³ 才明显超过 smem）|
| tc_04 | 25.1 | **CTA barrier 53.5%** | "Shared/Tensor(FP)" 是最高管线；占用率仅 7.6% 却 SM Busy 82.8% —— 异步+warp specialization 用流水线代替 TLP 隐藏延迟，stall 主要是生产者/消费者 mbarrier 交接 |
| tc_05 | 25.2 | CTA barrier 48.7% | 与 tc_04 同构，fp8 半字节、吞吐翻倍 |

## 一句话结论

- **warp 级 WMMA（tc_01→03）**：靠**高占用率(TLP)** 隐藏访存延迟，瓶颈在 L1/TEX + MIO（搬 fragment 的访存指令），张量核吃不饱（SM Busy ≤ 41%）。
- **warpgroup 级 WGMMA + TMA + warp specialization（tc_04/05）**：占用率只有 ~7.6%，却把 SM Busy 拉到 **73–83%**，靠**异步多级流水线**而非 TLP 隐藏延迟，张量核（Tensor FP 管线）成为最高利用管线 —— 这就是 Hopper 原生路线和 WMMA 的本质差别。
