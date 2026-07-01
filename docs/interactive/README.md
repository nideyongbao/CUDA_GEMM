# 交互式可视化：H20 GEMM 优化阶梯（两篇）

用 **algorithmic-art skill** 的 p5.js 交互框架，把本仓库的两条优化主线做成了**可交互、可动画、新手友好**的网页：CUDA Core（FP32）与 Tensor Core（BF16/FP8）。两篇同一套设计语言，对称呈现。

## 文件

| 文件 | 说明 |
| --- | --- |
| `cuda_core_optimization_ladder.html` | **第 1 篇**：`kernels/cuda_core/` 的 FP32 阶梯（naive → double buffer），单文件自包含。 |
| `tensor_core_optimization_ladder.html` | **第 2 篇**：`kernels/tensor_core/` 的张量核阶梯（WMMA → WGMMA+TMA+WS → FP8）。 |
| `PHILOSOPHY-latency-cartography.md` | 第 1 篇的生成式美学宣言：「延迟制图学」。 |
| `PHILOSOPHY-asynchronous-cathedral.md` | 第 2 篇的生成式美学宣言：「异步大教堂」。 |

两篇 HTML 顶部/底部互相链接，可直接跳转。

## 打开方式

直接用浏览器打开任一 `*.html` 即可（需联网以加载 p5.js CDN）。例如：

```bash
xdg-open docs/interactive/cuda_core_optimization_ladder.html        # Linux
xdg-open docs/interactive/tensor_core_optimization_ladder.html
# 或把文件拖进浏览器；也可起个本地服务：
python3 -m http.server -d docs/interactive 8000   # 然后访问 http://localhost:8000/
```

## 它展示了什么

两篇都用同一种交互：左侧「优化阶梯」点击切换场景，主画布是一段**动画**（播放/暂停/单步/调速），每个 kernel 场景配 4 张卡片 **🎯 直观目标 / 🔧 优化点 / 💡 为什么有效 / 📈 最终效果**，并附 H20 实测的「优化前 → 优化后」对比；每个必要名词都在用到之前先做背景介绍；还有算法伪代码、数据流、启动配置（逻辑调用）。

**第 1 篇 · CUDA Core（10 个场景）**：概念入门 → 8 级 kernel → 总结
- naive → shared memory → 寄存器分块(1D/2D) → float4 向量化 → warp 分块 → bank conflict → double buffering，对应 `kernels/cuda_core/01…10`。
- 动画主轴：数据包在「全局内存 → 共享内存 → 寄存器 → 计算单元」之间流动。
- 名词：warp、shared memory、occupancy、ILP、bank conflict…

**第 2 篇 · Tensor Core（7 个场景）**：换引擎入门 → 5 级 kernel → 总结
- WMMA naive → WMMA+smem → cp.async 流水线 → **WGMMA+TMA+warp specialization** → FP8，对应 `kernels/tensor_core/tc_01…tc_05`。
- 动画主轴：张量核引擎被「喂饱」的过程——从被饿死(SM Busy 17%)，到生产者/消费者异步流水线把它喂到 83% 忙。
- 名词：Tensor Core、WMMA、fragment、cp.async、WGMMA、warpgroup、TMA、mbarrier、warp specialization、FP8…

## 数据来源

所有 GFLOPS、利用率、占用率、SM Busy、cyc/issue、bank conflict 等数字均为 **NVIDIA H20（Hopper, CC 9.0）实测**，取自：

- `baselines/throughput_4096.txt`（headline 吞吐 @4096³）
- `../h20/ncu-cuda_core.md`、`../h20/ncu-tensor_core.md`（逐核 ncu 指标）
- `docs/02–13`（逐步分析与总结）

> **两条主线，一个母题——藏延迟。**
> 第 1 篇（CUDA core）：**ILP 取代 TLP**——占用率 99%→24%，吞吐 3532→25132 GFLOPS（cuBLAS 的 83%，FP32 峰值 57%）。
> 第 2 篇（Tensor core）：**异步流水线取代 TLP**——占用率 71%→7.6%，SM Busy 17%→83%，利用率 11%→80.6%(BF16)/75.8%(FP8)，逼到 cuBLAS ~90%。
> 合起来就是 H20 全栈 GEMM：把数据搬近、让等待有事做、让引擎永不挨饿。
