# 文档导航（按机型分目录）

本项目是一份 **GEMM 优化阶梯教程 + 多机型复现**。文档按"**哪台卡**"拆分，方便横向对比：同一套 kernel、同一套驱动，在不同架构上各自的实测结果与瓶颈分析各占一个目录。

```
docs/
  跨代际适配设计.md   ★ 如何统一支持 Ampere/Ada/Hopper/Blackwell（arch 矩阵 + gpu_specs + 精度门控）
  h20/        NVIDIA H20 (Hopper, sm_90)   —— 原始教程 + 全栈复现（CUDA core→WMMA→WGMMA→FP8）
    00–07     CUDA core FP32 阶梯逐 kernel 分析（naive→smem→register tiling→…→double buffer）
    08–12     Tensor Core 阶梯逐用例分析（WMMA→cp.async→WGMMA+TMA+WS→FP8）
    13 H20 GEMM 复现总结.md      ★ H20 收口总结
    FP8-利用率口径修正与H20结果分析.md   ★ FP8 为何不能对 BF16 峰值算利用率（口径修正）
    H20复现结论.md / 6.1–6.5.md / 总结.md   早期复现详录与阶段交接
    img/      ncu 截图（H20 实测）
  a800/       NVIDIA A800-SXM4-80GB (Ampere, sm_80)
    A800 GEMM 复现总结.md        ★ A800 迁移复现总结（含与 H20/cuBLAS/理论峰值/公开基准四重对账）
  interactive/  两条优化阶梯的交互式 p5.js 可视化（HTML，跨机型通用）
```

> 说明：`h20/00–12` 是**优化方法论 + 逐 kernel/用例讲解**，其 ncu 数据为 H20 实测；原理对所有架构通用，A800 的实测画像见 `a800/` 与 `baselines/a800/`。
> **各卡的峰值/精度支持统一由 [`scripts/gpu_specs.py`](../scripts/gpu_specs.py) 提供**（BF16 手工维护、FP8=2×、FP4=4× 按 CC 推导）；
> 新增机型时在 `docs/<机型>/` 放复现总结、把 headline 补进下表，若加的是**已有代际**通常只需在 `gpu_specs.py` 加一行。
> A10G/L4 等云端结果用 [`modal/run_gemm.py`](../modal/run_gemm.py) 跑，见 [modal/README](../modal/README.md)。

---

## 机型对比速览（headline，各自最佳；详见各机型总结）

| 指标（4096³/8192³） | **H20 (sm_90)** | **A800 (sm_80)** | 说明 |
| --- | ---: | ---: | --- |
| FP32 峰值 | ~40 TFLOPS | 19.49 TFLOPS | A800 FP32 核少一半(64 vs 128/SM) |
| FP32 cuBLAS SGEMM | 30.3 TFLOPS | 19.0 TFLOPS（锁频，98%峰） | |
| FP32 手写最佳 | 25.1 TFLOPS | 17.6 TFLOPS（90%峰/93%cuBLAS） | |
| **BF16 张量核峰值** | **148 TFLOPS** | **312 TFLOPS** | **A800 是 H20 的 2.1×** |
| **BF16 cuBLAS** | 134 TFLOPS（锁频，91%峰） | **264.7 TFLOPS**（锁频，85%峰） | **A800 ≈ H20 的 2.0×** |
| 手写张量核最佳(BF16) | **120.6 TFLOPS**（tc_04 WGMMA, 82%峰） | **150.3 TFLOPS**（tc_06 mma.sync, 48%峰）| A800 补齐 mma.sync 后**绝对值反超** H20 手写(峰厚 2.1×) |
| ↳ 补 mma.sync 前 | — | 43.1 TFLOPS（tc_03 WMMA, 14%峰） | tc_06 = 3.49× tc_03 |
| **FP8 张量核峰值** | **296 TFLOPS**（=2×BF16） | **不支持**（Ampere 无 FP8） | |
| 手写 FP8 (WGMMA) | **226.1 TFLOPS**（tc_05, **76% 的 FP8 峰**） | — | 对 FP8 峰算利用率(非 BF16 峰，否则=153% 假象) |

**一句话对比**：H20 = 带宽厚、FP32 高、BF16 张量核薄(148T)但有 WGMMA 吃满；A800 = **BF16 张量核厚(312T)、FP32 薄，无 WGMMA 但有 mma.sync**——手写从 WMMA(14%) 补到 **Ampere 原生 `mma.sync`+`ldmatrix`+cp.async 的 tc_06(48%,150.3T)**，绝对值已超 H20 手写最佳(120.6T)；再往 85% 才需 cuBLAS/CUTLASS。做 BF16 GEMM 选 A800，纯 FP32 选 H20。

> ⚠️ **对比 MFU 必看时钟策略**：GFLOPS 随实际 SM 时钟线性变化，A800 默认 boost 在重张量负载只跑 ~1140–1215MHz（非额定 1410），公开基准/默认时钟 ≈ 78% 峰、锁频 ≈ 94% 峰，效率其实一致。详见 [a800 总结 §5](a800/A800%20GEMM%20复现总结.md)。

### 入门/单槽卡（Modal 云端实测，默认时钟）

同一套 kernel 在 Ampere-图形 / Ada 上跑通（[modal/run_gemm.py](../modal/run_gemm.py)），验证「跨代际能跑、口径对、精度门控正确」：

| 指标（4096³，默认时钟） | **A10G (sm_86)** | **L4 (sm_89)** | 说明 |
| --- | ---: | ---: | --- |
| SM 数 / FP32 峰值 | 72 / 31.2T | 58 / 30.3T | 均 128 FP32/SM |
| BF16 张量核峰值 | 125 TFLOPS | 121 TFLOPS | FP8 峰 250 / **242**（=2×，见↓） |
| BF16 cuBLAS | 92.7T（74%峰） | 83.5T（69%峰） | 小功耗卡默认时钟**降频**：大尺寸更低 |
| 手写张量核最佳(BF16) | 25.1T（tc_03 WMMA, 20%峰） | 25.4T（tc_03 WMMA, 21%峰） | 与 A800 同属「无 WGMMA」，止步 WMMA |
| FP8 | **不支持**（Ampere 无 FP8 硬件） | 有硬件但**未跑**（仓库 FP8 走 Hopper WGMMA） | Ada 测 FP8 需 `mma.sync`（未写） |

> A10(150W)/L4(72W) 是功耗受限卡：默认时钟下重负载撞功耗墙，L4 BF16 从 2048³ 的 80.5T 掉到 8192³ 的 66.7T。
> 云容器无法锁频，故这两列的 %峰值偏低有一半是降频而非效率差。完整报告见 `result/{A10,L4}_*_modal/`。

---

## 快速索引

- **H20 全栈总结** → [h20/13 H20 GEMM 复现总结.md](h20/13%20H20%20GEMM%20复现总结.md)
- **A800 迁移总结** → [a800/A800 GEMM 复现总结.md](a800/A800%20GEMM%20复现总结.md)
- **跨代际适配设计**（Ampere/Ada/Hopper/Blackwell 怎么统一） → [跨代际适配设计.md](跨代际适配设计.md)
- **FP8 利用率口径修正**（为何 226T 是 76% 而非 153%） → [h20/FP8-利用率口径修正与H20结果分析.md](h20/FP8-利用率口径修正与H20结果分析.md)
- **跨机型结果总表**（A800/H20/A10G/L4） → [result/README.md](../result/README.md)
- **一键跑本机全量测试** → 仓库根 `bash run_all.sh`（自动识别架构，结果存 `result/<时间戳>/`），见根 [README](../README.md)
- **云端在你没有的卡上跑** → `GPU_TYPE=L4 uv run modal run modal/run_gemm.py`，见 [modal/README](../modal/README.md)
- **交互式可视化** → [interactive/](interactive/)
