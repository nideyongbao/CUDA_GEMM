# 文档导航（按机型分目录）

本项目是一份 **GEMM 优化阶梯教程 + 多机型复现**。文档按"**哪台卡**"拆分，方便横向对比：同一套 kernel、同一套驱动，在不同架构上各自的实测结果与瓶颈分析各占一个目录。

```
docs/
  h20/        NVIDIA H20 (Hopper, sm_90)   —— 原始教程 + 全栈复现（CUDA core→WMMA→WGMMA→FP8）
    00–07     CUDA core FP32 阶梯逐 kernel 分析（naive→smem→register tiling→…→double buffer）
    08–12     Tensor Core 阶梯逐用例分析（WMMA→cp.async→WGMMA+TMA+WS→FP8）
    13 H20 GEMM 复现总结.md      ★ H20 收口总结
    H20复现结论.md / 6.1–6.5.md / 总结.md   早期复现详录与阶段交接
    img/      ncu 截图（H20 实测）
  a800/       NVIDIA A800-SXM4-80GB (Ampere, sm_80)
    A800 GEMM 复现总结.md        ★ A800 迁移复现总结（含与 H20/cuBLAS/理论峰值/公开基准四重对账）
  interactive/  两条优化阶梯的交互式 p5.js 可视化（HTML，跨机型通用）
```

> 说明：`h20/00–12` 是**优化方法论 + 逐 kernel/用例讲解**，其 ncu 数据为 H20 实测；原理对所有架构通用，A800 的实测画像见 `a800/` 与 `baselines/a800/`。新增机型时，在 `docs/<机型>/` 下放该机的复现总结，并把 headline 结果补进下表。

---

## 机型对比速览（headline，各自最佳；详见各机型总结）

| 指标（4096³/8192³） | **H20 (sm_90)** | **A800 (sm_80)** | 说明 |
| --- | ---: | ---: | --- |
| FP32 峰值 | ~40 TFLOPS | 19.49 TFLOPS | A800 FP32 核少一半(64 vs 128/SM) |
| FP32 cuBLAS SGEMM | 30.3 TFLOPS | 19.0 TFLOPS（锁频，98%峰） | |
| FP32 手写最佳 | 25.1 TFLOPS | 17.6 TFLOPS（90%峰/93%cuBLAS） | |
| **BF16 张量核峰值** | **148 TFLOPS** | **312 TFLOPS** | **A800 是 H20 的 2.1×** |
| **BF16 cuBLAS** | 132 TFLOPS | **214–294 TFLOPS**（默认→锁频） | **A800 ≈ H20 的 1.6–2.2×** |
| 手写张量核最佳 | **119 TFLOPS**（tc_04 WGMMA, 80.6%） | 43 TFLOPS（tc_03 WMMA, 13.8%） | A800 **无 WGMMA/TMA/FP8** |
| FP8 (WGMMA) | 224 TFLOPS | **不支持**（Hopper 独占） | |

**一句话对比**：H20 = 带宽厚、FP32 高、BF16 张量核薄(148T)但有 WGMMA 吃满；A800 = **BF16 张量核厚(312T)、FP32 薄，但无 WGMMA**——手写只能到 WMMA(13.8%)，要吃满 312T 得靠 cuBLAS/CUTLASS（Ampere 原生 `mma.sync`+`ldmatrix`）。做 BF16 GEMM 选 A800，纯 FP32 选 H20。

> ⚠️ **对比 MFU 必看时钟策略**：GFLOPS 随实际 SM 时钟线性变化，A800 默认 boost 在重张量负载只跑 ~1140–1215MHz（非额定 1410），公开基准/默认时钟 ≈ 78% 峰、锁频 ≈ 94% 峰，效率其实一致。详见 [a800 总结 §5](a800/A800%20GEMM%20复现总结.md)。

---

## 快速索引

- **H20 全栈总结** → [h20/13 H20 GEMM 复现总结.md](h20/13%20H20%20GEMM%20复现总结.md)
- **A800 迁移总结** → [a800/A800 GEMM 复现总结.md](a800/A800%20GEMM%20复现总结.md)
- **一键跑本机全量测试** → 仓库根 `bash run_all.sh`（自动识别架构，结果存 `result/<时间戳>/`），见根 [README](../README.md)
- **交互式可视化** → [interactive/](interactive/)
