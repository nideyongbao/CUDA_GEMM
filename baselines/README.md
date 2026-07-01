# baselines/ — 各机型策展的 Nsight Compute 参考基线

这里放**人工策展、文档引用的 "golden" ncu 快照**（与每次 `run_all.sh` 的瞬时 profiling 分开）。
`docs/` 里的逐 kernel 分析引用的就是这里的 `*.details.txt`。

## 现状（是否过时？—— 不过时）

| 目录 | 机型 / CC | 覆盖 | 状态 |
| --- | --- | --- | --- |
| `cuda_core/` | **H20 (Hopper, CC 9.0)** | FP32 阶梯 01–10 + autotune + BF16 对照，各含 `.details.txt`(+`.ncu-rep`) | ✅ 当前 |
| `tensor_core/` | **H20 (Hopper, CC 9.0)** | tc_01–03 WMMA + **tc_04 WGMMA** + **tc_05 FP8**，含 `doc_raw/` 原始 ncu | ✅ 当前，**已含 FP8** |
| `a800/` | **A800-SXM (Ampere, CC 8.0)** | FP32 阶梯 + BF16 + tc_01–03 WMMA + 对账/缩放/verify | ✅ 当前 |
| `throughput_4096.txt` | H20 | FP32 阶梯 @4096³ headline 吞吐 | ✅ 当前 |

**结论**：baselines **不过时、不需要重写**。
- `cuda_core/`+`tensor_core/` 的 `.details.txt` 头部均为 `Device 0, CC 9.0` → 确系 **H20 实测**（早期误用 Turing/1650S 的问题已在本仓库修过，见根 README 记录）。
  - 注：cuBLAS 内部把它的 kernel 命名成 `sm80_xmma_gemm_...` 是**库的 tile 命名习惯**，不代表跑在 sm_80；看头部 `CC 9.0` 为准。
- **完整覆盖当前 kernel 阶梯（含 Hopper 独占的 WGMMA/TMA/FP8）**，且 kernel 源码本轮未改动 → 基线与代码仍对应。
- ncu 采集时 SM 被降频到 ~1.69GHz（ncu 串行化会禁用 boost，属正常）；**占用率/停顿/各引擎 throughput% 等结构性指标有效**，绝对 GFLOPS 以非-ncu 的 bench 为准（`throughput_4096.txt` / `result/`）。

## 什么时候需要动它

- **加同代际新卡**（H100/L40S…）：一般**不必**加 ncu 基线，`result/` 的 bench+telemetry 已足够；除非要做逐 kernel 瓶颈剖析。
- **多代际扩展**（本轮方向）：当给 A10/L4/Blackwell 补 ncu 时，建议把结构改成 **`baselines/<arch>/{cuda_core,tensor_core}/`**（如 `baselines/sm89/…`），让现在隐式="H20" 的 `cuda_core/`+`tensor_core/` 变成显式分代际。目前先不动，避免打断 `docs/` 里的现有引用路径。
- **云端（Modal）卡**：容器内采 ncu 需要特权且本机 ncu 需 sudo（见项目记忆），A10/L4 的 ncu 基线**暂不强求**——它们的 `run_all.sh` bench + 遥测已能支撑跨机对比。

## 复跑

```bash
# 本机重采（需 root；架构自适应）：
sudo bash scripts/run_ncu.sh <输出目录> 2048 <build目录> quick
# 或对单个 kernel：
sudo /usr/local/cuda/bin/ncu --set full ./build/cuda_core/bench 10 4096 4096 4096
```

`.ncu-rep`（Nsight UI 可开）本地生成、`.gitignore` 不入库（A800 的）；`.details.txt` 文本入库便于 diff 与文档引用。
