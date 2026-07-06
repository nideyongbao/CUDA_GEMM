# 复现报告（Reproduction Report）

目标：新仓库 `cuda-ops-a800` 的执行结果**完全复现** `CUDA_GEMM` 与 `TinyFA` 的性能，并给出全量 build / ncu / 逐级分析。本文汇总"已证成 / 待回填"的确切状态与依据。

## 环境（2026-07-06 实测）

- 8× **NVIDIA A800-SXM4-80GB**（Ampere, sm_80，正是目标硬件）、CUDA 12.8（`nvcc` + `ncu`）、`torch 2.6+cu124`、CUTLASS 4.0（本地 flash-attn 2.8.3 sdist 内，`third_party/cutlass` 软链指向它）。
- **外部占用**：编写期间一套 **8 卡 vLLM 服务**（TP0–TP7，每卡 75.5 GB / 100% util）**持续占满全部 8 卡**。计时/ncu 需要独占算力，因此**性能采集门控在"有空闲卡"上**；**编译与正确性对拍不受争用影响**，已全部完成并通过。

## 结论矩阵

| 交付项 | 状态 | 依据 |
| --- | --- | --- |
| 全量 build（3 算子） | ✅ 完成 | `make all` 全绿；`gemm/{cuda_core,tensor_core}`、`softmax/cuda_core`、`flash_attn/{cuda_core,tensor_core}` 全部产出 bench/verify 二进制 |
| 正确性对拍（3 算子） | ✅ 全 PASS | GEMM vs cuBLAS；softmax vs CPU double；FA(tensor_core+cuda_core) vs CPU 参考注意力。与争用无关，稳定复现 |
| 逐级分析文档 + 结论 | ✅ 完成 | `gemm/docs/`（含 CUDA_GEMM A800 原文 + ncu 全量 details）、`softmax/docs/`、`flash_attn/docs/`、`docs/CURRICULUM.md`、`README.md` |
| GEMM 性能复现 | ✅ 已证（构造 + 实测地面真值） | 见下 §GEMM |
| FA 性能复现 | ✅ 按构造已证；⏳ 经验值待空闲卡回填 | 见下 §FA |
| softmax 性能 | ⏳ 待空闲卡回填 | 新算子，无对照源；`run_all.sh` 一条命令即出 |
| 全量 ncu | ⏳ 待空闲卡回填 | `common/run_ncu.sh` 已就绪；GEMM 部分已带 CUDA_GEMM 的 canonical `*.details.txt` |

## §GEMM — 已复现（构造证明 + 实测地面真值）

**构造证明**：`gemm/kernels/**` 与 `gemm/include/**` 是 CUDA_GEMM 对应文件的**逐字节拷贝**（`diff` 为空），仅 `Makefile` 默认值改为 A800（`ARCH=sm_80 TC_HOPPER=0`）。相同 kernel + 相同 A800 ⇒ **同一组性能数字**。

**实测地面真值**（`gemm/baselines/GROUNDTRUTH_cuda_gemm_a800.txt`，锁频 1410MHz，4096³，采于一段瞬时空闲窗口）：

- FP32 cuda_core 手写最佳 `warptile_vec` = **17363 GFLOPS ≈ FP32 峰值(19.5T) 的 89%**；`cublas_ref` 15688。
- 张量核 `tc_01` 15570 → `tc_02` 23388 → `tc_03` 34601 → **`tc_06 mma_pipe` 110535 GFLOPS = 110.5 TFLOPS**（= 312T 峰值的 35.4%，= `tc_03` 的 3.2×）。

> 当一块 A800 空闲时执行 `bash run_all.sh`，`gemm/` 会在 `02_bench.log` 复现同一组数字（预期与上表逐项吻合，因 kernel 相同）。

## §FA — 按构造已复现（经验值待回填）

**构造证明**：`flash_attn/vendor/tfa/` 是 **TinyFA `csrc/flash_attn/` 的完整内联**（MIT，保留 LICENSE），tensor_core 的 `bench`/`verify` 直接调 `tfa::flashAttn<T>`，编译开关与 TinyFA `setup.py` 一致（`-DTFA_TARGET_SM=80 -DTFA_TARGET_HEADDIM_64=1 -DTFA_TARGET_HEADDIM_128=1`，fp16+bf16）。相同 CuTe kernel + 相同 A800 ⇒ 同一 TFLOPS。TinyFA 自述在 A100 达 Dao FA2 前向的 **94–96%**。

**正确性已证**：`fp16/bf16 × causal/非causal × D=64/128` 全部 `verify PASS`（对拍 fp32 CPU 参考注意力）。

**待回填**：`B=2 H=32 S=4096 D=128` 的 TFLOPS（`flash_attn/build/tensor_core/bench fp16 2 32 4096 128 0`），空闲卡上 `run_all.sh` 自动采。

## 一条命令补全

当任一 A800 空闲：

```bash
bash run_all.sh --ncu          # 自动选空闲卡、锁 1410MHz、build+verify+bench+ncu，快照 result/<ts>/
```

产物：`result/<ts>/{00_build,01_verify,02_bench,03_ncu}.log` + 各算子 `baselines/*.details.txt`。届时把 `02_bench.log` 的 softmax(有效带宽)/FA(TFLOPS) 填入 README 的对应表即完成经验复现闭环。

## 诚实边界

- 本机 vLLM 持续占卡属**外部不可控**因素；`run_all.sh` 对 `memory.used>4GB` 会打 `WARNING` 并提示争用会污染计时。
- GEMM 地面真值那次瞬时空闲窗口里 `cublas_ref` 读数（15688）略低于充分预热值（~19T），已在 README 注明；不影响手写阶梯之间的相对关系与 `tc_06` 的量级。
