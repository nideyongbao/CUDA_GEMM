# 构建参考仓库 flash_attention_from_scratch (fafs) 于 A800

fafs 是 lubits.ch 的 16 级 raw-CUDA FA2 教程仓库(sm_80),我们用它当**性能参考基线**(不进本仓库构建,只对比)。它在本机 A800 上能干净编译+运行,只需修两个环境路径。

## 步骤

```bash
FAFS=/data/env/workspace/0629/flash_attention_from_scratch
TORCH_LIB=/usr/local/lib/python3.10/dist-packages/torch/lib

# 1) 编译期:libcuda.so 只有 .so.1(驱动),没有 -lcuda 需要的裸 .so 符号链接;
#    用 CUDA 的 stub 目录补上(纯链接用,运行时用真驱动)。
cd "$FAFS"
LIBRARY_PATH=/usr/local/cuda-12.8/lib64/stubs:$LIBRARY_PATH pip install --no-build-isolation .
pip install ./py    # flash_helpers(配置/测试工具)

# 2) 运行期:torch 的 libc10.so 等不在默认 loader 路径。
export LD_LIBRARY_PATH=$TORCH_LIB:$LD_LIBRARY_PATH

# 3) 跑参考基线
CUDA_VISIBLE_DEVICES=<空闲卡> python3 flash_attn/reference/bench_fafs.py
```

## 说明
- 原始 `pip install .` 在本环境报 `legacy-install-failure`——**不是代码问题**,kernel 全部编译通过(128 registers, 0 spill),只是链接期找不到 `-lcuda`。加 stub 路径即解。
- `bench_fafs.py` 需要 `torch`(已装),**不需要官方 flash-attn**(它的 test/utils 会 import `flash_attn_2_cuda`,我们绕开,自己造 `[B,S,H,D]` 张量——与 fafs 及本仓库同布局)。
- 结果见 `fafs_a800_results.md`。
