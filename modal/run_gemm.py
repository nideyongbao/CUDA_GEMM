"""
Modal 入口：在云端任意代际 GPU 上一键跑本仓库的 CUDA GEMM 全量测试（run_all.sh）。

设计对齐 /shard_data/brooksli/workspace/platform/modal/run_platform.py，但因为本仓库是
纯 CUDA C++（只需 nvcc + cuBLAS + make + nvidia-smi），镜像**刻意做得很轻**：不装 torch/triton。
SM 数在 torch 缺席时由 scripts/gpu_specs.py 的兜底表提供，架构由 run_all.sh 按 CC 自适应。

用法（GPU 类型通过 env GPU_TYPE 在 import 时绑定给 @app.function 装饰器）：
=====
  cd /shard_data/brooksli/workspace/0625/CUDA_MATMUAL/modal

  GPU_TYPE=A10G uv run modal run run_gemm.py     # Ampere sm_86
  GPU_TYPE=L4   uv run modal run run_gemm.py     # Ada sm_89
  GPU_TYPE=H100 uv run modal run run_gemm.py     # Hopper sm_90（全量含 WGMMA/FP8）
  GPU_TYPE=A100 uv run modal run run_gemm.py     # Ampere sm_80

  # 快速版（跳过尺寸缩放/autotune，省时省钱）：
  GPU_TYPE=L4 BENCH_QUICK=1 uv run modal run run_gemm.py

Modal 支持的 GPU 串：T4 / L4 / A10G / A100 / A100-80GB / L40S / H100 / H200 / B200。
拉结果：  modal volume get cuda-matmul-gemm-results <打印出来的 vol_tag>
"""

import os
import pathlib
import modal

# ---------------------------------------------------------------------------
# GPU 配置：env 在 import 时读取（@app.function(gpu=...) 装饰器 import 时求值）
# ---------------------------------------------------------------------------
GPU_TYPE = os.environ.get("GPU_TYPE", "L4")
GPU_SPEC = GPU_TYPE                       # 本仓库是单卡 GEMM，无需多卡/P2P
BENCH_QUICK = os.environ.get("BENCH_QUICK", "0")   # 1=run_all.sh --quick

# 本仓库根目录 = 本文件(modal/run_gemm.py)的上一级
REPO_DIR = pathlib.Path(__file__).resolve().parent.parent

app = modal.App(name=f"cuda-matmul-gemm-{GPU_TYPE.lower()}")

VOLUME_NAME = "cuda-matmul-gemm-results"
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

# ---------------------------------------------------------------------------
# 镜像：Debian + CUDA toolkit（nvcc + cuBLAS + 头文件），装 make。不装 torch。
#   分层缓存：CUDA toolkit 层最重(~3GB)，只在首次或改动时构建一次。
# ---------------------------------------------------------------------------
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("wget", "gnupg", "make", "ca-certificates")
    .run_commands(
        "wget -q https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb",
        "dpkg -i cuda-keyring_1.1-1_all.deb && rm cuda-keyring_1.1-1_all.deb",
        "apt-get update -qq",
    )
    # nvcc + 运行时；cuBLAS dev(含 -lcublas 需要的 .so 与头文件)
    .apt_install("cuda-toolkit-12-6", "libcublas-dev-12-6")
    .env({"PATH": "/usr/local/cuda/bin:/usr/local/nvidia/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
          "LD_LIBRARY_PATH": "/usr/local/cuda/lib64:/usr/local/nvidia/lib64"})
    # 只挂代码/构建/工具，不挂 result/docs/baselines（瞬时物在容器里现生成）
    .add_local_dir(str(REPO_DIR / "kernels"),  remote_path="/workspace/repo/kernels")
    .add_local_dir(str(REPO_DIR / "include"),  remote_path="/workspace/repo/include")
    .add_local_dir(str(REPO_DIR / "scripts"),  remote_path="/workspace/repo/scripts")
    .add_local_file(str(REPO_DIR / "Makefile"),   remote_path="/workspace/repo/Makefile")
    .add_local_file(str(REPO_DIR / "run_all.sh"), remote_path="/workspace/repo/run_all.sh")
)


@app.function(gpu=GPU_SPEC, image=image, timeout=2400,
              volumes={"/workspace/results": volume})
def run_gemm_suite():
    """在目标 GPU 上跑 run_all.sh（默认时钟——云容器一般无法锁频），落盘到 Volume。"""
    import subprocess, os, shutil, time
    from datetime import datetime

    _print_env()
    repo = "/workspace/repo"

    # run_all.sh 会自建 result/<ts>/；用 RUN_DIR_OVERRIDE 固定一个我们能拿到的路径
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = f"{repo}/result/modal_{timestamp}"
    os.makedirs(run_dir, exist_ok=True)

    flags = "--quick" if BENCH_QUICK == "1" else ""
    # 云容器多为 root 但驱动通常禁锁频；不加 --lock，run_all.sh 记录遥测真实时钟即可。
    cmd = f"cd {repo} && RUN_DIR_OVERRIDE='{run_dir}' bash run_all.sh {flags}"
    print(f">>> {cmd}\n")

    t0 = time.time()
    # 不 capture：让 run_all.sh 的进度直接流到 Modal 日志，方便实时看编译/跑测
    rc = subprocess.run(cmd, shell=True).returncode
    print(f"\n<<< run_all.sh exit={rc} ({int(time.time()-t0)}s)")

    # 落盘 Volume：<gpu>_<ts>_modal
    gpu_tag = _gpu_tag()
    vol_tag = f"{gpu_tag}_{timestamp}_modal"
    vol_dest = f"/workspace/results/{vol_tag}"
    # run_all.sh 里 result/latest 是软链，copytree 会失败 → 用 ignore 跳过软链
    shutil.copytree(run_dir, vol_dest, ignore=shutil.ignore_patterns("latest"))
    volume.commit()

    # 回显汇总，便于在 Modal 控制台直接看结论
    summ = os.path.join(run_dir, "00_summary.txt")
    if os.path.exists(summ):
        print("\n" + "=" * 60 + "\n" + open(summ).read())
    print(f"\n下载：modal volume get {VOLUME_NAME} {vol_tag}")
    return {"vol_tag": vol_tag, "rc": rc}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _gpu_tag():
    """从 nvidia-smi 名字提取短标签用于结果目录命名（无 torch 依赖）。"""
    try:
        import subprocess
        name = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                              capture_output=True, text=True).stdout.strip().splitlines()[0]
    except Exception:
        name = os.environ.get("GPU_TYPE", "gpu")
    for tag in ["B200", "H200", "H100", "H20", "A100", "A800", "A10G", "A10",
                "L40S", "L4", "T4"]:
        if tag.lower() in name.lower():
            return tag
    return name.replace(" ", "_")[:20]


def _print_env():
    import subprocess
    print("=" * 60)
    for q in ["name", "compute_cap", "clocks.max.sm", "memory.total", "power.limit"]:
        r = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader"],
                           capture_output=True, text=True)
        print(f"  {q:16s}: {r.stdout.strip()}")
    r = subprocess.run(["nvcc", "--version"], capture_output=True, text=True)
    for line in r.stdout.splitlines():
        if "release" in line.lower():
            print(f"  nvcc            : {line.strip()}")
    print("=" * 60 + "\n")


@app.local_entrypoint()
def main():
    """
    通过 Modal 跑 GEMM 全量测试。GPU 由 env GPU_TYPE 决定（import 时绑定）：
        GPU_TYPE=A10G uv run modal run run_gemm.py
        GPU_TYPE=L4   uv run modal run run_gemm.py
    """
    print("=== CUDA_MATMUAL GEMM via Modal ===")
    print(f"GPU (from env GPU_TYPE): {GPU_SPEC}   quick={BENCH_QUICK}")
    out = run_gemm_suite.remote()
    print(f"\nDONE: {out}")
