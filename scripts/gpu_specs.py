#!/usr/bin/env python3
# ============================================================================
# gpu_specs.py — 全仓库唯一的「设备算力事实表」(single source of truth)
#
# 谁在用：
#   - scripts/gemm_summary.py  : 生成 00_summary.md 时算 %峰值 / MFU
#   - run_all.sh (间接)         : 架构自适应说明的依据
#   - modal/run_gemm.py         : 跨机型汇总时统一峰值口径
#
# 设计原则（对齐 /shard_data/brooksli/workspace/platform/benchmarks/gpu_specs.py）：
#   1. BF16 张量核 dense 峰值是**唯一手工维护的一张表**；FP8=2×BF16、FP4=4×BF16 都是
#      按代际能力**推导**出来的，不再逐卡手填，避免漏填/写错。
#   2. 精度支持看**计算能力(CC)**而非卡名：
#        - BF16 张量核：Ampere(8.x)/Ada(8.9)/Hopper(9.x)/Blackwell(10+,12) 都有；Turing(7.5) 无(用 FP16)。
#        - FP8  张量核：Ada(8.9) 起才有；Ampere(8.0/8.6) **没有**。
#        - FP4  张量核：Blackwell(10+) 才有。
#        - WGMMA/TMA   ：Hopper(9.0) 独占指令(本仓库手写阶梯 tc_04/tc_05 依赖它)。
#   3. 卡名匹配按**长度降序**，保证 "H20" 不会命中 "H200"、"A10" 不命中 "A100"/"A10G"、
#      "L4" 不命中 "L40S"。
#
# 只依赖标准库，可直接 `import gpu_specs`，也可 CLI：`python3 gpu_specs.py "NVIDIA H20" 9.0`
# ============================================================================

# ---- BF16 张量核 dense 峰值 (TFLOPS, FP32 累加, 无稀疏)。手工维护的唯一一张表。----
PEAK_TFLOPS_BF16 = {
    # Blackwell (sm_100 / sm_120)
    "B200": 2250.0, "GB200": 2450.0, "B100": 1750.0, "5090": 419.0,
    # Hopper (sm_90 / sm_90a)
    "H200": 989.0, "H100": 989.0, "H800": 989.0, "GH200": 989.0,
    "H20": 148.0,                     # 大幅裁剪版 Hopper：算力低、显存/带宽高
    # Ada Lovelace (sm_89)  —— 有 FP8, 无 WGMMA/TMA
    "L40S": 362.0, "L40": 181.0, "L4": 121.0, "4090": 165.0,
    # Ampere (sm_80 数据中心 / sm_86 图形)  —— 无 FP8
    "A100": 312.0, "A800": 312.0, "A30": 165.0, "A40": 149.0,
    "A10G": 125.0, "A10": 125.0,
    # Turing (sm_75)  —— 无原生 BF16，此值为 FP16
    "T4": 65.0,
    # Volta (sm_70)   —— 无 BF16 张量核
    "V100": 0.0,
}

# ---- 多处理器(SM)数量。torch/CUDA 运行时不可用时(如 torch 与驱动版本不匹配)的兜底。----
#      运行时优先用 torch.cuda / cudaGetDeviceProperties 实测；这里是最后一道保险。
SM_COUNT = {
    "B200": 148, "GB200": 148, "B100": 132, "5090": 170,
    "H200": 132, "H100": 132, "H800": 132, "GH200": 132,
    "H20": 78,                        # 实测确认(ncu: # SMs 78)
    "L40S": 142, "L40": 142, "L4": 58, "4090": 128,
    "A100": 108, "A800": 108, "A30": 56, "A40": 84,
    "A10G": 80, "A10": 72,
    "T4": 40, "V100": 80,
}

# ---- HBM/显存 理论带宽 GB/s（memory-bound 场景/带宽墙分析用）----
PEAK_HBM_BW_GBS = {
    "B200": 8000.0, "H200": 4800.0, "H20": 4000.0, "H100": 3350.0, "H800": 3350.0,
    "A100": 2039.0, "A800": 2039.0, "A10G": 600.0, "A10": 600.0,
    "L40S": 864.0, "L40": 864.0, "L4": 300.0, "4090": 1008.0, "T4": 320.0,
}


def _match(table, name):
    """按长度降序匹配卡名子串，避免 H20⊂H200 / A10⊂A100 / L4⊂L40S 之类误命中。"""
    if not name:
        return None
    low = name.lower()
    for k in sorted(table, key=len, reverse=True):
        if k.lower() in low:
            return table[k]
    return None


def _cc(cc):
    """'9.0' / '8.6' / (8,9) → (major, minor)。解析失败返回 (0,0)。"""
    if isinstance(cc, (tuple, list)):
        return int(cc[0]), int(cc[1])
    try:
        s = str(cc).strip()
        major, minor = s.split(".") if "." in s else (s, "0")
        return int(major), int(minor)
    except Exception:
        return 0, 0


# ---- 代际能力：只认计算能力(CC)，不认卡名 ----
def supports_bf16_tc(cc):
    """BF16 张量核：Ampere(8.0)起。Turing(7.5)/Volta(7.0) 无原生 BF16。"""
    major, minor = _cc(cc)
    return (major, minor) >= (8, 0)

def supports_fp8(cc):
    """FP8 张量核：Ada(8.9) 与 Hopper(9.0)/Blackwell(10+) 才有；Ampere(8.0/8.6) 没有。"""
    major, minor = _cc(cc)
    return (major, minor) == (8, 9) or major >= 9

def supports_fp4(cc):
    """FP4 张量核：Blackwell(sm_100/sm_120, major>=10) 独占。"""
    return _cc(cc)[0] >= 10

def supports_wgmma_tma(cc):
    """WGMMA + TMA(本仓库手写 tc_04/tc_05 阶梯依赖)：Hopper sm_90 独占指令。"""
    return _cc(cc)[0] == 9


# ---- 峰值查询 ----
def bf16_peak_tflops(name):
    """BF16 dense 峰值 TFLOPS；未知卡或无 BF16(V100)→None。"""
    v = _match(PEAK_TFLOPS_BF16, name)
    return v if (v and v > 0) else None

def fp8_peak_tflops(name, cc):
    """FP8 峰值 = 2×BF16（当代际支持 FP8 时）；否则 None。"""
    b = bf16_peak_tflops(name)
    return b * 2 if (b and supports_fp8(cc)) else None

def fp4_peak_tflops(name, cc):
    """FP4 峰值 = 4×BF16（Blackwell）；否则 None。"""
    b = bf16_peak_tflops(name)
    return b * 4 if (b and supports_fp4(cc)) else None

def sm_count(name):
    """卡名 → SM 数（兜底表）；未知→None。"""
    return _match(SM_COUNT, name)

def hbm_bw_gbs(name):
    return _match(PEAK_HBM_BW_GBS, name)


def fp32_cores_per_sm(cc):
    """每 SM 的 FP32 lane 数：GA100(8.0)=64；其余(GA10x/Ada/Hopper/Blackwell)=128；Volta/Turing(7.x)=64。"""
    major, minor = _cc(cc)
    if (major, minor) == (8, 0):   # GA100: A100/A800，计算型 die，每 SM 仅 64 FP32
        return 64
    if major == 7:                  # Volta/Turing
        return 64
    return 128                       # GA10x(8.6)/Ada(8.9)/Hopper(9.0)/Blackwell(10+)

def fp32_peak_tflops(name, cc, max_clock_mhz, sm=None):
    """FP32 CUDA-core 峰值 = SM × cores/SM × 2(FMA) × clock。sm 缺省用兜底表。"""
    sm = sm or sm_count(name)
    if not (sm and max_clock_mhz):
        return None
    return sm * fp32_cores_per_sm(cc) * 2 * float(max_clock_mhz) / 1e6  # → TFLOPS


def peak_for_label(label, name, cc):
    """按 tensor-core 用例标签选对应峰值基准 → (peak_tflops, precision_tag)。
    标签含 'fp8'→FP8峰值；含 'fp4'→FP4峰值；否则按 BF16（WMMA/WGMMA 都是 bf16 输入 f32 累加）。"""
    l = (label or "").lower()
    if "fp8" in l:
        return fp8_peak_tflops(name, cc), "FP8"
    if "fp4" in l:
        return fp4_peak_tflops(name, cc), "FP4"
    return bf16_peak_tflops(name), "BF16"


if __name__ == "__main__":
    import sys
    nm = sys.argv[1] if len(sys.argv) > 1 else "NVIDIA H20"
    cc = sys.argv[2] if len(sys.argv) > 2 else "9.0"
    clk = float(sys.argv[3]) if len(sys.argv) > 3 else 1980.0
    print(f"GPU={nm}  CC={cc}")
    print(f"  SM count            : {sm_count(nm)}")
    print(f"  FP32 cores/SM       : {fp32_cores_per_sm(cc)}")
    print(f"  FP32 peak @ {clk:.0f}MHz : "
          f"{fp32_peak_tflops(nm, cc, clk)} TFLOPS")
    print(f"  BF16 TC peak        : {bf16_peak_tflops(nm)} TFLOPS")
    print(f"  FP8  TC peak        : {fp8_peak_tflops(nm, cc)} TFLOPS  (supported={supports_fp8(cc)})")
    print(f"  FP4  TC peak        : {fp4_peak_tflops(nm, cc)} TFLOPS  (supported={supports_fp4(cc)})")
    print(f"  HBM bandwidth       : {hbm_bw_gbs(nm)} GB/s")
    print(f"  WGMMA/TMA (Hopper)  : {supports_wgmma_tma(cc)}")
