#!/usr/bin/env python3
# ============================================================================
# 解析 run_all.sh 的日志，结合设备算力，生成机型 GEMM 汇总报告（00_summary.md/.txt）。
# 用法: gemm_summary.py <RUN_DIR> <GPU_NAME> <CC> <CLOCK_POLICY> <MAXCLK_MHz>
# 口径：GFLOPS 原始值 + 对 FP32 理论峰值 + 对 cuBLAS(库基线) + 对「对应精度」张量核峰值(MFU)。
#   张量核每个用例按其精度选峰值：WMMA/WGMMA(bf16)→BF16 峰值；FP8→FP8 峰值(=2×BF16)；
#   FP4→FP4 峰值(=4×BF16)。峰值/SM 数等设备事实统一来自 scripts/gpu_specs.py。
# ============================================================================
import sys, os, re, glob

RUN_DIR = sys.argv[1]
GPU_NAME = sys.argv[2] if len(sys.argv) > 2 else "unknown"
CC = sys.argv[3] if len(sys.argv) > 3 else "?"
CLOCK_POLICY = sys.argv[4] if len(sys.argv) > 4 else "default"
MAXCLK = float(sys.argv[5]) if len(sys.argv) > 5 and sys.argv[5] not in ("", "0") else None

# ---- 设备算力：唯一事实表在 scripts/gpu_specs.py（BF16 峰值/SM 数；FP8=2×、FP4=4× 按 CC 推导）----
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gpu_specs as G

def sm_count():
    """优先 torch 实测 SM 数；torch 与驱动版本不匹配时(本机常见)退回 gpu_specs 兜底表。"""
    try:
        import torch
        return torch.cuda.get_device_properties(0).multi_processor_count
    except Exception:
        return G.sm_count(GPU_NAME)

SM = sm_count()
CORES = G.fp32_cores_per_sm(CC)
FP32_PEAK = G.fp32_peak_tflops(GPU_NAME, CC, MAXCLK, SM)   # TFLOPS @ 额定 max clock
BF16_PK = G.bf16_peak_tflops(GPU_NAME)
FP8_PK  = G.fp8_peak_tflops(GPU_NAME, CC)                  # None if 该代际无 FP8(Ampere)
FP4_PK  = G.fp4_peak_tflops(GPU_NAME, CC)                  # None if 非 Blackwell

# ---- 解析日志 ----
def read(name):
    p = os.path.join(RUN_DIR, name)
    return open(p).read() if os.path.exists(p) else ""

def parse_bench(text):
    """返回 [(label, gflops), ...]，兼容三种驱动输出行。"""
    out = []
    for line in text.splitlines():
        m = re.search(r'GFLOPS=([0-9.]+)', line)
        if not m:
            continue
        g = float(m.group(1))
        # 标签：行首 name（cuda_core）或 [tc_xx ...]（tensor）
        lm = re.match(r'\s*\[([^\]]+)\]', line) or re.match(r'\s*([A-Za-z0-9_]+):', line) or re.match(r'\s*([A-Za-z0-9_]+)\s', line)
        label = lm.group(1).strip() if lm else "?"
        out.append((label, g))
    return out

# headline: 逐示例日志在 bench/<engine>/<id>_<name>_<size>.log，每文件一个 kernel。取最大尺寸那组。
BENCH_SUBDIR = {"fp32": "cuda_core_fp32", "bf16": "cuda_core_bf16", "tc": "tensor_core"}
def latest_bench(kind):
    d = os.path.join(RUN_DIR, "bench", BENCH_SUBDIR[kind])
    files = glob.glob(os.path.join(d, "*.log"))
    if not files: return None, None
    def sz(f):
        m = re.search(r'_(\d+)\.log$', f); return int(m.group(1)) if m else 0
    maxsz = max((sz(f) for f in files), default=0)
    rows = []
    for f in sorted(files):          # 文件名 00_/01_/… → id 顺序
        if sz(f) != maxsz: continue
        rows += parse_bench(open(f).read())
    return maxsz, rows

sz_fp32, fp32 = latest_bench("fp32")
sz_bf16, bf16 = latest_bench("bf16")
sz_tc, tc = latest_bench("tc")
HSZ = sz_fp32 or sz_bf16 or sz_tc or "?"

def gof(rows, key):
    if not rows: return None
    for lbl, g in rows:
        if key in lbl: return g
    return None

# ---- 正确性（逐示例日志在 verify/<engine>/*.log）----
def verify_stats(subdir):
    p = f = 0
    for fl in glob.glob(os.path.join(RUN_DIR, "verify", subdir, "*.log")):
        t = open(fl).read()
        p += len(re.findall(r'PASS', t)); f += len(re.findall(r'FAIL', t))
    return p, f

# ---- 遥测：实测 SM(pclk) 时钟 → (计算期典型, 峰值)。按表头定位 pclk 列，排除 mclk(显存)/空闲 ----
def clock_range():
    t = read("telemetry_dmon.txt")
    lines = t.splitlines()
    idx = None
    for line in lines:
        if line.lstrip().startswith("#") and "pclk" in line:
            toks = line.replace("#", " ").split()
            if "pclk" in toks:
                idx = toks.index("pclk"); break
    if idx is None:
        return None
    clks = []
    for line in lines:
        s = line.strip()
        if not s or s.startswith("#"): continue
        parts = line.split()
        if len(parts) > idx:
            try: clks.append(int(parts[idx]))
            except ValueError: pass
    active = [c for c in clks if c > 0]
    if not active: return None
    peak = max(active)
    comp = sorted(c for c in active if c >= peak * 0.6)   # 计算期(高频簇)
    typ = comp[len(comp)//2] if comp else peak            # 中位数 ≈ 计算期典型频率
    return typ, peak

# ============================================================================
# 生成报告
# ============================================================================
def pct(g, peak):
    return f"{g/(peak*1000)*100:5.1f}%" if (g and peak) else "  -  "

cublas_fp32 = gof(fp32, "cublas")
cublas_bf16 = gof(bf16, "cublas")

md = []
md.append(f"# GEMM 测试汇总 — {GPU_NAME}\n")
md.append(f"- 计算能力: **CC {CC}**  |  SM 数: {SM}  |  FP32 核/SM: {CORES}")
md.append(f"- 时钟策略: **{CLOCK_POLICY}**（额定 max {int(MAXCLK) if MAXCLK else '?'} MHz）")
cr = clock_range()
if cr: md.append(f"- 运行时实测 SM 时钟(遥测 pclk): 计算期典型 **{cr[0]} MHz** / 峰 {cr[1]} MHz（额定 {int(MAXCLK) if MAXCLK else '?'}）")
peak_bits = []
if FP32_PEAK: peak_bits.append(f"FP32 = **{FP32_PEAK:.1f} TFLOPS**")
if BF16_PK:   peak_bits.append(f"BF16 张量核 = **{BF16_PK:.0f} TFLOPS**")
if FP8_PK:    peak_bits.append(f"FP8 张量核 = **{FP8_PK:.0f} TFLOPS**")
if FP4_PK:    peak_bits.append(f"FP4 张量核 = **{FP4_PK:.0f} TFLOPS**")
md.append("- 理论峰值: " + ("  |  ".join(peak_bits) if peak_bits else "未知")
          + ("" if BF16_PK else "（BF16 峰值未知，张量核只以 cuBLAS 为基线）"))
md.append(f"- headline 尺寸: {HSZ}³\n")

# 正确性
vp32 = verify_stats("cuda_core_fp32"); vb16 = verify_stats("cuda_core_bf16"); vtc = verify_stats("tensor_core")
md.append("## 正确性（对拍 cuBLAS）")
md.append(f"- FP32: {vp32[0]} PASS / {vp32[1]} FAIL  |  BF16: {vb16[0]} PASS / {vb16[1]} FAIL  |  Tensor(WMMA/WGMMA): {vtc[0]} PASS / {vtc[1]} FAIL\n")

# FP32 表
if fp32:
    md.append(f"## FP32 (CUDA core) @ {sz_fp32}³")
    md.append("| kernel | GFLOPS | %FP32峰值 | %cuBLAS |")
    md.append("| --- | ---: | ---: | ---: |")
    for lbl, g in fp32:
        cub = f"{g/cublas_fp32*100:5.1f}%" if cublas_fp32 else "-"
        md.append(f"| {lbl} | {g:.0f} | {pct(g, FP32_PEAK)} | {cub} |")
    md.append("")

# BF16 表
if bf16:
    md.append(f"## BF16 (CUDA core, bf16输入+fp32累加) @ {sz_bf16}³")
    hdr = "| kernel | GFLOPS | %BF16峰值 | %cuBLAS |" if BF16_PK else "| kernel | GFLOPS | %cuBLAS |"
    md.append(hdr); md.append("| --- | ---: | ---: | ---: |" if BF16_PK else "| --- | ---: | ---: |")
    for lbl, g in bf16:
        cub = f"{g/cublas_bf16*100:5.1f}%" if cublas_bf16 else "-"
        if BF16_PK: md.append(f"| {lbl} | {g:.0f} | {pct(g, BF16_PK)} | {cub} |")
        else: md.append(f"| {lbl} | {g:.0f} | {cub} |")
    md.append("")

# WMMA/Tensor 表：每个用例按其精度选峰值——FP8 用例对 FP8 峰值(=2×BF16)算利用率，
# 不能套 BF16 峰值(否则 226T/148T 会算出 >150% 的"超过物理峰值"假象)。
if tc:
    md.append(f"## Tensor Core @ {sz_tc}³")
    md.append("> 利用率口径：WMMA/WGMMA(bf16)→BF16 峰值；FP8→FP8 峰值(=2×BF16)；FP4→FP4 峰值(=4×BF16)。")
    md.append("| 用例 | 精度 | GFLOPS | %对应精度峰值 | %cuBLAS_bf16 |")
    md.append("| --- | --- | ---: | ---: | ---: |")
    for lbl, g in tc:
        peak, prec = G.peak_for_label(lbl, GPU_NAME, CC)
        cub = f"{g/cublas_bf16*100:5.1f}%" if cublas_bf16 else "-"
        md.append(f"| {lbl} | {prec} | {g:.0f} | {pct(g, peak)} | {cub} |")
    md.append("")

# 缩放
scal = read("09_scaling.log")
if scal.strip():
    md.append("## 尺寸缩放")
    md.append("```"); md.append(scal.strip()); md.append("```\n")

# 关键结论
md.append("## 关键数字")
best_fp32 = max((g for l,g in fp32 if "cublas" not in l), default=None) if fp32 else None
# 张量核分精度各取「手写」最佳(排除内嵌的 cuBLAS fair 参考行)：BF16 路径 vs FP8 路径，各对自己峰值
tc_bf16 = [(l,g) for l,g in tc if "fp8" not in l.lower() and "fp4" not in l.lower() and "cublas" not in l.lower()] if tc else []
tc_fp8  = [(l,g) for l,g in tc if "fp8" in l.lower() and "cublas" not in l.lower()] if tc else []
best_tc_bf16 = max((g for _,g in tc_bf16), default=None)
best_tc_fp8  = max((g for _,g in tc_fp8),  default=None)
if cublas_fp32: md.append(f"- FP32 cuBLAS: **{cublas_fp32/1000:.1f} TFLOPS**" + (f"（{cublas_fp32/(FP32_PEAK*1000)*100:.0f}% 峰值）" if FP32_PEAK else ""))
if best_fp32: md.append(f"- FP32 手写最佳: **{best_fp32/1000:.1f} TFLOPS**" + (f"（{best_fp32/cublas_fp32*100:.0f}% cuBLAS）" if cublas_fp32 else ""))
if cublas_bf16: md.append(f"- BF16 cuBLAS: **{cublas_bf16/1000:.1f} TFLOPS**" + (f"（{cublas_bf16/(BF16_PK*1000)*100:.0f}% BF16峰值）" if BF16_PK else ""))
if best_tc_bf16: md.append(f"- 手写张量核最佳(BF16 路径): **{best_tc_bf16/1000:.1f} TFLOPS**" + (f"（{best_tc_bf16/(BF16_PK*1000)*100:.0f}% BF16峰值）" if BF16_PK else ""))
if best_tc_fp8: md.append(f"- 手写张量核最佳(FP8 路径): **{best_tc_fp8/1000:.1f} TFLOPS**" + (f"（{best_tc_fp8/(FP8_PK*1000)*100:.0f}% FP8峰值；≈{best_tc_fp8/(BF16_PK*1000)*100:.0f}% BF16峰值，因 FP8 吞吐是 BF16 的 2×）" if (FP8_PK and BF16_PK) else ""))
md.append("")
md.append(f"> 时钟提示：GFLOPS 随实际 SM 时钟线性变化。本轮时钟策略={CLOCK_POLICY}"
          + (f"，实测计算期约 {cr[0]} MHz(峰 {cr[1]})" if cr else "")
          + "。跨机型/跨轮对比 MFU 时务必统一时钟策略（`--lock` 锁额定频最可复现）。")

md_text = "\n".join(md)
open(os.path.join(RUN_DIR, "00_summary.md"), "w").write(md_text)

# 纯文本精简版（控制台）
txt = []
txt.append(f"===== GEMM 汇总: {GPU_NAME} (CC {CC}, {SM} SM) =====")
txt.append(f"时钟: {CLOCK_POLICY}" + (f" | 计算期~{cr[0]}MHz(峰{cr[1]})" if cr else ""))
if FP32_PEAK: txt.append(f"峰值: FP32 {FP32_PEAK:.1f}T" + (f" | BF16 {BF16_PK:.0f}T" if BF16_PK else "") + (f" | FP8 {FP8_PK:.0f}T" if FP8_PK else ""))
txt.append(f"正确性: FP32 {vp32[0]}P/{vp32[1]}F, BF16 {vb16[0]}P/{vb16[1]}F, TC {vtc[0]}P/{vtc[1]}F")
if cublas_fp32: txt.append(f"FP32 cuBLAS {cublas_fp32/1000:.1f}T | 手写最佳 {best_fp32/1000:.1f}T" if best_fp32 else f"FP32 cuBLAS {cublas_fp32/1000:.1f}T")
if cublas_bf16: txt.append(f"BF16 cuBLAS {cublas_bf16/1000:.1f}T" + (f" ({cublas_bf16/(BF16_PK*1000)*100:.0f}%峰)" if BF16_PK else ""))
if best_tc_bf16: txt.append(f"手写TC(BF16)最佳 {best_tc_bf16/1000:.1f}T" + (f" ({best_tc_bf16/(BF16_PK*1000)*100:.0f}%峰)" if BF16_PK else ""))
if best_tc_fp8: txt.append(f"手写TC(FP8)最佳 {best_tc_fp8/1000:.1f}T" + (f" ({best_tc_fp8/(FP8_PK*1000)*100:.0f}%FP8峰)" if FP8_PK else ""))
txt.append(f"报告: {os.path.join(RUN_DIR,'00_summary.md')}")
txt_text = "\n".join(txt)
open(os.path.join(RUN_DIR, "00_summary.txt"), "w").write(txt_text + "\n")
print(txt_text)
