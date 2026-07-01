#!/usr/bin/env python3
# ============================================================================
# 对比两轮 run_all.sh 结果（通常是 --both 的 default vs locked），逐 kernel 出差异 + 时钟 + 结论。
# 用法: gemm_compare.py <DIR_A(默认)> <DIR_B(锁频)>   → markdown 到 stdout
# ============================================================================
import sys, os, re, glob

A, B = sys.argv[1], sys.argv[2]

def read(p):
    return open(p, encoding="utf-8").read() if os.path.exists(p) else ""

def policy(d):
    t = read(os.path.join(d, "00_clock_policy.txt")).strip()
    return t or os.path.basename(d)

def gflops(p):
    m = re.search(r'GFLOPS=([0-9.]+)', read(p))
    return float(m.group(1)) if m else None

def clock_typ(d):  # 遥测 pclk 计算期典型 + 峰值
    lines = read(os.path.join(d, "telemetry_dmon.txt")).splitlines()
    idx = None
    for ln in lines:
        if ln.lstrip().startswith("#") and "pclk" in ln:
            toks = ln.replace("#", " ").split()
            if "pclk" in toks: idx = toks.index("pclk"); break
    if idx is None: return None
    clks = []
    for ln in lines:
        s = ln.strip()
        if not s or s.startswith("#"): continue
        parts = ln.split()
        if len(parts) > idx:
            try: clks.append(int(parts[idx]))
            except ValueError: pass
    active = [c for c in clks if c > 0]
    if not active: return None
    peak = max(active)
    comp = sorted(c for c in active if c >= peak*0.6)
    return comp[len(comp)//2] if comp else peak, peak

ENGINES = [("cuda_core_fp32", "FP32"), ("cuda_core_bf16", "BF16"), ("tensor_core", "Tensor Core")]

out = []
out.append(f"# 两轮对比：默认 vs 锁频\n")
out.append(f"- A（默认）: `{os.path.relpath(A)}`  —  {policy(A)}")
out.append(f"- B（锁频）: `{os.path.relpath(B)}`  —  {policy(B)}")
ca, cb = clock_typ(A), clock_typ(B)
if ca and cb:
    out.append(f"- 遥测 SM 时钟（计算期典型/峰）: A **{ca[0]}/{ca[1]} MHz** · B **{cb[0]}/{cb[1]} MHz**")
out.append("")

maxdiff = 0.0; noisy = []
for sub, title in ENGINES:
    files = sorted(glob.glob(os.path.join(A, "bench", sub, "*.log")))
    if not files: continue
    rows = []
    for fa in files:
        name = os.path.basename(fa)
        ga, gb = gflops(fa), gflops(os.path.join(B, "bench", sub, name))
        if ga is None or gb is None: continue
        d = (gb-ga)/ga*100 if ga else 0
        rows.append((name, ga, gb, d))
        if abs(d) > abs(maxdiff): maxdiff = d
        if abs(d) > 5: noisy.append((title, name, ga, gb, d))
    if not rows: continue
    out.append(f"## {title} @ headline（默认 → 锁频，Δ%）")
    out.append("| kernel | 默认 GFLOPS | 锁频 GFLOPS | Δ% |")
    out.append("| --- | ---: | ---: | ---: |")
    for name, ga, gb, d in rows:
        star = " ⚠️" if abs(d) > 5 else ""
        nm = re.sub(r'_\d+\.log$', '', name)
        out.append(f"| {nm} | {ga:.0f} | {gb:.0f} | {d:+.1f}%{star} |")
    out.append("")

# 正确性 + 汇总
def vpass(d, sub):
    p = f = 0
    for fl in glob.glob(os.path.join(d, "verify", sub, "*.log")):
        t = read(fl); p += t.count("PASS"); f += t.count("FAIL")
    return p, f
out.append("## 正确性（两轮均应全 PASS）")
for sub, title in ENGINES:
    pa, fa = vpass(A, sub); pb, fb = vpass(B, sub)
    out.append(f"- {title}: A {pa}P/{fa}F · B {pb}P/{fb}F")
out.append("")

out.append("## 结论")
if noisy:
    out.append(f"- **默认轮有 {len(noisy)} 个 kernel 因 boost 抖动被欠采（|Δ|>5%）**，锁频后回正：")
    for title, name, ga, gb, d in noisy:
        nm = re.sub(r'_\d+\.log$', '', name)
        out.append(f"  - {title} `{nm}`: {ga:.0f} → {gb:.0f}（{d:+.1f}%）")
    out.append("- 其余 kernel 两轮基本一致（低功耗或恰好撞满频）。**要可复现/公平的逐 kernel 数字用锁频；默认仅适合看整体趋势/真实开箱值。**")
else:
    out.append("- 两轮所有 kernel 差异 <5%：本次默认轮恰好全程接近满频。**但默认不保证如此**（boost 抖动随机），跨轮/跨机对比仍建议锁频。")
out.append(f"- 最大单核差异: {maxdiff:+.1f}%。")

print("\n".join(out))
