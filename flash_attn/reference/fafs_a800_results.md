# flash_attention_from_scratch 在本机 A800 上的实测参考曲线

我们把参考仓库 fafs **实际编译并跑在本机 A800** 上(`bench_fafs.py`),得到权威性能曲线,用来校准我们手写的 `fa_tc` 阶梯。**这修正了一个重要误解**:fafs 的 "99.2% FA2" 是 A100 + 其特定 benchmark 配置下的数,**在本机 A800 的 B2 H32 S4096 D128 上,fafs 自己也只到 75% FA2**。

配置:B2 H32 S4096 D128 fp16 非causal,A800 锁频 1410MHz,口径 4·B·H·S²·D。

## fafs 自带的 7 级 curated 进阶(Br=64/Bc=64/4warps —— 与我们 fa_tc 完全同配置)

| rung | 特性 | TFLOPS |
| ---: | --- | ---: |
| 1 | async base(无 swizzle) | 34.9 |
| 2 | **+swizzle** | **139.6** |
| 3 | +eager load | 140.5 |
| 4 | +load tiles→RF | 144.8 |
| 5 | +double-buffer | **145.3** |
| 6 | +opt_softmax(exp2) | 144.4 |
| 7 | (load_0_2_2 变体) | 101.0（此配置下回退） |

**fafs 进阶在本配置封顶 ~145 TFLOPS(68% FA2)。**

## fafs 全部 80 个已编译配置里最快的(autotune 扫描)

| TFLOPS | 配置 |
| ---: | --- |
| **159.9** | **Br=128**/Bc=64/4warps + eager+swizzle+load_tiles+buffer+opt_softmax |
| 158.4 | Br=128 … |
| …前 8 名**全是 Br=128** | |

**fafs 在本机 A800 的真实峰值 = 159.9 TFLOPS = 75% FA2**,且关键增量来自 **Br=128**(更大 Q tile,4warps×2 M-tile)。

## 与我们手写 `fa_tc` 阶梯的对账(按"实际特性"对齐)

| 特性层 | fafs | 我们 fa_tc | 结论 |
| --- | ---: | ---: | --- |
| base(无 swizzle) | 34.9(async) | 33.3(sync) | ✅ 吻合 |
| swizzle | 139.6(async+swz) | 134.9(fa_tc_03 cpasync+swz) | ✅ 吻合（±3%）|
| 进阶封顶(Br=64) | 145.3 | **146.3(fa_tc_05)** | ✅ **吻合/略高** |
| autotune 峰值 | **159.9(Br=128)** | —（未实现 Br=128 2-M-tile） | fafs +10% |
| double-buffer | 145.3(+0.5% vs 无) | 137.7(fa_tc_06,占用率↓net loss) | 都印证:buffer 在此配置**边际/无用** |

> **关键更正**:我们早先说"fa_tc_02 swizzle 49% 远低于 fafs 72.6%"是**误比**——fafs rung2 已含 async,我们 fa_tc_02 是 sync;把 async 对齐后(我们 fa_tc_03 vs fafs rung2)是 **134.9 vs 139.6,吻合**。**我们的手写阶梯在同配置(Br=64)与 fafs 进阶逐级吻合、封顶都在 ~145–146 TFLOPS。**

## 完整性能地图(本机 A800,B2 H32 S4096 D128 fp16 nc)

| 实现 | TFLOPS | %FA2 | 说明 |
| --- | ---: | ---: | --- |
| 我们 fa_tc_01 base | 33 | 16% | = fafs base |
| 我们 fa_tc_05(顶,Br=64) | **146** | 68% | = fafs Br=64 进阶封顶 |
| **fafs autotune 峰值(Br=128)** | **160** | 75% | 关键增量=Br=128;我们未实现该 2-M-tile 变体 |
| TinyFA(CuTe 终态) | 200 | 93% | CuTe 更优 tiling/多级流水 |
| cuBLAS-级 官方 FA2(2.8.3) | 214 | 100% | 参考上限 |

## 到 99.2% / TinyFA 的差距 = 三层,已全部定位
1. **Br=128(autotune)**:+10%(146→160)。需 4warps×2-M-tile 重构(寄存器/占用率再平衡)。fafs 靠它拿到 A800 峰值。
2. **CuTe 级 tiling/多级流水**:160→200(TinyFA)。CuTe 的 layout 代数 + 多 stage pipeline,超出可读手写 raw-CUDA 的性价比区。
3. **官方 FA2 的 warp-specialization 等**:200→214。
> "99.2% FA2" 是 **A100 + fafs 自身 benchmark 配置**下、对 A100 FA2(186.4)算的;**在 A800 的本配置下,fafs 本身也只到 75% FA2**——所以"本地没复现 99.2%"不是 bug,而是**配置/机器口径差异 + 我们止于 Br=64 可读阶梯**。
