对应用例：`kernels/tensor_core/tc_05_wgmma_fp8.cu`（独立可执行 `tc_05_wgmma_fp8`，需 `-lcuda`）。

## 0、为什么补 FP8
H20 的算力分档：FP32 ~44 TFLOPS、BF16 Tensor Core ~148 TFLOPS、**FP8(e4m3) Tensor Core ~296 TFLOPS**（BF16 的两倍）。LLM 推理/训练越来越多用 FP8，所以 tensor core 这条线必须把 FP8 补上。

本用例在 [11 WGMMA](11%20tensor%20core%20-%20WGMMA%20TMA%20warp%20specialization.md)（BF16）基础上换成 FP8，改动出奇地少，正好说明 **Hopper 这套异步流水线对精度是"可插拔"的**：
- ① 元素类型 bf16(2B) → `__nv_fp8_e4m3`(1B)；
- ② WGMMA 指令 `m64n128k16.f32.bf16.bf16` → `m64n128k32.f32.e4m3.e4m3`（FP8 的 K 维是 32，且**没有 transA/transB 立即数**——FP8/INT8 只支持 TN）；
- ③ BK 64 → 128，让 smem 每行仍是 **128 字节**，对齐 128B swizzle；TMA dtype 用 `UINT8`。

为什么描述符魔数(leading=16, stride=1024)能直接沿用：因为 smem 的**字节布局**和 BF16 完全一致——128 行 × 128 字节，每条 wgmma 沿 K 跨 32 字节（bf16 是 16×2B、fp8 是 32×1B，都是 32 字节）。Hopper 的 swizzle/core-matrix 是按字节定义的，所以换精度不换布局。

## 1、正确性怎么验
FP8 没有简单的 `cublasGemmEx` 路径（要走 cuBLASLt，需 TN + scale，较繁），所以本用例用 **CPU double 参考**：把同一份 fp8 输入解码成 float、做双精度累加，在小尺寸(512³)上对拍。
```
[tc_05 WGMMA_fp8] VERIFY(512^3, vs CPU double) max_abs=2.927e-02 bad=0/262144  PASS
```
max_abs 2.9e-2 看着比 bf16 大，但这是 fp8(e4m3 只有 3 位尾数)的固有粗糙 + 大 C 值上的相对误差，落在 allclose 容差内 → kernel 算的是对的。

## 2、观大局
```
sudo ncu --set full -k regex:"wgmma_fp8_kernel" -s 1 -c 1 ./tc_05_wgmma_fp8 2048 2048 2048
```

```
    Metric Name             Metric Unit Metric Value
    ----------------------- ----------- ------------
    Duration                         us       108.64
    Compute (SM) Throughput           %        58.87
    L1/TEX Cache Throughput           %        22.88
    DRAM Throughput                   %         1.94
    Achieved Occupancy                %         7.40
    Registers Per Thread  register/thread       154
    SM Busy                           %        73.09
```
和 BF16 WGMMA 同构：L1/TEX 低（TMA 走专用路径）、SM Busy 高（73%）、占用率仅 7.4%。头号 stall 同样是 CTA barrier(48.7%)——生产者/消费者交接。结构完全没变，只是吃的是 fp8。

## 3、性能：吞吐翻倍
```
4096³: GFLOPS=224245  util(vs296T)=75.8%  (= 1.88× BF16 WGMMA 的 119311)
8192³: GFLOPS=250949  util(vs296T)=84.8%
```
FP8 把同一条流水线的吞吐**几乎翻倍**（224 vs 119 TFLOPS），对 FP8 峰值(296T)的利用率 75.8%（8192³ 84.8%）。代价是精度：e4m3 只有 3 位尾数，实际用时通常配 per-tensor/per-block scaling（本用例为最简没做 scaling）。

## 总结
FP8 WGMMA 复用了 BF16 那套 TMA+WGMMA+warp specialization 流水线，**只改类型/指令/BK 三处**就拿到 ~2× 吞吐、75.8% 的 FP8 峰值利用率（8192³ 84.8%），并用 CPU double 参考验证正确。

这印证了 Hopper GEMM 的工程范式：**先把异步流水线骨架搭对，精度(bf16/fp8/未来 fp4)只是骨架上的可插拔参数。** 完整的精度/利用率阶梯见 [H20复现结论](H20%E5%A4%8D%E7%8E%B0%E7%BB%93%E8%AE%BA.md)。
