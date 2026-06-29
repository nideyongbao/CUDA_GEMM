#!/bin/bash
# 为 docs 08–12 逐个 tensor core 用例复跑“真实命令”@2048³（H20，sudo+全路径），
# 与 cuda_core 的 collect_doc_ncu.sh 对称。每个用例 5 个探针：
#  P1 SOL(--set basic)  P2 stalls(WarpStateStats+ratio，含 barrier/membar)
#  P3 SchedulerStats    P4 ComputeWorkloadAnalysis(看哪条 pipe 最高=Tensor)
#  P5 访存(WMMA 取 fragment 的 sectors/req + bank；WGMMA 走 TMA 仅作对照)
# 驱动 kernels/tensor_core/bench <id 1-5>；-k regex 只抓主核（跳过 tc_04/05 的 transpose_*），
# -s 1 -c 1 跳过第 1 次、剖析第 2 次(warmup)。尺寸 2048³，与 docs 08–12 / SUMMARY.md 一致。
# 输出：profiling/tensor_core/doc_raw/<name>.txt（按探针分段）。
cd /shard_data/brooksli/workspace/0625/CUDA_MATMUAL
SZ=2048
N="sudo -n /usr/local/cuda/bin/ncu"
B=kernels/tensor_core/bench
OUT=profiling/tensor_core/doc_raw
mkdir -p "$OUT"
STALL="smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio,smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio,smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio,smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio,smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio,smsp__average_warps_issue_stalled_membar_per_issue_active.ratio,smsp__average_warps_issue_stalled_wait_per_issue_active.ratio,smsp__average_warps_issue_stalled_lg_throttle_per_issue_active.ratio"
COAL="l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio,l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_st.ratio"
BANK="l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum"
clean(){ grep -vE "^==(PROF|WARNING|ERROR)==|Unable to access|VERIFY|: M=|M=.* N=.* K=|util\(|^\[tc_"; }

# name id regex
run(){
  local name=$1 id=$2 rgx=$3 f="$OUT/$1.txt"
  echo "==== $name (id $id, regex $rgx) @ ${SZ}^3 ===="
  : > "$f"
  echo "### P1 SOL (ncu --set basic)" >> "$f"
  $N --set basic -k "regex:$rgx" -s 1 -c 1 $B $id $SZ $SZ $SZ 2>&1 | clean >> "$f"
  echo -e "\n### P2 STALLS (WarpStateStats + stall ratio，含 barrier/membar)" >> "$f"
  $N --section WarpStateStats --metrics "$STALL" -k "regex:$rgx" -s 1 -c 1 $B $id $SZ $SZ $SZ 2>&1 | clean >> "$f"
  echo -e "\n### P3 SCHEDULER (SchedulerStats)" >> "$f"
  $N --section SchedulerStats -k "regex:$rgx" -s 1 -c 1 $B $id $SZ $SZ $SZ 2>&1 | clean >> "$f"
  echo -e "\n### P4 PIPE (ComputeWorkloadAnalysis：最高 pipe / Tensor)" >> "$f"
  $N --section ComputeWorkloadAnalysis -k "regex:$rgx" -s 1 -c 1 $B $id $SZ $SZ $SZ 2>&1 | clean >> "$f"
  echo -e "\n### P5 MEM (global sectors/req + shared bank；WMMA 有意义，WGMMA 走 TMA 仅对照)" >> "$f"
  $N --metrics "$COAL,$BANK" -k "regex:$rgx" -s 1 -c 1 $B $id $SZ $SZ $SZ 2>&1 | clean >> "$f"
  echo "  done -> $f"
}

run tc_01_wmma_naive   1 wmma_naive_kernel
run tc_02_wmma_smem    2 wmma_smem_kernel
run tc_03_wmma_pipe    3 wmma_pipe_kernel
run tc_04_wgmma_tma_ws 4 wgmma_kernel
run tc_05_wgmma_fp8    5 wgmma_fp8_kernel
sudo -n chown -R $(id -u):$(id -g) "$OUT" 2>/dev/null || true
echo "ALL_TC_DOC_NCU_DONE"
