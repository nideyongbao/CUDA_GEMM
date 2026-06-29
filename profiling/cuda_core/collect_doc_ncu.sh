#!/bin/bash
# 为 docs 逐个 kernel 复跑“真实命令”@4096³（H20，sudo+全路径）。每核 5 个探针：
#  P1 SOL(--set basic)  P2 stalls(WarpStateStats+ratio)  P3 SchedulerStats
#  P4 访存合并(sectors/req ld+st)  P5 bank conflict(shared ld+st)
# 输出：profiling/cuda_core/doc_raw/<name>.txt（按探针分段，便于贴回 docs）。
cd /shard_data/brooksli/workspace/0625/CUDA_MATMUAL
SZ=4096
N="sudo -n /usr/local/cuda/bin/ncu"
B=kernels/cuda_core/bench
OUT=profiling/cuda_core/doc_raw
mkdir -p "$OUT"
STALL="smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio,smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio,smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio,smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio"
COAL="l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio,l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_st.ratio"
BANK="l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum"
clean(){ grep -vE "^==(PROF|WARNING|ERROR)==|Unable to access|: M=|M=.* N=.* K=|util\(|^\[tc_"; }

# name id regex
run(){
  local name=$1 id=$2 rgx=$3 f="$OUT/$1.txt"
  echo "==== $name (id $id, regex $rgx) @ ${SZ}^3 ===="
  : > "$f"
  echo "### P1 SOL (ncu --set basic)" >> "$f"
  $N --set basic -k "regex:$rgx" --launch-count 1 $B $id $SZ $SZ $SZ 2>&1 | clean >> "$f"
  echo -e "\n### P2 STALLS (WarpStateStats + stall ratio)" >> "$f"
  $N --section WarpStateStats --metrics "$STALL" -k "regex:$rgx" --launch-count 1 $B $id $SZ $SZ $SZ 2>&1 | clean >> "$f"
  echo -e "\n### P3 SCHEDULER (SchedulerStats)" >> "$f"
  $N --section SchedulerStats -k "regex:$rgx" --launch-count 1 $B $id $SZ $SZ $SZ 2>&1 | clean >> "$f"
  echo -e "\n### P4 COALESCE (global sectors/req ld,st)" >> "$f"
  $N --metrics "$COAL" -k "regex:$rgx" --launch-count 1 $B $id $SZ $SZ $SZ 2>&1 | clean >> "$f"
  echo -e "\n### P5 BANK CONFLICT (shared ld,st)" >> "$f"
  $N --metrics "$BANK" -k "regex:$rgx" --launch-count 1 $B $id $SZ $SZ $SZ 2>&1 | clean >> "$f"
  echo "  done -> $f"
}

run naive        1  naive_kernel
run smem         2  smem_kernel
run blocktiling  3  blocktiling_kernel
run Dblocktiling 4  Dblocktiling_kernel
run vectorized   5  vectorized_kernel
run warptile     9  warptile_kernel
run warptile_vec 10 warptile_vec_kernel
run bankconflict 11 bank_conflict_kernel
run doublebuffer 12 double_buffer_kernel
sudo -n chown -R $(id -u):$(id -g) "$OUT" 2>/dev/null || true
echo "ALL_DOC_NCU_DONE"
