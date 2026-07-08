#!/usr/bin/env bash
# SASS opcode histogram for one compiled object/binary — the static, no-GPU,
# deterministic analysis tier BELOW ncu. Answers "which machine instructions did
# this rung emit, and how many" — exactly the thing an incremental ladder must
# prove but ncu cannot show ("reduce IMAD.MOV count" is a cuobjdump fact, not a
# Nsight metric).
#   usage: sass_count.sh <file.o|binary> [kernel_name_regex]
# prints:  <count> <OPCODE>   (sorted desc), plus a total line.
set -u
F="${1:?usage: sass_count.sh <file.o|binary> [kernel_regex]}"
KRE="${2:-}"
CUOBJDUMP="${CUOBJDUMP:-/usr/local/cuda-12.8/bin/cuobjdump}"

dump=$("$CUOBJDUMP" -sass "$F" 2>/dev/null)
[ -z "$dump" ] && { echo "no SASS in $F (not a cubin/.o with device code?)" >&2; exit 1; }

# optionally restrict to one kernel's disassembly block
if [ -n "$KRE" ]; then
  dump=$(printf '%s\n' "$dump" | awk -v re="$KRE" '
    /Function :/ { inblk = ($0 ~ re) }
    { if (inblk) print }')
fi

# SASS line looks like:  /*0080*/  @!P0 IMAD.MOV.U32 R5, RZ, RZ, R4 ;
# extract the opcode = first token after an optional @predicate, strip modifiers
printf '%s\n' "$dump" \
  | grep -oE '/\*[0-9a-f]+\*/[[:space:]]+(@!?P[0-9]+[[:space:]]+)?[A-Z][A-Z0-9._]+' \
  | sed -E 's#.*/\*[0-9a-f]+\*/[[:space:]]+(@!?P[0-9]+[[:space:]]+)?##; s#\..*##' \
  | grep -vE '^$' \
  | sort | uniq -c | sort -rn \
  | awk '{printf "%8d  %s\n", $1, $2; t+=$1} END{printf "%8d  TOTAL\n", t}'
