#!/usr/bin/env python3
"""Diff two SASS opcode histograms (from sass_count.sh) into a markdown table.
This is the instruction-level "one-delta-per-rung" artifact: it makes visible the
gains that ncu cannot (e.g. rung N->N+1 removed 130 CS2R and 290 MOV instructions
while HMMA count stays constant -> same tensor-core work, less integer overhead).

  usage: sass_compare.py <before.txt> <after.txt> [--before-name A --after-name B]
where each *.txt is the output of common/sass_count.sh.
"""
import sys, re

def load(path):
    h = {}
    for line in open(path):
        m = re.match(r"\s*(\d+)\s+([A-Z][A-Z0-9._]*|TOTAL)\s*$", line)
        if m:
            h[m.group(2)] = int(m.group(1))
    return h

def main():
    a_name, b_name = "before", "after"
    args = sys.argv[1:]
    files = []
    i = 0
    while i < len(args):
        if args[i] == "--before-name": a_name = args[i+1]; i += 2
        elif args[i] == "--after-name": b_name = args[i+1]; i += 2
        else: files.append(args[i]); i += 1
    if len(files) != 2:
        print(__doc__); sys.exit(1)
    A, B = load(files[0]), load(files[1])
    keys = sorted(set(A) | set(B), key=lambda k: (k == "TOTAL", -max(A.get(k,0), B.get(k,0))))
    print(f"| Instr | {a_name} | {b_name} | Δ |")
    print("|---|---:|---:|---:|")
    for k in keys:
        a, b = A.get(k, 0), B.get(k, 0)
        if a == b and k != "TOTAL":
            continue
        d = b - a
        print(f"| {k} | {a} | {b} | {d:+d} |")

if __name__ == "__main__":
    main()
