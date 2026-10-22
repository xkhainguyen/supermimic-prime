"""Summarize compare_grads.py results: per window, mean WM-vs-sim cosine and relative error over blocks and steps."""

import glob
import json
import sys

import numpy as np

rows = []
for f in sorted(glob.glob(sys.argv[1] if len(sys.argv) > 1 else "results/grads/*.json")):
    r = json.load(open(f))
    line = [f"{r['select']:9s}", f"{r['clip']:24s}", f"t0={r['t0'] * 0.02:5.2f}s"]
    for key in ("d_s/d_a0", "d_s/d_s0"):
        cos = np.array([[x["cos"] for x in r[key][b]] for b in r[key]])  # (blocks, N)
        rel = np.array([[x["rel"] for x in r[key][b]] for b in r[key]])
        nr = np.array([[x["norm_wm"] / max(x["norm_sim"], 1e-12) for x in r[key][b]] for b in r[key]])
        line.append(f"{key}: cos {cos.mean():.2f} (min {cos.min():.2f}) rel {rel.mean():.2f} |wm|/|sim| {np.median(nr):.2f}")
    line.append(f"open-loop err max {max(r['wm_open_loop_max_abs_state_err']):.2f}")
    print("  ".join(line))
