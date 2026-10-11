#!/usr/bin/env python3
"""R2-15 board latency benchmark: generate random-weight header files for each architecture configuration.

Per-window latency depends only on tensor sizes and where weights are stored (Flash constants), not on weight
values. Array sizes follow pico-former/firmware/include/ceqt_weights.h, scaled by the configuration's d, number of
layers and number of CGM points; values are drawn from U(-1/sqrt(fan_in), 1/sqrt(fan_in)) with a fixed random seed.

Usage: python gen_bench_weights.py
Output: include/bench_w_<config>.h, include/bench_select.h
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
INC = HERE / "include"

# name: (d, layers, heads, n_cgm, max_marks), consistent with CONFIGS in ../train_grid.py
CONFIGS = {
    "d32_l1": (32, 1, 4, 72, 24), "d32_l2": (32, 2, 4, 72, 24), "d32_l3": (32, 3, 4, 72, 24),
    "d48_l1": (48, 1, 4, 72, 24), "base": (48, 2, 4, 72, 24), "d48_l3": (48, 3, 4, 72, 24),
    "d64_l1": (64, 1, 4, 72, 24), "d64_l2": (64, 2, 4, 72, 24), "d64_l3": (64, 3, 4, 72, 24),
    "d96_l2": (96, 2, 4, 72, 24),
    "L48": (48, 2, 4, 36, 12), "L60": (48, 2, 4, 36, 24), "L84": (48, 2, 4, 72, 12),
    "h2": (48, 2, 2, 72, 24), "h8": (48, 2, 8, 72, 24),
}
N_OUT, N_EVT = 48, 16


def arrays(d, layers, n_cgm):
    ff = 2 * d
    a = [("W_POS", n_cgm * d, d), ("W_CURVE_Q", N_OUT * d, d), ("W_EVENT_Q", N_EVT * d, d),
         ("W_CGM", d, 1), ("B_CGM", d, 1), ("W_TYPE", 4 * d, d), ("W_MARK1", 2 * d, 2), ("B_MARK1", d, 2),
         ("W_MARK2", d * d, d), ("B_MARK2", d, d), ("W_DELTA1", d * d, d), ("B_DELTA1", d, d),
         ("W_DELTA2", d, d), ("B_DELTA2", 1, d), ("W_EVT1", d * d, d), ("B_EVT1", d, d), ("W_EVT2", d, d),
         ("B_EVT2", 1, d), ("W_CROSS_IN", 3 * d * d, d), ("B_CROSS_IN", 3 * d, d), ("W_CROSS_OUT", d * d, d),
         ("B_CROSS_OUT", d, d), ("W_THR", N_EVT, 1)]
    for l in range(layers):
        p = f"E{l}_"
        a += [(f"W_{p}IN", 3 * d * d, d), (f"B_{p}IN", 3 * d, d), (f"W_{p}OUT", d * d, d), (f"B_{p}OUT", d, d),
              (f"W_{p}L1", ff * d, d), (f"B_{p}L1", ff, d), (f"W_{p}L2", d * ff, ff), (f"B_{p}L2", d, ff),
              (f"W_{p}N1", d, 0), (f"B_{p}N1", d, -1), (f"W_{p}N2", d, 0), (f"B_{p}N2", d, -1)]
    return a


def main():
    INC.mkdir(exist_ok=True)
    rng = np.random.default_rng(20261009)
    sel = ["#pragma once"]
    for i, (name, (d, layers, heads, n_cgm, max_marks)) in enumerate(CONFIGS.items()):
        lines = ["#pragma once", f"// R2-15 latency benchmark, config {name}: random weights (seed 20261009)",
                 f"#define BENCH_NAME \"{name}\"", f"#define BD {d}", f"#define BL {layers}", f"#define BH {heads}",
                 f"#define BN_CGM {n_cgm}", f"#define BN_MARK {max_marks}"]
        n_param = 0
        for arr, n, fan in arrays(d, layers, n_cgm):
            if fan == 0:
                v = np.ones(n)
            elif fan == -1:
                v = np.zeros(n)
            else:
                b = 1.0 / np.sqrt(fan)
                v = rng.uniform(-b, b, n)
            if not arr.startswith("W_THR"):
                n_param += n
            body = ",".join(f"{x:.6e}f" for x in v.astype(np.float32))
            lines.append(f"static const float {arr}[{n}] = {{{body}}};")
        enc = ",".join(f"{{W_E{l}_IN,B_E{l}_IN,W_E{l}_OUT,B_E{l}_OUT,W_E{l}_L1,B_E{l}_L1,W_E{l}_L2,B_E{l}_L2,"
                       f"W_E{l}_N1,B_E{l}_N1,W_E{l}_N2,B_E{l}_N2}}" for l in range(layers))
        lines.append(f"#define BENCH_ENC_INIT {{{enc}}}")
        lines.append(f"#define BENCH_N_PARAM {n_param}")
        (INC / f"bench_w_{name}.h").write_text("\n".join(lines) + "\n", encoding="utf-8")
        sel.append(f"{'#if' if i == 0 else '#elif'} defined(CFG_{name})\n#include \"bench_w_{name}.h\"")
        print(name, n_param)
    sel += ["#else", "#error \"define CFG_<config>\"", "#endif"]
    (INC / "bench_select.h").write_text("\n".join(sel) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
