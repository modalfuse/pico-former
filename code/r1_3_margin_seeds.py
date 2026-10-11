#!/usr/bin/env python3
"""R1-3 (also R2-13): decision margin |logit - tau| distributions of three independently trained models
on their own held-out sets.

Input: results/r2_4_r2_5/azt1d_s<seed>_scores.npz (host CPU float32 logits and frozen checkpoint thresholds).
Seed 20260828 is also compared with the R2-13 float64 result (results/margin/r2_13_margin.json).
Outputs: results/r1_3/margin_seeds.json, margin_seeds_table.md
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "r1_3"
OUT.mkdir(parents=True, exist_ok=True)
SEEDS = (20260828, 20260829, 20260830)
CUTS = [1e-5, 1e-4, 6.2e-4, 1.84e-3, 1e-2, 1e-1]
NAMES = [f"{e}@{h}h" for e in ("hypo70", "hypo54", "hyper180", "hyper250") for h in (1, 2, 3, 4)]

res = {}
for s in SEEDS:
    z = np.load(HERE / f"results/r2_4_r2_5/azt1d_s{s}_scores.npz")
    m = np.abs(z["logits"].astype(np.float64) - z["tau"][None, :].astype(np.float64))
    res[s] = {
        "n_windows": int(m.shape[0]), "n_cells": int(m.size),
        "tau": z["tau"].astype(float).tolist(),
        "min": float(m.min()), "median": float(np.median(m)),
        **{f"lt_{c:g}": int((m < c).sum()) for c in CUTS},
        "windows_any_lt_1.84e-3": int((m < 1.84e-3).any(1).sum()),
        "per_output_min": dict(zip(NAMES, m.min(0).astype(float).tolist())),
        "per_output_lt_1.84e-3": dict(zip(NAMES, (m < 1.84e-3).sum(0).astype(int).tolist())),
    }
ref = json.loads((HERE / "results/margin/r2_13_margin.json").read_text())["all"]
res["s20260828_float64_r2_13"] = {k: ref[k] for k in ref if k.startswith("lt_") or k in ("min", "median", "n_cells")}
(OUT / "margin_seeds.json").write_text(json.dumps(res, indent=1))

hdr = "| Model | Windows | Cells | Min margin | Median | " + " | ".join(f"<{c:g}" for c in CUTS) + " | Windows with a <1.84e-3 cell |"
lines = [hdr, "|" + " --- |" * (6 + len(CUTS))]
for s in SEEDS:
    r = res[s]
    lines.append(f"| {s} | {r['n_windows']:,} | {r['n_cells']:,} | {r['min']:.1e} | {r['median']:.2f} | "
                 + " | ".join(f"{r[f'lt_{c:g}']:,}" for c in CUTS) + f" | {r['windows_any_lt_1.84e-3']} |")
r = res["s20260828_float64_r2_13"]
lines.append(f"| 20260828（R2-13，float64） | — | {r['n_cells']:,} | {r['min']:.1e} | {r['median']:.2f} | "
             + " | ".join(f"{r[f'lt_{c:g}']:,}" for c in CUTS) + " | — |")
(OUT / "margin_seeds_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
for s in SEEDS:
    print(s, "tau hypo70@1h", round(res[s]["tau"][0], 3), "min per output",
          {k: f"{v:.1e}" for k, v in res[s]["per_output_min"].items() if v < 1e-4})
