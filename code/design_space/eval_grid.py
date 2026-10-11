#!/usr/bin/env python3
"""R2-15 design space: recompute held-out curve errors with the paper's metric and summarise 15 configurations x 3 seeds.

Fields such as mae_1h recorded by train_grid.py use only the last point of the prediction horizon; Table 3 of the
paper and the RICE-Former release code (ceqt.metrics.layer1_table) use all trajectory points from 5 min to the
horizon. This script loads the best weights saved by each run, recomputes 1-4 h MAE and RMSE with layer1_table on
the same held-out set, writes them back to the traj_* fields of <config>_s<seed>.json, and outputs summary tables.

Usage: python eval_grid.py
Output: code/results/design_space/grid_summary.csv, grid_summary.json, grid_table.md
"""

from __future__ import annotations

import csv
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_grid import CACHE, CONFIGS, OUT, Net, pack, predict  # noqa: E402

from ceqt.metrics import layer1_table  # noqa: E402
from ceqt.ops.gate import residual_gate  # noqa: E402

SEEDS = (20260828, 20260829, 20260830)
ORDER = ["d32_l1", "d32_l2", "d32_l3", "d48_l1", "base", "d48_l3", "d64_l1", "d64_l2", "d64_l3", "d96_l2",
         "L48", "L60", "L84", "h2", "h8"]
EVT = ["hypo70_4h_fnr", "hypo70_4h_fpr", "hyper180_4h_fnr", "hyper180_4h_fpr", "hypo70_1h_fnr", "hyper180_1h_fnr"]


def traj_metrics(name, seed, test, dev):
    ck = torch.load(OUT / f"{name}_s{seed}.pt", map_location="cpu", weights_only=False)
    d, layers, heads, n_cgm, max_marks = ck["cfg"]
    net = Net(d, layers, heads, n_cgm, max_marks)
    net.load_state_dict(ck["state"])
    net.to(dev)
    yy, mm, kk, _, _, _, iner = pack(test, ck["y_mean"], ck["y_std"], n_cgm, max_marks)
    dd, _ = predict(net, dev, yy, mm, kk)
    gate = residual_gate(test.n_marks, test.mark_a, test.mark_e)
    yhat = iner + gate[:, None] * dd * ck["y_std"]
    out = {}
    for row in layer1_table(test.y_future, yhat, test.horizons):
        h = int(row["ph"]) // 60
        out[f"traj_mae_{h}h"] = row["mae"]
        out[f"traj_rmse_{h}h"] = row["rmse"]
    return out


def main():
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    with CACHE.open("rb") as f:
        test = pickle.load(f)["test"]
    runs = []
    for name in ORDER:
        for seed in SEEDS:
            p = OUT / f"{name}_s{seed}.json"
            r = json.loads(p.read_text(encoding="utf-8"))
            if "traj_mae_1h" not in r:
                r.update(traj_metrics(name, seed, test, dev))
                p.write_text(json.dumps(r, indent=1), encoding="utf-8")
            runs.append(r)
            print(f"{name} s{seed} traj_mae_1h={r['traj_mae_1h']:.2f} traj_mae_4h={r['traj_mae_4h']:.2f}", flush=True)

    keys = ["traj_mae_1h", "traj_rmse_1h", "traj_mae_4h", "traj_rmse_4h", "mae_1h", "mae_4h"] + EVT + \
        ["best_epoch", "train_s"]
    rows = []
    for name in ORDER:
        rs = [r for r in runs if r["config"] == name]
        r0 = rs[0]
        row = {k: r0[k] for k in ("config", "d", "layers", "heads", "n_cgm", "max_marks", "L", "act_bytes", "params",
                                   "weight_bytes_fp32")}
        row["n_seeds"] = len(rs)
        for k in keys:
            v = np.array([r[k] for r in rs], dtype=float)
            row[f"{k}_mean"] = float(v.mean())
            row[f"{k}_sd"] = float(v.std(ddof=1)) if len(v) > 1 else 0.0
        rows.append(row)

    with (OUT / "grid_summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (OUT / "grid_summary.json").write_text(json.dumps({"rows": rows, "runs": runs}, indent=1), encoding="utf-8")

    def ms(row, k, f="{:.2f}"):
        return f"{f.format(row[k + '_mean'])} ± {f.format(row[k + '_sd'])}"

    lines = ["| Configuration | d | Layers | Heads | L | Activation memory (KB) | Parameters | 1 h MAE | 4 h MAE | 4 h hypo70 FNR | "
             "4 h hyper180 FNR |", "|" + " --- |" * 11]
    for row in rows:
        lines.append(f"| {row['config']} | {row['d']} | {row['layers']} | {row['heads']} | {row['L']} | "
                     f"{row['act_bytes'] / 1024:.1f} | {row['params']:,} | {ms(row, 'traj_mae_1h')} | "
                     f"{ms(row, 'traj_mae_4h')} | {ms(row, 'hypo70_4h_fnr', '{:.3f}')} | "
                     f"{ms(row, 'hyper180_4h_fnr', '{:.3f}')} |")
    (OUT / "grid_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
