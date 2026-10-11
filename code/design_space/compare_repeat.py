#!/usr/bin/env python3
"""R2-15 / R2-21: differences between repeated MPS training runs with the same configuration and seed.

run1 is the existing grid run base_s20260828 (results/design_space/); run2 and run3 are produced by
train_grid.py --configs base --seeds 20260828 --out results/design_space_repeat/run{2,3}.
The three runs share code, data, seed and hyperparameters. Compared: maximum difference of each parameter tensor,
best epoch and validation loss, AZT1D held-out curve MAE, per-window prediction difference, and the number of
decision cells whose event output (16 columns) differs.

Usage: python compare_repeat.py
Output: code/results/design_space_repeat/compare.json
"""

from __future__ import annotations

import json
import pickle
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_grid import CACHE, OUT, Net, pack, predict  # noqa: E402

from ceqt.metrics import layer1_table  # noqa: E402
from ceqt.ops.gate import residual_gate  # noqa: E402

REP = OUT.parent / "design_space_repeat"
RUNS = {"run1": OUT, "run2": REP / "run2", "run3": REP / "run3"}
NAME, SEED = "base", 20260828


def main():
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    with CACHE.open("rb") as f:
        test = pickle.load(f)["test"]
    gate = residual_gate(test.n_marks, test.mark_a, test.mark_e)
    ck, out, res = {}, {}, {"runs": {}, "pairs": {}}
    for r, d in RUNS.items():
        ck[r] = torch.load(d / f"{NAME}_s{SEED}.pt", map_location="cpu", weights_only=False)
        meta = json.loads((d / f"{NAME}_s{SEED}.json").read_text(encoding="utf-8"))
        dm, layers, heads, n_cgm, max_marks = ck[r]["cfg"]
        net = Net(dm, layers, heads, n_cgm, max_marks)
        net.load_state_dict(ck[r]["state"])
        net.to(dev)
        yy, mm, kk, _, _, _, iner = pack(test, ck[r]["y_mean"], ck[r]["y_std"], n_cgm, max_marks)
        dd, lg = predict(net, dev, yy, mm, kk)
        yhat = iner + gate[:, None] * dd * ck[r]["y_std"]
        pred = lg >= np.asarray(ck[r]["thresholds"])[None, :]
        out[r] = (yhat.astype(np.float64), pred)
        tab = {int(x["ph"]) // 60: x["mae"] for x in layer1_table(test.y_future, yhat, test.horizons)}
        res["runs"][r] = {"best_epoch": meta["best_epoch"], "best_val": meta["best_val"], "train_s": meta["train_s"],
                          "traj_mae_1h": tab[1], "traj_mae_4h": tab[4],
                          "hypo70_4h_fnr": meta["hypo70_4h_fnr"], "hypo70_4h_fpr": meta["hypo70_4h_fpr"],
                          "thresholds": ck[r]["thresholds"]}
        print(r, res["runs"][r]["best_epoch"], f"{tab[1]:.3f} {tab[4]:.3f}", flush=True)
    for a, b in combinations(RUNS, 2):
        sa, sb = ck[a]["state"], ck[b]["state"]
        wd = {k: float((sa[k].float() - sb[k].float()).abs().max()) for k in sa}
        rel = {k: float((sa[k].float() - sb[k].float()).norm() / max(sa[k].float().norm(), 1e-12)) for k in sa}
        ya, pa = out[a]
        yb, pb = out[b]
        dy = np.abs(ya - yb)
        res["pairs"][f"{a}-{b}"] = {
            "weights_identical": all(v == 0.0 for v in wd.values()),
            "weight_maxabs": max(wd.values()), "weight_maxabs_tensor": max(wd, key=wd.get),
            "weight_rel_l2_median": float(np.median(list(rel.values()))),
            "yhat_maxabs": float(np.nanmax(dy)), "yhat_meanabs": float(np.nanmean(dy)),
            "event_cells_differ": int((pa != pb).sum()), "event_cells": int(pa.size),
            "d_traj_mae_1h": res["runs"][a]["traj_mae_1h"] - res["runs"][b]["traj_mae_1h"],
            "d_traj_mae_4h": res["runs"][a]["traj_mae_4h"] - res["runs"][b]["traj_mae_4h"],
            "threshold_maxabs": float(np.abs(np.subtract(ck[a]["thresholds"], ck[b]["thresholds"])).max()),
        }
        print(a, b, json.dumps(res["pairs"][f"{a}-{b}"]), flush=True)
    m = np.array([res["runs"][r]["traj_mae_1h"] for r in RUNS])
    m4 = np.array([res["runs"][r]["traj_mae_4h"] for r in RUNS])
    res["same_seed_sd"] = {"traj_mae_1h": float(m.std(ddof=1)), "traj_mae_4h": float(m4.std(ddof=1))}
    (REP / "compare.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
