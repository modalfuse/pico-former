#!/usr/bin/env python3
"""R2-14 step 3 (streamlit3.13 environment): comparison of FP32, FP32 (tanh GELU), INT16 and INT8 of the same checkpoint on the AZT1D held-out set.

Trajectory: yhat = inertial baseline + g·delta (g and the baseline computed in FP32 per the firmware rules); MAE 1 h averages h ≤ 60 min, MAE 4 h averages all 48 steps.
Events: two threshold treatments. Frozen: the 16 frozen FP32 thresholds; re-selected: thresholds re-selected on the validation split from the variant's logits per the release-package rule
(highest recall at FPR ≤ 0.06 among 37 quantile candidates, ceqt.metrics.calibrate_threshold);
frozen thresholds are stored in float32, and a float64 re-selected threshold within 1e-5 of it is treated as unchanged.
Decision agreement relative to the FP32 host decisions: window agreement A_w (all 16 bits identical) and cell agreement A_c.
Output: code/results/int8/int8_eval.json, int8_eval_table.md
"""

from __future__ import annotations

import os
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "results" / "int8"
sys.path.insert(0, os.environ.get("RICE_FORMER_SRC", str(Path(os.environ.get("PF_WORKSPACE", ".")) / "rice-former/src")))
from ceqt.metrics import calibrate_threshold  # noqa: E402

import torch  # noqa: E402

COLS = {"hypo70_4h": 3, "hyper180_4h": 11}


def rates(pred, lab):
    tp = int(((pred == 1) & (lab == 1)).sum()); fn = int(((pred == 0) & (lab == 1)).sum())
    fp = int(((pred == 1) & (lab == 0)).sum()); tn = int(((pred == 0) & (lab == 0)).sum())
    return fn / max(tp + fn, 1), fp / max(fp + tn, 1)


def main():
    blob = torch.load(RES / "state_s20260828.pt", map_location="cpu", weights_only=False)
    y_std, thr0 = float(blob["y_std"]), np.asarray(blob["thresholds"], np.float64)
    sv, st = np.load(RES / "side_val.npz"), np.load(RES / "side_test.npz")
    hz = st["horizons"].astype(float)
    ref_l, ref_y = st["logits_fp32"], st["yhat_fp32"]
    ref_evt = ref_l >= thr0[None, :]
    lab = st["labels"]
    variants = {"FP32": ({"logits": sv["logits_fp32"]}, {"logits": ref_l, "yhat": ref_y})}
    for name, tag in (("FP32-tanh", "fp32tanh"), ("INT16", "q16"), ("INT8", "q8"),
                      ("INT8+4×INT16", "qmix"), ("W8A16", "qw8a16")):
        fv, ft = RES / f"{tag}_val.npz", RES / f"{tag}_test.npz"
        if fv.exists() and ft.exists():
            v, t = np.load(fv), np.load(ft)
            if len(t["logits"]) == len(ref_l) and len(v["logits"]) == len(sv["logits_fp32"]):
                yh = st["iner"] + st["g"][:, None] * t["delta_norm"] * y_std
                variants[name] = ({"logits": v["logits"]}, {"logits": t["logits"], "yhat": yh})
    out, rows = {}, []
    for name, (v, t) in variants.items():
        err = np.abs(t["yhat"] - st["y_future"])
        thr_r = np.array([calibrate_threshold(v["logits"][:, k], sv["labels"][:, k]) for k in range(16)])
        r = {"n_test": int(len(ref_l)), "mae_1h": float(err[:, hz <= 60].mean()), "mae_4h": float(err.mean()),
             "rmse_4h": float(np.sqrt(((t["yhat"] - st["y_future"]) ** 2).mean())),
             "yhat_maxabs_vs_fp32": float(np.abs(t["yhat"] - ref_y).max()),
             "yhat_meanabs_vs_fp32": float(np.abs(t["yhat"] - ref_y).mean()),
             "logit_maxabs_vs_fp32": float(np.abs(t["logits"] - ref_l).max()),
             "logit_median_window_max_vs_fp32": float(np.median(np.abs(t["logits"] - ref_l).max(1))),
             "thr_reselected": thr_r.tolist(), "thr_shift_maxabs": float(np.abs(thr_r - thr0).max()),
             "thr_changed": int((np.abs(thr_r - thr0) > 1e-5).sum())}
        for mode, thr in (("retain", thr0), ("reselect", thr_r)):
            evt = t["logits"] >= thr[None, :]
            same = evt == ref_evt
            m = {"A_w": float(same.all(1).mean()), "A_c": float(same.mean()), "cells_changed": int((~same).sum()),
                 "windows_changed": int((~same.all(1)).sum())}
            for e, c in COLS.items():
                fnr, fpr = rates(evt[:, c], lab[:, c])
                m[f"{e}_fnr"], m[f"{e}_fpr"] = fnr, fpr
            r[mode] = m
        out[name] = r
        rows.append(f"| {name} | {r['mae_1h']:.2f} | {r['mae_4h']:.2f} | {r['yhat_maxabs_vs_fp32']:.3g} | "
                    f"{r['logit_maxabs_vs_fp32']:.3g} | {r['retain']['A_w']:.4f} | {r['retain']['A_c']:.5f} | "
                    f"{r['retain']['cells_changed']} | {r['retain']['hypo70_4h_fnr']:.3f}/{r['retain']['hypo70_4h_fpr']:.4f} | "
                    f"{r['thr_changed']} | {r['thr_shift_maxabs']:.3g} | {r['reselect']['A_w']:.4f} | "
                    f"{r['reselect']['cells_changed']} | {r['reselect']['hypo70_4h_fnr']:.3f}/{r['reselect']['hypo70_4h_fpr']:.4f} |")
    (RES / "int8_eval.json").write_text(json.dumps(out, indent=1))
    head = ("| Variant | MAE 1 h | MAE 4 h | Trajectory max diff | Logit max diff | Frozen A_w | Frozen A_c | Frozen: changed cells | Frozen hypo70 4 h FNR/FPR | "
            "Re-selected: thresholds changed | Max threshold shift | Re-selected A_w | Re-selected: changed cells | Re-selected hypo70 4 h FNR/FPR |\n"
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    (RES / "int8_eval_table.md").write_text(head + "\n" + "\n".join(rows) + "\n")
    print(head + "\n" + "\n".join(rows))


if __name__ == "__main__":
    main()
