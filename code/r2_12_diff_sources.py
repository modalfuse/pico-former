#!/usr/bin/env python3
"""R2-12: decomposition of the difference between board logits of the revised firmware (exact GELU)
and host logits across precisions and backends.

Deployed AZT1D model, full held-out set: 19,152 windows x 16 outputs. Host reference and board results are
taken from code/results/audit/azt1d_full/audit.npz (float64 reference, frozen firmware 10cba768...).
This script reuses the input checks of host2/run_audit.py (pf2.guard) and, on identical inputs, additionally
computes two host forward passes, CPU float32 and MPS float32. On the 512-window sample, the default build and
the builds with fused multiply-add disabled, with -O2, and without stage timers are taken from
code/results/audit/l4_{n16r8,nofma,o2,notiming}_sample512/.
Results are written to code/results/r2_12/diff_sources.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "host2"))
import pf2  # noqa: E402

RES = HERE / "results"
RUN = pf2.BASE / "runs/20260830_s20260828"


def stats(a: np.ndarray, b: np.ndarray, thr: np.ndarray) -> dict:
    d = np.abs(a.astype(np.float64) - b.astype(np.float64))
    return {"max": float(d.max()), "median": float(np.median(d)), "p99": float(np.percentile(d, 99)),
            "bitwise_equal_cells_fp32": int((a.astype(np.float32) == b.astype(np.float32)).sum()),
            "cells": int(d.size), "decision_cells_differ": int(((a >= thr) != (b >= thr)).sum())}


def guarded_inputs(idx: np.ndarray):
    batch = pf2.load_cache(RUN / "cache/azt1d_s20260828_str15.pkl")["test"]
    fr = pf2.frames(batch)
    n = len(idx)
    st = np.zeros(n, np.int32)
    g_y, g_feat, g_mask = fr["y"][idx].copy(), fr["feat"][idx].copy(), fr["mask"][idx].copy()
    g_q, g_flags = fr["q"][idx].copy(), fr["flags"][idx].copy()
    for k, i in enumerate(idx):
        s, yc, f, m, q, fl, _ = pf2.guard(fr["y"][i], fr["feat"][i], fr["mask"][i], fr["q"][i], fr["flags"][i],
                                          fr["t0_s"][i], fr["t_last_s"][i])
        st[k], g_y[k], g_feat[k], g_mask[k], g_q[k], g_flags[k] = s, yc, f, m, q, fl
    clamped = (g_flags & pf2.FLAG_CLAMPED) > 0
    g_iner = np.where(clamped[:, None], g_y[:, -1:].astype(np.float64), fr["iner"][idx])
    return st == pf2.OK, g_y, g_feat, g_mask, fr["n_marks"][idx], g_q, g_iner


def main() -> None:
    a = np.load(RES / "audit/azt1d_full/audit.npz")
    idx = np.asarray(a["idx"], int)
    net, y_mean, y_std, thr = pf2.load_model(RUN / "ckpt/ceqt.pt")
    ok, y, feat, mask, nm, q, iner = guarded_inputs(idx)
    assert ok.all()
    l64, _, _ = pf2.host_forward(net, y_mean, y_std, y, feat, mask, nm, q, iner)
    d_ref = float(np.abs(l64 - a["ref_logits"]).max())
    print(f"float64 rerun vs audit reference: max {d_ref:.1e}", flush=True)
    assert d_ref < 1e-12, "float64 reference does not match the audit record"
    l32, _, _ = pf2.host_forward(net, y_mean, y_std, y, feat, mask, nm, q, iner, dtype=torch.float32)
    res = {"n_windows": int(len(idx)), "torch": torch.__version__, "float64_rerun_vs_audit_max": d_ref}
    lm = None
    if torch.backends.mps.is_available():
        lm, _, _ = pf2.host_forward(net, y_mean, y_std, y, feat, mask, nm, q, iner, device="mps",
                                    dtype=torch.float32)
    board = a["logits"]
    pairs = {"board_vs_cpu64": (board, l64), "board_vs_cpu32": (board, l32), "cpu32_vs_cpu64": (l32, l64)}
    if lm is not None:
        pairs.update({"board_vs_mps32": (board, lm), "mps32_vs_cpu64": (lm, l64), "cpu32_vs_mps32": (l32, lm)})
    for k, (u, v) in pairs.items():
        res[k] = stats(u, v, thr)
    f = np.load(RES / "audit/l4_n16r8_sample512/audit.npz")
    builds = {k: np.load(RES / f"audit/l4_{k}_sample512/audit.npz") for k in ("nofma", "o2", "notiming")}
    assert all((b["idx"] == f["idx"]).all() for b in builds.values())
    n = builds["nofma"]
    res["sample512"] = {"nofma_vs_fma": stats(n["logits"], f["logits"], thr),
                        "fma_vs_cpu64": stats(f["logits"], f["ref_logits"], thr),
                        "nofma_vs_cpu64": stats(n["logits"], n["ref_logits"], thr),
                        "o2_vs_default": stats(builds["o2"]["logits"], f["logits"], thr),
                        "notiming_vs_default": stats(builds["notiming"]["logits"], f["logits"], thr)}
    for k, b in builds.items():
        res["sample512"][f"{k}_curve_maxabs_vs_default"] = float(np.abs(b["yhat"].astype(np.float64) - f["yhat"]).max())
        res["sample512"][f"{k}_median_us"] = float(np.median(b["us_total"]))
    res["sample512"]["default_median_us"] = float(np.median(f["us_total"]))
    out = RES / "r2_12"
    out.mkdir(exist_ok=True)
    (out / "diff_sources.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    for k in pairs:
        v = res[k]
        print(f"{k:16s} max={v['max']:.2e} median={v['median']:.1e} p99={v['p99']:.1e} "
              f"equal={v['bitwise_equal_cells_fp32']}/{v['cells']} flips={v['decision_cells_differ']}")
    for k, v in res["sample512"].items():
        if not isinstance(v, dict):
            print(f"512 {k:30s} {v}")
            continue
        print(f"512 {k:14s} max={v['max']:.2e} median={v['median']:.1e} "
              f"equal={v['bitwise_equal_cells_fp32']}/{v['cells']} flips={v['decision_cells_differ']}")


if __name__ == "__main__":
    main()
