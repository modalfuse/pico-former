#!/usr/bin/env python3
"""Short retest of the revised firmware: 512 stratified-sample windows + the 9 windows whose decisions flipped in
the original submission + windows with margin below 1.84e-3.

Usage: python run_short_retest.py --port <port> --tag wifiduino_erf [--reset]
Host reference: run_validation.predict (MPS float32, exact GELU); also compared with CPU float64 exact-GELU logits
(code/results/gelu/gelu_host_logits.npz). Results are written to code/results/short_retest/<tag>/.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BASE = Path(os.environ.get("PF_WORKSPACE", "."))
RUN = BASE / "runs/20260830_s20260828"
os.environ.setdefault("RICE_FORMER_SRC", str(BASE / "rice-former/src"))
os.environ.setdefault("RICE_CKPT", str(RUN / "ckpt/ceqt.pt"))
os.environ.setdefault("RICE_CACHE", str(RUN / "cache/azt1d_s20260828_str15.pkl"))
sys.path.insert(0, str(HERE / "pico-former" / "host"))

import run_validation as rv  # noqa: E402

GELU = HERE / "results" / "gelu"
MARGIN_MAX = 1.84e-3


def target_idx(test) -> dict:
    z = np.load(GELU / "gelu_host_logits.npz")
    margin = np.abs(z["ref_cpu64"] - z["thresholds"][None, :])
    near = np.where((margin < MARGIN_MAX).any(axis=1))[0]
    flips = sorted({f["idx"] for f in json.loads((GELU / "gelu_host_sim.json").read_text())["flips_vs_ref"]})
    sample = np.load(BASE / "validation/sample512.npz")["idx"]
    return {"sample512": np.asarray(sample, int), "flips9": np.asarray(flips, int), "near": np.asarray(near, int),
            "ref64": z["ref_cpu64"], "tanh32": z["tanh_cpu32"], "thr": z["thresholds"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--reset", action="store_true")
    ap.add_argument("--n-burn", type=int, default=80)
    args = ap.parse_args()
    out = HERE / "results" / "short_retest" / args.tag
    out.mkdir(parents=True, exist_ok=True)

    model, thresholds = rv.load_model()
    test = rv.load_test()
    t = target_idx(test)
    idx = np.array(sorted(set(t["sample512"]) | set(t["flips9"]) | set(t["near"])), dtype=int)
    print(f"windows: sample512={len(t['sample512'])} flips9={len(t['flips9'])} near={len(t['near'])} union={len(idx)}",
          flush=True)

    board = rv.attach(args.port, reset=args.reset)
    banner = board.banner
    try:
        summary, rows, pc, _, _ = rv.run_indexed(board, model, test, idx, thresholds, args.tag)
        burn = board.burn(args.n_burn)
    finally:
        board.close()

    lb = np.stack([r["logits"] for r in rows]).astype(np.float64)
    eb = np.stack([r["evt"] for r in rows]).astype(int)
    yb = np.stack([r["yhat"] for r in rows]).astype(np.float64)
    us = np.array([r["us_total"] for r in rows], dtype=np.float64) / 1000.0
    ref64 = t["ref64"][idx]
    thr = t["thr"]
    e64 = (ref64 >= thr[None, :]).astype(int)
    etanh = (t["tanh32"][idx] >= thr[None, :]).astype(int)
    pos = {int(i): k for k, i in enumerate(idx)}

    def sub(name):
        k = np.array([pos[int(i)] for i in t[name]], dtype=int)
        d = eb[k] != e64[k]
        return {"windows": int(len(k)), "cells_disagree_vs_host64": int(d.sum()),
                "windows_disagree_vs_host64": int(d.any(axis=1).sum()),
                "cells_disagree_vs_tanh_host": int((eb[k] != etanh[k]).sum()),
                "logit_maxabs_vs_host64": float(np.max(np.abs(lb[k] - ref64[k])))}

    res = {
        "tag": args.tag, "port": args.port, "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "banner_tail": banner[-600:],
        "n_windows": int(len(idx)),
        "summary_vs_host_mps32": {k: v for k, v in summary.items() if k != "idx"},
        "vs_host64": {"all": {"cells_disagree": int((eb != e64).sum()),
                              "logit_maxabs": float(np.max(np.abs(lb - ref64))),
                              "logit_maxabs_median": float(np.median(np.max(np.abs(lb - ref64), axis=1)))},
                      "sample512": sub("sample512"), "flips9": sub("flips9"), "near": sub("near")},
        "latency_ms": {"median": float(np.median(us)), "p95": float(np.percentile(us, 95)),
                       "p99": float(np.percentile(us, 99)), "max": float(us.max())},
        "burn": {k: v for k, v in burn.items() if k != "raw"},
    }
    (out / "short_retest.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    np.savez_compressed(out / "short_retest.npz", idx=idx, logits_board=lb, evt_board=eb, yhat_board=yb, us_ms=us,
                        logits_host_mps32=pc["logits"], yhat_host_mps32=pc["yhat"])
    print(json.dumps({k: res[k] for k in ("n_windows", "vs_host64", "latency_ms", "burn")}, indent=1))
    print("evt_mismatch_vs_mps32", summary["evt_mismatch"], "logit_max_vs_mps32", summary["logit_maxabs_max"])


if __name__ == "__main__":
    main()
