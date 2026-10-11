#!/usr/bin/env python3
"""R2-8: actual placement of the seven working buffers and internal heap headroom in the original firmware
(v1.0.1, with diagnostic output added only).

Steps: flash code/firmware-v1.0.1-probe/, reset, and parse the PROBE lines in the boot output;
then replay the 512 stratified-sample windows of the original submission and compare bit by bit with the
original-submission board record (rice-s3/eval/validation/sample512.npz), confirming that the diagnostic code
does not change inference results. Results are written to code/results/r2_8_probe/.

Usage: python r2_8_sram_probe.py --port <port>
"""

from __future__ import annotations

import argparse
import json
import os
import re
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

ORIG = BASE / "validation/sample512.npz"
OUT = HERE / "results" / "r2_8_probe"
BUF = re.compile(r"PROBE buf (\S+) addr=(\S+) bytes=(\d+) internal=(\d) external=(\d)")
HEAP = re.compile(r"PROBE heap (\S+) int_free=(\d+) int_largest=(\d+) int_min=(\d+) spiram_free=(\d+) total_free=(\d+)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    model, thresholds = rv.load_model()
    test = rv.load_test()
    orig = np.load(ORIG)
    idx = np.asarray(orig["idx"], int)

    board = rv.attach(args.port, reset=True)
    banner = board.banner
    (OUT / "boot.log").write_text(banner, encoding="utf-8")
    bufs = [{"name": m[0], "addr": m[1], "bytes": int(m[2]), "internal": int(m[3]), "external": int(m[4])}
            for m in BUF.findall(banner)]
    heap = {m[0]: {"int_free": int(m[1]), "int_largest": int(m[2]), "int_min": int(m[3]),
                   "spiram_free": int(m[4]), "total_free": int(m[5])} for m in HEAP.findall(banner)}
    print(json.dumps({"bufs": bufs, "heap": heap}, indent=1), flush=True)
    try:
        summary, rows, pc, _, _ = rv.run_indexed(board, model, test, idx, thresholds, "probe")
    finally:
        board.close()

    lb = np.stack([r["logits"] for r in rows]).astype(np.float32)
    yb = np.stack([r["yhat"] for r in rows]).astype(np.float32)
    eb = np.stack([r["evt"] for r in rows]).astype(int)
    us = np.array([r["us_total"] for r in rows], dtype=np.float64)
    res = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"), "port": args.port,
        "bufs": bufs, "heap": heap,
        "all_internal": bool(bufs) and all(b["internal"] == 1 and b["external"] == 0 for b in bufs),
        "bytes_total": sum(b["bytes"] for b in bufs),
        "vs_original_board": {
            "n": int(len(idx)),
            "logits_bitwise_equal_windows": int((lb == orig["logits_board"]).all(axis=1).sum()),
            "yhat_bitwise_equal_windows": int((yb == orig["yhat_board"]).all(axis=1).sum()),
            "evt_equal_cells": int((eb == orig["evt_board"]).sum()),
            "logits_maxabs": float(np.max(np.abs(lb - orig["logits_board"]))),
            "yhat_maxabs": float(np.max(np.abs(yb - orig["yhat_board"]))),
        },
        "latency_ms": {"probe_median": float(np.median(us) / 1000), "orig_median": float(np.median(orig["us"]) / 1000),
                       "probe_p99": float(np.percentile(us, 99) / 1000), "orig_p99": float(np.percentile(orig["us"], 99) / 1000)},
        "summary_vs_host_mps32": {k: v for k, v in summary.items() if k != "idx"},
    }
    (OUT / "r2_8_probe.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    np.savez_compressed(OUT / "r2_8_probe.npz", idx=idx, logits_board=lb, yhat_board=yb, evt_board=eb, us=us)
    print(json.dumps({k: res[k] for k in ("all_internal", "bytes_total", "vs_original_board", "latency_ms")}, indent=1))


if __name__ == "__main__":
    main()
