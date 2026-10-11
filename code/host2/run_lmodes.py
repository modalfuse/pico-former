#!/usr/bin/env python3
"""R2-11 L1 / L2: first-inference latency under cold start and concurrent Wi-Fi (boot test mode of the frozen firmware).

The firmware reads lmode from NVS: 1 = Wi-Fi off, 20 back-to-back inferences on the golden window after power-up; 2 = Wi-Fi connection and 20 inferences started concurrently.
Each mode is booted --reps times via EN-pin reset (RTS); each boot records the LRUN line (total and per-stage latency) and the reset reason.
On exit (including abnormal exit), lmode is written back to 0 and the reset counter is cleared.

Usage: python run_lmodes.py --port <port> [--reps 5]
Results: code/results/latency/lmodes.json, lmodes_boot.log
"""

from __future__ import annotations

import os
import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pf2  # noqa: E402

OUT = HERE.parent / "results" / "latency"
LRUN = re.compile(r"LRUN mode=(\d) i=(\d+) us_total=(\d+) us_enc=(\d+) us_cross=(\d+) us_heads=(\d+) wifi=(-?\d+)")
RESET = re.compile(r"RESET reason=(\d+) count=(\d+) lmode=(\d+)")


def set_mode(port, mode):
    b = pf2.attach(port, reset=True, timeout=60)
    b.debug(7, mode, expect=f"DBG lmode={mode}")
    b.close()
    time.sleep(1.0)


def boot(port):
    b = pf2.Board(port, reset=True)
    text = b.wait_ready(90)
    b.close()
    return text


def summarise(runs):
    tot = np.array([[x["us_total"] for x in r["lrun"]] for r in runs], float) / 1000
    out = {"n_boots": len(runs), "first_ms": tot[:, 0].tolist(), "first_median_ms": float(np.median(tot[:, 0])),
           "warm_median_ms": float(np.median(tot[:, 1:])), "warm_p99_ms": float(np.percentile(tot[:, 1:], 99)),
           "first_minus_warm_ms": float(np.median(tot[:, 0]) - np.median(tot[:, 1:]))}
    for k in ("us_enc", "us_cross", "us_heads"):
        a = np.array([[x[k] for x in r["lrun"]] for r in runs], float) / 1000
        out[f"{k[3:]}_first_median_ms"] = float(np.median(a[:, 0]))
        out[f"{k[3:]}_warm_median_ms"] = float(np.median(a[:, 1:]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--modes", default="1,2")
    ap.add_argument("--tag", default="", help="if non-empty, write results to results/latency/<tag>/")
    args = ap.parse_args()
    out = OUT / args.tag if args.tag else OUT
    out.mkdir(parents=True, exist_ok=True)
    log = (out / "lmodes_boot.log").open("a", encoding="utf-8")
    res = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "port": args.port, "modes": {}}
    try:
        for mode in (int(m) for m in args.modes.split(",")):
            set_mode(args.port, mode)
            runs = []
            for rep in range(args.reps):
                text = boot(args.port)
                log.write(f"== mode {mode} rep {rep} {time.strftime('%F %T')}\n{text}\n")
                log.flush()
                lr = [dict(zip(("mode", "i", "us_total", "us_enc", "us_cross", "us_heads", "wifi"), map(int, m)))
                      for m in LRUN.findall(text)]
                rs = RESET.search(text)
                runs.append({"rep": rep, "reset_reason": int(rs.group(1)) if rs else None,
                             "lmode_seen": int(rs.group(3)) if rs else None, "lrun": lr})
                print(f"mode {mode} rep {rep}: n={len(lr)} first={lr[0]['us_total'] / 1000 if lr else None} ms",
                      flush=True)
            ok = [r for r in runs if len(r["lrun"]) == 20]
            res["modes"][str(mode)] = {"runs": runs, **(summarise(ok) if ok else {})}
            (out / "lmodes.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    finally:
        set_mode(args.port, 0)
        b = pf2.attach(args.port, reset=True, timeout=60)
        b.debug(8, 0, expect="")
        st = b.status()
        b.close()
        res["lmode_after"] = st.get("lmode")
        (out / "lmodes.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
        print("lmode after:", st.get("lmode"), flush=True)
    for m, v in res["modes"].items():
        print(m, {k: v[k] for k in v if k != "runs"}, flush=True)


if __name__ == "__main__":
    main()
