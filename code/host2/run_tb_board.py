#!/usr/bin/env python3
"""R2-17 T-B on the board: replay the injected frames written by r2_17_tb_host.py and compare with the host guard.

Usage:
  python run_tb_board.py --port <port> [--n 512] [--only F_,M_] [--no-reset]
Results: code/results/r2_17/tb_board/{tb_board.json, tb_board.csv, <injection>.npz}
"""

from __future__ import annotations

import os
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pf2  # noqa: E402

FRAMES = HERE.parent / "results" / "r2_17" / "tb_frames"
OUT = HERE.parent / "results" / "r2_17" / "tb_board"
CKPT = pf2.BASE / "runs/20260830_s20260828/ckpt/ceqt.pt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True)
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--only", default="")
    ap.add_argument("--no-reset", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    _, _, _, thr = pf2.load_model(CKPT)
    files = sorted(FRAMES.glob("*.npz"))
    if args.only:
        files = [f for f in files if f.stem == "clean" or any(f.stem.startswith(p) for p in args.only.split(","))]
    board = pf2.attach(args.port, reset=not args.no_reset)
    (OUT / f"boot_{time.strftime('%H%M%S')}.log").write_text(board.banner, encoding="utf-8")
    rows, seq = [], 0
    try:
        for f in files:
            z = np.load(f)
            n = min(args.n, len(z["y"]))
            st = np.full(n, -1, np.int32)
            fl = np.full(n, -1, np.int32)
            nd = np.full(n, -1, np.int32)
            lg = np.full((n, 16), np.nan, np.float32)
            yh = np.full((n, 48), np.nan, np.float32)
            g = np.full(n, np.nan, np.float32)
            us = np.full(n, -1, np.int64)
            t_wall = time.time()
            for i in range(n):
                r = None
                for attempt in range(3):
                    seq += 1
                    try:
                        r = board.infer(seq, z["n_marks"][i], z["q"][i], z["flags"][i], z["t0_s"][i],
                                        z["t_last_s"][i], z["y"][i], z["feat"][i], z["mask"][i])
                        if r["status"] not in (pf2.BUSY, pf2.BAD_FRAME):
                            break
                    except TimeoutError as exc:
                        print(f"  timeout {f.stem} i={i} attempt={attempt}: {exc}", flush=True)
                        board.ser.reset_input_buffer()
                if r is None:
                    raise RuntimeError(f"{f.stem} window {i} failed")
                st[i], fl[i], nd[i], us[i] = r["status"], r["flags"], r["n_dropped"], r["us_total"]
                lg[i], yh[i], g[i] = r["logits"], r["yhat"], r["g"]
            es, ef = z["exp_status"][:n], z["exp_flags"][:n]
            el, ey = z["exp_logits"][:n].astype(np.float64), z["exp_yhat"][:n].astype(np.float64)
            both = (st == pf2.OK) & (es == pf2.OK)
            evt_b = lg >= thr[None, :]
            evt_h = el >= thr[None, :]
            row = {"injection": f.stem, "n": n, "status_agree": int((st == es).sum()),
                   "status_board": json.dumps({int(a): int(b) for a, b in zip(*np.unique(st, return_counts=True))}),
                   "flags_agree": int(((fl & 0b1011) == (ef & 0b1011)).sum()),
                   "clamped_board": int(((fl & pf2.FLAG_CLAMPED) > 0).sum()),
                   "dropped_board": int((nd > 0).sum()),
                   "evt_cells_compared": int(both.sum() * 16),
                   "evt_cells_disagree": int(((evt_b != evt_h) & both[:, None]).sum()),
                   "logit_maxabs": float(np.abs(lg[both] - el[both]).max()) if both.any() else None,
                   "yhat_maxabs": float(np.abs(yh[both] - ey[both]).max()) if both.any() else None,
                   "ms_median": float(np.median(us[st == pf2.OK]) / 1000) if (st == pf2.OK).any() else None,
                   "wall_s": round(time.time() - t_wall, 1)}
            rows.append(row)
            np.savez_compressed(OUT / f"{f.stem}.npz", status=st, flags=fl, n_dropped=nd, logits=lg, yhat=yh, g=g,
                                us_total=us)
            print(json.dumps(row), flush=True)
    finally:
        board.close()
    with (OUT / "tb_board.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summary = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "port": args.port, "n_per_injection": args.n,
               "injections": len(rows), "windows": int(sum(r["n"] for r in rows)),
               "status_agree": int(sum(r["status_agree"] for r in rows)),
               "flags_agree": int(sum(r["flags_agree"] for r in rows)),
               "evt_cells_compared": int(sum(r["evt_cells_compared"] for r in rows)),
               "evt_cells_disagree": int(sum(r["evt_cells_disagree"] for r in rows)),
               "logit_maxabs": max((r["logit_maxabs"] or 0) for r in rows)}
    (OUT / "tb_board.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1), flush=True)


if __name__ == "__main__":
    main()
