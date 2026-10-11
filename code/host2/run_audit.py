#!/usr/bin/env python3
"""Board audit with the revised firmware: replay packed windows over PFW2 and compare with the host reference.

The board receives the windows as packed by the host (no host-side guard applied); the host computes the expected
status with the same rules (pf2.guard) and the reference outputs in float64 on the guarded tensors.

Usage:
  python run_audit.py --port <port> --tag azt1d_full --split test [--idx sample512|all|<npy>]
      [--cache <pkl>] [--ckpt <pt>] [--no-reset]
Results: code/results/audit/<tag>/{audit.json, audit.npz, ckpt.npz, run.log}
"""

from __future__ import annotations

import os
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pf2  # noqa: E402

RUN = pf2.BASE / "runs/20260830_s20260828"
SAMPLE512 = pf2.BASE / "validation/sample512.npz"
OUT = HERE.parent / "results" / "audit"


def pick(idx_arg: str, n: int) -> np.ndarray:
    if idx_arg == "all":
        return np.arange(n)
    if idx_arg == "sample512":
        return np.asarray(np.load(SAMPLE512)["idx"], int)
    if idx_arg.startswith("first"):
        return np.arange(min(n, int(idx_arg[5:])))
    return np.asarray(np.load(idx_arg), int)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--cache", default=str(RUN / "cache/azt1d_s20260828_str15.pkl"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--ckpt", default=str(RUN / "ckpt/ceqt.pt"))
    ap.add_argument("--idx", default="all")
    ap.add_argument("--no-reset", action="store_true")
    ap.add_argument("--save-every", type=int, default=256)
    args = ap.parse_args()
    out = OUT / args.tag
    out.mkdir(parents=True, exist_ok=True)

    batch = pf2.load_cache(Path(args.cache))[args.split]
    fr = pf2.frames(batch)
    idx = pick(args.idx, len(fr["y"]))
    n = len(idx)
    net, y_mean, y_std, thr = pf2.load_model(Path(args.ckpt))

    # host expectation (guard + float64 reference) for the selected windows
    exp_status = np.zeros(n, np.int32)
    g_feat = fr["feat"][idx].copy()
    g_mask = fr["mask"][idx].copy()
    g_q = fr["q"][idx].copy()
    g_flags = fr["flags"][idx].copy()
    g_drop = np.zeros(n, np.int32)
    g_y = fr["y"][idx].copy()
    for k, i in enumerate(idx):
        st, yc, f, m, q, fl, nd = pf2.guard(fr["y"][i], fr["feat"][i], fr["mask"][i], fr["q"][i], fr["flags"][i],
                                            fr["t0_s"][i], fr["t_last_s"][i])
        exp_status[k], g_y[k], g_feat[k], g_mask[k], g_q[k], g_flags[k], g_drop[k] = st, yc, f, m, q, fl, nd
    ok_h = exp_status == pf2.OK
    clamped = (g_flags & pf2.FLAG_CLAMPED) > 0
    # the inertia baseline is the last CGM value, so it follows the clamped series
    g_iner = np.where(clamped[:, None], g_y[:, -1:].astype(np.float64), fr["iner"][idx])
    ref_l = np.full((n, 16), np.nan)
    ref_y = np.full((n, 48), np.nan)
    if ok_h.any():
        l, y, _ = pf2.host_forward(net, y_mean, y_std, g_y[ok_h], g_feat[ok_h], g_mask[ok_h],
                                   fr["n_marks"][idx][ok_h], g_q[ok_h], g_iner[ok_h])
        ref_l[ok_h], ref_y[ok_h] = l, y
    print(f"windows {n}; host status counts {dict(zip(*np.unique(exp_status, return_counts=True)))}; "
          f"marks dropped in {int((g_drop > 0).sum())} windows; CGM clamped in {int(clamped.sum())} windows",
          flush=True)

    ck = out / "ckpt.npz"
    keys_i = ("status", "flags", "n_dropped", "core", "us_total", "us_enc", "us_cross", "us_heads", "int_free",
              "int_min", "int_largest", "spiram_free", "stack_infer", "stack_serial", "rssi", "link", "reconnects",
              "http_req", "http_fail", "https_req", "https_fail", "reset_reason", "reset_count")
    rec = {k: np.full(n, -1, np.int64) for k in keys_i}
    rec["evt"] = np.full((n, 16), -1, np.int32)
    rec["logits"] = np.full((n, 16), np.nan, np.float32)
    rec["yhat"] = np.full((n, 48), np.nan, np.float32)
    rec["g"] = np.full(n, np.nan, np.float32)
    start = 0
    if ck.exists():
        z = np.load(ck)
        if len(z["status"]) == n:
            start = int(z["done"])
            for k in rec:
                rec[k] = z[k]
            print(f"resume at {start}", flush=True)

    board = pf2.attach(args.port, reset=not args.no_reset)
    (out / f"boot_{time.strftime('%H%M%S')}.log").write_text(board.banner, encoding="utf-8")
    print(board.banner[-1500:], flush=True)
    t_wall = time.time()
    n_retry = 0
    try:
        for k in range(start, n):
            i = int(idx[k])
            r = None
            for attempt in range(3):
                try:
                    r = board.infer(k + 1, fr["n_marks"][i], fr["q"][i], fr["flags"][i], fr["t0_s"][i],
                                    fr["t_last_s"][i], fr["y"][i], fr["feat"][i], fr["mask"][i])
                    if r["status"] not in (pf2.BUSY, pf2.BAD_FRAME):
                        break
                except TimeoutError as exc:
                    print(f"  timeout k={k} attempt={attempt}: {exc}", flush=True)
                    n_retry += 1
                    with (out / "timeouts.log").open("a") as fh:
                        fh.write(f"{time.strftime('%H:%M:%S')} k={k} idx={i} {exc}\n"
                                 f"  raw={getattr(board, 'last_raw', b'').hex()}\n")
                    board.ser.reset_input_buffer()
            if r is None:
                raise RuntimeError(f"window k={k} idx={i} failed")
            for key in keys_i:
                rec[key][k] = r[key]
            rec["evt"][k], rec["logits"][k], rec["yhat"][k], rec["g"][k] = r["evt"], r["logits"], r["yhat"], r["g"]
            if (k + 1) % 32 == 0 or k + 1 == n:
                print(f"  {args.tag} {k + 1}/{n}  {r['us_total'] / 1000:.1f} ms  int_min={r['int_min']}", flush=True)
            if (k + 1) % args.save_every == 0 or k + 1 == n:
                np.savez_compressed(ck, done=k + 1, **rec)
    finally:
        tail = board.text[-4000:]
        board.close()
        (out / "board_text.log").write_text(tail, encoding="utf-8")

    st = rec["status"]
    ok_b = st == pf2.OK
    both = ok_b & ok_h
    evt_h = (ref_l >= thr[None, :]).astype(np.int32)
    dis = (rec["evt"] != evt_h) & both[:, None]
    dl = np.abs(rec["logits"].astype(np.float64) - ref_l)
    dy = np.abs(rec["yhat"].astype(np.float64) - ref_y)
    us = rec["us_total"][ok_b] / 1000.0
    lab = pf2.labels(batch)[idx]
    cells = np.argwhere(dis)
    res = {
        "tag": args.tag, "time": time.strftime("%Y-%m-%d %H:%M:%S"), "port": args.port, "cache": args.cache,
        "split": args.split, "ckpt": args.ckpt, "idx": args.idx, "n": n, "wall_s": time.time() - t_wall,
        "serial_retries": n_retry,
        "status_host": {int(a): int(b) for a, b in zip(*np.unique(exp_status, return_counts=True))},
        "status_board": {int(a): int(b) for a, b in zip(*np.unique(st, return_counts=True))},
        "status_agree": int((st == exp_status).sum()),
        "windows_marks_dropped_host": int((g_drop > 0).sum()),
        "windows_marks_dropped_board": int((rec["n_dropped"] > 0).sum()),
        "windows_cgm_clamped_host": int(clamped.sum()),
        "windows_cgm_clamped_board": int(((rec["flags"] & pf2.FLAG_CLAMPED) > 0).sum()),
        "flags_agree": int(((rec["flags"] & 0b1011) == (g_flags & 0b1011)).sum()),
        "evt_cells_compared": int(both.sum() * 16), "evt_cells_disagree": int(dis.sum()),
        "evt_windows_disagree": int(dis.any(axis=1).sum()),
        "disagree_cells": [{"k": int(a), "idx": int(idx[a]), "out": int(b), "logit_board": float(rec["logits"][a, b]),
                            "logit_host": float(ref_l[a, b]), "thr": float(thr[b])} for a, b in cells[:50]],
        "logit_maxabs": float(np.nanmax(dl[both])) if both.any() else None,
        "logit_maxabs_median": float(np.nanmedian(np.nanmax(dl[both], axis=1))) if both.any() else None,
        "yhat_maxabs": float(np.nanmax(dy[both])) if both.any() else None,
        "g_maxabs": float(np.nanmax(np.abs(rec["g"][both] - np.where(fr["n_marks"][idx][both] > 0,
                                                                      np.where(g_q[both] > 0, 1, .45), 0))))
        if both.any() else None,
        "latency_ms": {"n": int(us.size), "median": float(np.median(us)), "p95": float(np.percentile(us, 95)),
                       "p99": float(np.percentile(us, 99)), "max": float(us.max()), "min": float(us.min()),
                       "first": float(us[0])} if us.size else None,
        "mem": {"int_free_min": int(rec["int_free"].min()), "int_min_min": int(rec["int_min"].min()),
                "int_largest_min": int(rec["int_largest"].min()), "spiram_free_min": int(rec["spiram_free"].min()),
                "stack_infer_hw_min": int(rec["stack_infer"].min()), "stack_serial_hw_min": int(rec["stack_serial"].min())},
        "cores": sorted({int(c) for c in rec["core"][ok_b]}),
        "reset_count_end": int(rec["reset_count"][-1]), "reconnects_end": int(rec["reconnects"][-1]),
    }
    for e, col in (("hypo70_4h", 3), ("hyper180_4h", 11)):
        for who, ev in (("board", rec["evt"]), ("host", evt_h)):
            y_, p_ = lab[both, col], ev[both, col]
            tp = int(((p_ == 1) & (y_ == 1)).sum()); fn = int(((p_ == 0) & (y_ == 1)).sum())
            fp = int(((p_ == 1) & (y_ == 0)).sum()); tn = int(((p_ == 0) & (y_ == 0)).sum())
            res[f"{e}_{who}"] = {"tp": tp, "fn": fn, "fp": fp, "tn": tn, "fnr": fn / max(tp + fn, 1),
                                 "fpr": fp / max(fp + tn, 1)}
    (out / "audit.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    np.savez_compressed(out / "audit.npz", idx=idx, exp_status=exp_status, ref_logits=ref_l, ref_yhat=ref_y,
                        thr=thr, g_drop=g_drop, **rec)
    print(json.dumps({k: res[k] for k in ("n", "status_agree", "evt_cells_compared", "evt_cells_disagree",
                                          "logit_maxabs", "latency_ms", "mem", "cores")}, indent=1), flush=True)


if __name__ == "__main__":
    main()
