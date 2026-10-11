#!/usr/bin/env python3
"""Request failures and latency in the wireless load test, binned by node RSSI (R2-10; runs that push windows, such as T3).

Each window result frame carries the node RSSI and the host time t_s (same origin as the request log). Each request takes the RSSI of the
latest window before it was sent; per bin, request count, failure count, failure rate and latency quantiles are reported, as well as inference latency.
Usage: python wireless_rssi_bins.py [--tag t3_c4]
Results: code/results/wireless/<tag>/rssi_bins.json
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent.parent / "results" / "wireless"
EDGES = (-200, -70, -65, -60, 0)  # bins: ≤ −71, −70 to −66, −65 to −61, ≥ −60 dBm
LABELS = ("<= -71", "-70 to -66", "-65 to -61", ">= -60")


def pct(a, q):
    return float(np.percentile(a, q)) if len(a) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="t3_c4")
    args = ap.parse_args()
    d = OUT / args.tag
    w = np.load(d / "windows.npz")
    n = int(w["done"])
    wt, wr, wus = w["t_s"][:n], w["rssi"][:n].astype(float), w["us_total"][:n] / 1000
    order = np.argsort(wt)
    wt, wr, wus = wt[order], wr[order], wus[order]

    t, lat, ok = [], [], []
    with (d / "requests.csv").open() as f:
        for r in csv.DictReader(f):
            t.append(float(r["t_s"]))
            lat.append(float(r["latency_ms"]) if r["latency_ms"] else np.nan)
            ok.append(r["outcome"] == "ok")
    t, lat, ok = np.array(t), np.array(lat), np.array(ok)
    j = np.searchsorted(wt, t, side="right") - 1
    keep = j >= 0
    rr = np.where(keep, wr[np.clip(j, 0, None)], np.nan)

    bins = []
    for lo, hi, lab in zip(EDGES[:-1], EDGES[1:], LABELS):
        m = keep & (rr > lo) & (rr <= hi)
        mw = (wr > lo) & (wr <= hi)
        good = m & ok
        bins.append({
            "rssi_dbm": lab,
            "requests": int(m.sum()), "failed": int((m & ~ok).sum()),
            "fail_rate": float((m & ~ok).sum() / m.sum()) if m.sum() else None,
            "latency_ms": {"p50": pct(lat[good], 50), "p95": pct(lat[good], 95), "p99": pct(lat[good], 99),
                           "max": float(lat[good].max()) if good.any() else None},
            "windows": int(mw.sum()),
            "infer_ms": {"median": pct(wus[mw], 50), "p99": pct(wus[mw], 99),
                         "max": float(wus[mw].max()) if mw.any() else None},
        })
    res = {"tag": args.tag, "requests_before_first_window": int((~keep).sum()),
           "rssi_range_dbm": [float(wr.min()), float(wr.max())], "bins": bins}
    (d / "rssi_bins.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
