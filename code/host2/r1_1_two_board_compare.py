#!/usr/bin/env python3
"""R1-1: bitwise comparison of two boards on DiaTrend.

All 115,934 windows of the DevKitC (frozen firmware 10cba768…, results/audit/devkitc_diatrend17_full) serve as the baseline and are compared with
the WiFiduino stratified sample of 17,000 windows (diatrend17_sample1000, 10cba768…) and the second-half 57,967 windows
(fw16b3_diatrend17_second_half, 16b32e5f…), aligned by full-set index; status code, flags, event bits, the 16 logits,
the 48-point trajectory and g are compared bitwise (float32). While the DevKitC run is unfinished, the completed windows in ckpt.npz are compared (--partial).
Results: code/results/r1_1/two_board_compare.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

RES = Path(__file__).resolve().parent.parent / "results"
AUD = RES / "audit"
WIFI = {"sample17000": ("diatrend17_sample1000", RES / "r1_1" / "diatrend17_sample1000.npy"),
        "second_half": ("fw16b3_diatrend17_second_half", RES / "r1_1" / "diatrend17_second_half.npy")}
INT_KEYS = ("status", "flags")
FLOAT_KEYS = ("logits", "yhat", "g")


def load_devkitc(partial: bool):
    d = AUD / "devkitc_diatrend17_full"
    if (d / "audit.npz").exists():
        z = np.load(d / "audit.npz")
        return z, len(z["status"]), "audit.npz"
    if not partial:
        raise SystemExit("devkitc audit.npz missing; use --partial")
    z = np.load(d / "ckpt.npz")
    return z, int(z["done"]), "ckpt.npz"


def bits(a):
    a = np.ascontiguousarray(np.asarray(a, np.float32))
    return a.view(np.uint32).reshape(len(a), -1)


def compare(dk, n_dk, w, full_idx):
    m = full_idx < n_dk
    a, b = full_idx[m], np.where(m)[0]
    out = {"windows": int(m.sum())}
    for k in INT_KEYS:
        out[f"{k}_differ"] = int((np.asarray(dk[k])[a] != np.asarray(w[k])[b]).sum())
    ev = np.asarray(dk["evt"])[a] != np.asarray(w["evt"])[b]
    out["evt_cells"] = int(ev.size)
    out["evt_cells_differ"] = int(ev.sum())
    for k in FLOAT_KEYS:
        diff = (bits(np.asarray(dk[k])[a]) != bits(np.asarray(w[k])[b])).any(axis=1)
        out[f"{k}_windows_differ"] = int(diff.sum())
    out["bit_identical"] = all(v == 0 for kk, v in out.items() if kk.endswith("differ"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--partial", action="store_true")
    args = ap.parse_args()
    dk, n_dk, src = load_devkitc(args.partial)
    res = {"devkitc_source": src, "devkitc_windows": n_dk}
    for name, (tag, idx_path) in WIFI.items():
        w = np.load(AUD / tag / "audit.npz")
        full_idx = np.load(idx_path)
        if "idx" in w.files and not np.array_equal(np.asarray(w["idx"]), full_idx):
            raise SystemExit(f"{tag}: idx in audit.npz differs from {idx_path.name}")
        res[name] = {"wifiduino_tag": tag, **compare(dk, n_dk, w, full_idx)}
    out = RES / "r1_1" / ("two_board_compare_partial.json" if src == "ckpt.npz" else "two_board_compare.json")
    out.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False))


if __name__ == "__main__":
    main()
