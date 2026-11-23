"""R1-M1 (second round): temporal causality of the 5 min CGM resampling.

The released window construction fills each 5 min grid point from the nearest
raw observation within 15 min, on either side of the grid time. This script
traces, for every grid point, the raw observation that supplies its value, and
reports per window

  * the lead of the origin value: source observation time minus origin grid time;
  * whether any history value comes from an observation after the origin grid time;
  * whether the origin value and the first target value come from one raw observation;
  * the distance of the farthest history source from its grid time;
  * the change of the origin value under a causal rule (latest observation at or
    before the grid time, within 15 min).

OhioT1DM and DiaTrend pass through the resampling twice in the released code
(once in the loader, once in the window builder); the trace follows both passes.
The traced series is checked against the output of the released functions.

Usage:
  PF_RICE_SRC=<directory that contains the ceqt package> python r1m1_resample_causality.py
Output:
  results/r1m1/resample_causality.json
"""

from __future__ import annotations

import copy
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.environ["PF_RICE_SRC"])

import ceqt.data.diatrend as dt_mod  # noqa: E402
import ceqt.data.ohio as ohio_mod  # noqa: E402
from ceqt import DT_MIN, ENC_LOOKBACK_MIN, PH_MIN  # noqa: E402
from ceqt.data.azt1d import list_azt1d_subjects, load_azt1d_subject  # noqa: E402
from ceqt.data.resample import resample_5min  # noqa: E402
from ceqt.data.splits import azt1d_split  # noqa: E402
from ceqt.protocol import default_protocol  # noqa: E402

# Loaders return raw series; the passes are applied and traced below.
dt_mod.resample_5min = lambda rec: rec
ohio_mod.resample_5min = lambda rec: rec

MAX_GAP = 15.0
EPS = 1e-6
SEEDS = (20260828, 20260829, 20260830)
OUT = Path(__file__).resolve().parent / "results" / "r1m1"


def trace_pass(t, y, src):
    """One pass of resample_5min with the source time of every output value."""
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    src = np.asarray(src, float)
    ok = np.isfinite(t) & np.isfinite(y)
    tt, yy, ss = t[ok], y[ok], src[ok]
    if len(tt) < 8:
        return t, y, src, False
    order = np.argsort(tt)
    tt, yy, ss = tt[order], yy[order], ss[order]
    dts = np.diff(tt)
    if len(dts) and np.median(dts) < DT_MIN + 0.6 and np.max(dts) <= MAX_GAP + EPS:
        return t, y, src, False
    grid = np.arange(tt[0], tt[-1] + EPS, DT_MIN)
    out = np.full(grid.shape, np.nan)
    osrc = np.full(grid.shape, np.nan)
    j = 0
    for i, g in enumerate(grid):
        while j + 1 < len(tt) and abs(tt[j + 1] - g) <= abs(tt[j] - g):
            j += 1
        if abs(tt[j] - g) <= MAX_GAP:
            out[i] = yy[j]
            osrc[i] = ss[j]
    return grid, out, osrc, True


def causal_series(t_raw, y_raw, grid):
    """Latest raw observation at or before each grid time, within MAX_GAP."""
    ok = np.isfinite(t_raw) & np.isfinite(y_raw)
    tt, yy = t_raw[ok], y_raw[ok]
    order = np.argsort(tt, kind="stable")
    tt, yy = tt[order], yy[order]
    j = np.searchsorted(tt, grid + EPS, side="right") - 1
    valid = j >= 0
    jj = np.clip(j, 0, len(tt) - 1)
    age = grid - tt[jj]
    good = valid & (age <= MAX_GAP + EPS)
    return np.where(good, yy[jj], np.nan), np.where(good, tt[jj], np.nan)


def subject_windows(rec, passes: int, stride: float):
    t_raw = np.asarray(rec.t_min, float).copy()
    y_raw = np.asarray(rec.cgm, float).copy()
    t, y, src = t_raw, y_raw, t_raw.copy()
    gridded = []
    for _ in range(passes):
        t, y, src, g = trace_pass(t, y, src)
        gridded.append(g)

    ref = copy.deepcopy(rec)
    for _ in range(passes):
        ref = resample_5min(ref)
    assert np.array_equal(np.asarray(ref.t_min, float), t), f"grid mismatch, subject {rec.subject_id}"
    assert np.array_equal(np.asarray(ref.cgm, float), y, equal_nan=True), f"value mismatch, subject {rec.subject_id}"

    n_enc = int(round(ENC_LOOKBACK_MIN / DT_MIN))
    n_ph = int(round(PH_MIN / DT_MIN))
    n_stride = max(int(round(stride / DT_MIN)), 1)
    fin = np.isfinite(y)
    y_c, src_c = causal_series(t_raw, y_raw, t)
    rows = []
    for i in np.arange(n_enc - 1, len(t) - n_ph, n_stride):
        t0 = t[i]
        if t0 < t[0] + ENC_LOOKBACK_MIN or t0 > t[-1] - PH_MIN:
            continue
        h = slice(i - (n_enc - 1), i + 1)
        f = slice(i + 1, i + 1 + n_ph)
        if not (fin[h].all() and fin[f].all()):
            continue
        hs = src[h]
        rows.append((
            src[i] - t0,                                   # lead of the origin value
            int((hs > t0 + EPS).sum()),                    # history values taken after t0
            float(np.max(hs) - t0),                        # largest lead in the history
            int(src[i + 1] <= src[i] + EPS),               # origin and first target share a source
            int((src[f] <= src[i] + EPS).sum()),           # target values not later than the origin source
            float(np.max(np.abs(hs - t[h]))),              # farthest history source from its grid time
            float(np.max(np.abs(src[f] - t[f]))),          # farthest target source from its grid time
            y[i] - y_c[i] if np.isfinite(y_c[i]) else np.nan,
            int(np.isfinite(y_c[h]).all()),                # history complete under the causal rule
        ))
    return np.array(rows, float).reshape(-1, 9), gridded


def summarize(rows: np.ndarray) -> dict:
    n = len(rows)
    if n == 0:
        return {"windows": 0}
    lead, n_fut, max_lead, share, n_tgt, far_h, far_f, dy, causal_ok = rows.T
    bins = [(-np.inf, -EPS), (-EPS, EPS), (EPS, 2.5), (2.5, 5.0), (5.0, 10.0), (10.0, 15.0), (15.0, np.inf)]
    names = ["before_t0", "at_t0", "0-2.5", "2.5-5", "5-10", "10-15", ">15"]
    hist = {nm: int(((lead > lo) & (lead <= hi)).sum()) for nm, (lo, hi) in zip(names, bins)}
    d = np.abs(dy[np.isfinite(dy)])
    return {
        "windows": int(n),
        "origin_lead_min": {"bins": hist, "max": float(lead.max()), "min": float(lead.min()),
                            "median": float(np.median(lead))},
        "origin_from_after_t0": int((lead > EPS).sum()),
        "origin_from_after_t0_share": float((lead > EPS).mean()),
        "origin_lead_over_2.5min": int((lead > 2.5 + EPS).sum()),
        "origin_lead_over_2.5min_share": float((lead > 2.5 + EPS).mean()),
        "any_history_after_t0": int((n_fut > 0).sum()),
        "any_history_after_t0_share": float((n_fut > 0).mean()),
        "history_values_after_t0_max": int(n_fut.max()),
        "largest_history_lead_min": float(max_lead.max()),
        "origin_shares_source_with_first_target": int(share.sum()),
        "origin_shares_source_with_first_target_share": float(share.mean()),
        "target_values_not_after_origin_source_max": int(n_tgt.max()),
        "history_source_over_15min_from_grid": int((far_h > MAX_GAP + EPS).sum()),
        "history_source_farthest_min": float(far_h.max()),
        "target_source_over_15min_from_grid": int((far_f > MAX_GAP + EPS).sum()),
        "target_source_farthest_min": float(far_f.max()),
        "origin_value_change_under_causal_rule_mgdl": {
            "windows_with_causal_value": int(len(d)),
            "changed": int((d > EPS).sum()),
            "changed_share": float((d > EPS).mean()) if len(d) else None,
            "mean_abs": float(d.mean()) if len(d) else None,
            "p95_abs": float(np.percentile(d, 95)) if len(d) else None,
            "max_abs": float(d.max()) if len(d) else None,
            "over_5": int((d > 5).sum()),
            "over_10": int((d > 10).sum()),
        },
        "history_complete_under_causal_rule": int(causal_ok.sum()),
        "history_complete_under_causal_rule_share": float(causal_ok.mean()),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    stride = float(default_protocol()["stride_min"])
    t_start = time.time()
    out = {"stride_min": stride, "max_gap_min": MAX_GAP, "passes": {"azt1d": 1, "ohiot1dm": 2, "diatrend": 2}}

    az = {}
    for sid in list_azt1d_subjects():
        az[int(sid)], g = subject_windows(load_azt1d_subject(int(sid)), 1, stride)
        assert all(g), f"AZT1D subject {sid} took the early return"
    out["azt1d_all_25"] = summarize(np.concatenate(list(az.values())))
    for seed in SEEDS:
        split = azt1d_split(seed)
        for part in ("train", "val", "test"):
            rows = np.concatenate([az[int(s)] for s in split[part]])
            out[f"azt1d_s{seed}_{part}"] = summarize(rows)
    print("azt1d", {k: v["windows"] for k, v in out.items() if k.startswith("azt1d")}, flush=True)

    rows, second = [], []
    for rec in ohio_mod.load_ohio_test():
        r, g = subject_windows(rec, 2, stride)
        rows.append(r)
        second.append(g[1])
    out["ohiot1dm"] = summarize(np.concatenate(rows))
    out["ohiot1dm"]["subjects_regridded_in_second_pass"] = int(sum(second))
    print("ohiot1dm", out["ohiot1dm"]["windows"], "second-pass regrid", sum(second), "of", len(second), flush=True)

    rows, second = [], []
    for sid in dt_mod.BASAL_SUBJECTS:
        r, g = subject_windows(dt_mod.load_diatrend_subject(int(sid)), 2, stride)
        rows.append(r)
        second.append(g[1])
    out["diatrend"] = summarize(np.concatenate(rows))
    out["diatrend"]["subjects_regridded_in_second_pass"] = int(sum(second))
    print("diatrend", out["diatrend"]["windows"], "second-pass regrid", sum(second), "of", len(second), flush=True)

    out["elapsed_s"] = round(time.time() - t_start, 1)
    dst = OUT / "resample_causality.json"
    dst.write_text(json.dumps(out, indent=1))
    for k, v in out.items():
        if isinstance(v, dict) and "windows" in v:
            c = v["origin_value_change_under_causal_rule_mgdl"]
            print(f"{k}: windows {v['windows']}, origin after t0 {v['origin_from_after_t0_share']:.4f}, "
                  f"lead > 2.5 min {v['origin_lead_over_2.5min_share']:.4f}, "
                  f"shares source with first target {v['origin_shares_source_with_first_target_share']:.4f}, "
                  f"largest history lead {v['largest_history_lead_min']:.1f} min, "
                  f"history source > 15 min {v['history_source_over_15min_from_grid']}, "
                  f"causal change mean {c['mean_abs']:.2f} mg/dL")
    print(dst)


if __name__ == "__main__":
    main()
