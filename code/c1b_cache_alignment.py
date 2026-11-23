"""Align submitted window caches with a rebuild from the released loaders.

The rebuild uses the released resample and window builder, including the second
resample of OhioT1DM and DiaTrend. Source times are traced through those passes
back to a raw row. A source time shared by more than one raw row is unknown.

Inputs are read only. Set PYTHONDONTWRITEBYTECODE=1 before launching.

  PF_RICE_SRC   directory that contains the ceqt package
  PF_RUNS       directory with 20260830_s<seed>/cache/azt1d_s<seed>_str15.pkl
  PF_EXT_CACHE  directory with ohio_str15.pkl and diatrend17_str15.pkl
  RICE_DATA_DIR dataset root used by the released paths
  PF_OUT        run directory (created)
"""

from __future__ import annotations

import copy
import json
import os
import pickle
import platform
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
sys.path.insert(0, os.environ["PF_RICE_SRC"])

import ceqt.data.diatrend as dt_mod  # noqa: E402
import ceqt.data.ohio as ohio_mod  # noqa: E402
from ceqt import DT_MIN, ENC_LOOKBACK_MIN, PH_MIN  # noqa: E402
from ceqt.data.azt1d import list_azt1d_subjects, load_azt1d_subject  # noqa: E402
from ceqt.data.cache import build_subject_windows  # noqa: E402
from ceqt.data.resample import resample_5min  # noqa: E402
from ceqt.data.splits import azt1d_split  # noqa: E402
from ceqt.data.windows import concat_batches  # noqa: E402
from ceqt.protocol import default_protocol  # noqa: E402

EPS = 1e-6
MAX_GAP = 15.0
SEEDS = (20260828, 20260829, 20260830)


def trace_pass(t, y, row):
    """One released resample pass. row indexes the original raw series."""
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    row = np.asarray(row, np.int64)
    ok = np.isfinite(t) & np.isfinite(y)
    tt, yy, rr = t[ok], y[ok], row[ok]
    if len(tt) < 8:
        return t, y, row, False
    order = np.argsort(tt)
    tt, yy, rr = tt[order], yy[order], rr[order]
    dts = np.diff(tt)
    if len(dts) and np.median(dts) < DT_MIN + 0.6 and np.max(dts) <= MAX_GAP + EPS:
        return t, y, row, False
    grid = np.arange(tt[0], tt[-1] + EPS, DT_MIN)
    out = np.full(grid.shape, np.nan)
    orow = np.full(grid.shape, -1, dtype=np.int64)
    j = 0
    for i, g in enumerate(grid):
        while j + 1 < len(tt) and abs(tt[j + 1] - g) <= abs(tt[j] - g):
            j += 1
        if abs(tt[j] - g) <= MAX_GAP:
            out[i] = yy[j]
            orow[i] = rr[j]
    return grid, out, orow, True


def trace_with_raw_clock(t_raw, y_raw, n_pass):
    """Trace source row and source time through n_pass resamples."""
    raw_t = np.asarray(t_raw, float).copy()
    t = raw_t.copy()
    y = np.asarray(y_raw, float).copy()
    row = np.arange(len(t), dtype=np.int64)
    gridded = []
    for _ in range(n_pass):
        t, y, row, g = trace_pass(t, y, row)
        gridded.append(bool(g))
    src_t = np.full(len(t), np.nan)
    known = row >= 0
    src_t[known] = raw_t[row[known]]
    dup = _duplicate_times(raw_t)
    amb = np.zeros(len(t), dtype=bool)
    amb[known] = dup[row[known]]
    return t, y, row, src_t, amb, gridded


def _duplicate_times(t):
    """True where a finite timestamp occurs more than once."""
    out = np.zeros(len(t), dtype=bool)
    ok = np.isfinite(t)
    if ok.sum() == 0:
        return out
    tt = t[ok]
    order = np.argsort(tt)
    tt = tt[order]
    same = np.zeros(len(tt), dtype=bool)
    if len(tt) > 1:
        eq = np.abs(np.diff(tt)) <= EPS
        same[1:] |= eq
        same[:-1] |= eq
    idx = np.flatnonzero(ok)[order]
    out[idx] = same
    return out


def window_source_rows(t, y, row, src_t, amb, stride):
    """Per kept window, using the same origin rule as build_windows."""
    n_enc = int(round(ENC_LOOKBACK_MIN / DT_MIN))
    n_ph = int(round(PH_MIN / DT_MIN))
    n_stride = max(int(round(stride / DT_MIN)), 1)
    fin = np.isfinite(y)
    rows = []
    if len(t) == 0:
        return rows
    for i in np.arange(n_enc - 1, len(t) - n_ph, n_stride):
        t0 = float(t[i])
        if t0 < t[0] + ENC_LOOKBACK_MIN or t0 > t[-1] - PH_MIN:
            continue
        h = slice(i - (n_enc - 1), i + 1)
        f = slice(i + 1, i + 1 + n_ph)
        if not (fin[h].all() and fin[f].all()):
            continue
        hs = src_t[h]
        hr = row[h]
        fs = src_t[f]
        fr = row[f]
        n_hist_after = int(np.sum(hs > t0 + EPS))
        origin_after = bool(src_t[i] > t0 + EPS)
        first_le = bool(fs[0] <= src_t[i] + EPS)
        same_row = bool(fr[0] == hr[-1] and fr[0] >= 0 and not amb[i] and not amb[i + 1])
        n_unknown = int(np.sum(amb[h]) + np.sum(amb[f]) + np.sum(hr < 0) + np.sum(fr < 0))
        rows.append({
            "origin_after": origin_after,
            "n_hist_after": n_hist_after,
            "any_hist_after": n_hist_after > 0,
            "first_target_time_le_origin_time": first_le,
            "same_raw_row": same_row and first_le and abs(fs[0] - src_t[i]) <= EPS,
            "n_unknown": n_unknown,
        })
    return rows


def official_batch(records, stride):
    batches = []
    for rec in records:
        b = build_subject_windows(copy.deepcopy(rec), stride=stride)
        if b is not None:
            batches.append(b)
    if not batches:
        return None
    return concat_batches(batches)


def load_raw_ohio():
    real = ohio_mod.resample_5min
    ohio_mod.resample_5min = lambda rec: rec
    try:
        return ohio_mod.load_ohio_test()
    finally:
        ohio_mod.resample_5min = real


def load_raw_diatrend(sid):
    real = dt_mod.resample_5min
    dt_mod.resample_5min = lambda rec: rec
    try:
        return dt_mod.load_diatrend_subject(int(sid))
    finally:
        dt_mod.resample_5min = real


def mark_signature(batch, i):
    t = np.asarray(batch.mark_t[i], float)
    a = np.asarray(batch.mark_a[i], float)
    e = [str(x) for x in np.asarray(batch.mark_e[i], dtype=object).tolist()]
    return tuple(sorted(zip(np.round(t, 6).tolist(), np.round(a, 6).tolist(), e)))


def compare_batches(cache, rebuilt):
    out = {
        "cache_windows": int(len(cache)),
        "rebuilt_windows": int(len(rebuilt)),
        "length_equal": int(len(cache)) == int(len(rebuilt)),
    }
    n = min(len(cache), len(rebuilt))
    if n == 0:
        out["fields_equal"] = False
        return out
    id_ok = bool(np.array_equal(cache.subject_id[:n], rebuilt.subject_id[:n]))
    t0_max = float(np.max(np.abs(cache.t0[:n] - rebuilt.t0[:n])))
    y_exact = bool(np.array_equal(cache.y_hist[:n], rebuilt.y_hist[:n]))
    y_max = float(np.max(np.abs(cache.y_hist[:n] - rebuilt.y_hist[:n])))
    th_max = float(np.max(np.abs(cache.t_hist[:n] - rebuilt.t_hist[:n])))
    yf_exact = bool(np.array_equal(cache.y_future[:n], rebuilt.y_future[:n]))
    yf_max = float(np.max(np.abs(cache.y_future[:n] - rebuilt.y_future[:n])))
    hzn_max = float(np.max(np.abs(cache.horizons - rebuilt.horizons)))
    nmark_ok = bool(np.array_equal(cache.n_marks[:n], rebuilt.n_marks[:n]))
    mark_mismatch = 0
    if nmark_ok:
        for i in range(n):
            if mark_signature(cache, i) != mark_signature(rebuilt, i):
                mark_mismatch += 1
    else:
        mark_mismatch = n
    future_marks = 0
    for i in range(n):
        mt = np.asarray(cache.mark_t[i], float)
        if len(mt):
            future_marks += int(np.sum(mt > cache.t0[i] + EPS))
    out.update({
        "subject_id_equal": id_ok,
        "t0_max_abs_min": t0_max,
        "y_hist_exact": y_exact,
        "y_hist_max_abs": y_max,
        "t_hist_max_abs_min": th_max,
        "y_future_exact": yf_exact,
        "y_future_max_abs": yf_max,
        "horizons_max_abs_min": hzn_max,
        "n_marks_equal": nmark_ok,
        "mark_multiset_mismatches": int(mark_mismatch),
        "cache_windows_with_mark_after_t0": int(future_marks > 0) if False else None,
        "cache_mark_times_after_t0": int(future_marks),
    })
    # Count windows, not mark rows, that contain a future mark.
    n_win = 0
    for i in range(len(cache)):
        mt = np.asarray(cache.mark_t[i], float)
        if len(mt) and np.any(mt > cache.t0[i] + EPS):
            n_win += 1
    out["cache_windows_with_mark_after_t0"] = int(n_win)
    out["fields_equal"] = bool(
        out["length_equal"] and id_ok and t0_max <= EPS and y_exact and th_max <= EPS
        and yf_exact and hzn_max <= EPS and nmark_ok and mark_mismatch == 0
    )
    return out


def cross_tab(stats):
    n = len(stats)
    origin_after = np.array([s["origin_after"] for s in stats], bool)
    any_after = np.array([s["any_hist_after"] for s in stats], bool)
    only_origin = origin_after & ~np.array([s["n_hist_after"] > 1 for s in stats])
    # "only the origin is after t0" among windows with any history after t0:
    only_origin = origin_after & np.array([s["n_hist_after"] == 1 for s in stats], bool)
    first_le = np.array([s["first_target_time_le_origin_time"] for s in stats], bool)
    same_row = np.array([s["same_raw_row"] for s in stats], bool)
    unknown_windows = int(sum(s["n_unknown"] > 0 for s in stats))
    unknown_cells = int(sum(s["n_unknown"] for s in stats))
    hist_after_counts = np.array([s["n_hist_after"] for s in stats], int)
    return {
        "windows": n,
        "origin_after_t0": int(origin_after.sum()),
        "any_history_after_t0": int(any_after.sum()),
        "history_after_only_at_origin": int(only_origin.sum()),
        "history_values_after_t0_max": int(hist_after_counts.max()) if n else 0,
        "history_values_after_t0_sum": int(hist_after_counts.sum()),
        "first_target_time_le_origin_time": int(first_le.sum()),
        "same_raw_row_and_time": int(same_row.sum()),
        "cross": {
            "origin_after_and_first_target_time_le": int((origin_after & first_le).sum()),
            "origin_not_after_and_first_target_time_le": int((~origin_after & first_le).sum()),
            "origin_after_and_same_raw_row": int((origin_after & same_row).sum()),
            "origin_not_after_and_same_raw_row": int((~origin_after & same_row).sum()),
        },
        "unknown_windows": unknown_windows,
        "unknown_cells": unknown_cells,
    }


def source_for_records(raw_records, official_records, n_pass, stride):
    """Trace raw rows through n_pass resamples and align them with official windows.

    official_records must be the objects the released cache builder resamples:
    raw AZT1D records (one resample inside build_subject_windows), or the
    already once-resampled OhioT1DM and DiaTrend records (second resample).
    """
    stats = []
    value_mismatch = 0
    for raw, off in zip(raw_records, official_records):
        t, y, row, src_t, amb, _g = trace_with_raw_clock(raw.t_min, raw.cgm, n_pass)
        official = build_subject_windows(copy.deepcopy(off), stride=stride)
        traced = window_source_rows(t, y, row, src_t, amb, stride)
        if official is None:
            if traced:
                value_mismatch += len(traced)
            continue
        if len(official) != len(traced):
            value_mismatch += abs(len(official) - len(traced))
            stats.extend(traced[: len(official)])
            continue
        # Compare traced glucose with the official batch on the origin column.
        # A full-array check is done in compare_batches; here only the count alignment.
        stats.extend(traced)
        if not np.allclose(official.y_hist[:, -1], y_origin_of(t, y, stride), equal_nan=True):
            value_mismatch += 1
    return stats, value_mismatch


def y_origin_of(t, y, stride):
    n_enc = int(round(ENC_LOOKBACK_MIN / DT_MIN))
    n_ph = int(round(PH_MIN / DT_MIN))
    n_stride = max(int(round(stride / DT_MIN)), 1)
    fin = np.isfinite(y)
    vals = []
    for i in np.arange(n_enc - 1, len(t) - n_ph, n_stride):
        t0 = float(t[i])
        if t0 < t[0] + ENC_LOOKBACK_MIN or t0 > t[-1] - PH_MIN:
            continue
        h = slice(i - (n_enc - 1), i + 1)
        f = slice(i + 1, i + 1 + n_ph)
        if not (fin[h].all() and fin[f].all()):
            continue
        vals.append(y[i])
    return np.asarray(vals, float)


def summarize_group(name, cache, rebuilt, stats, trace_mismatch):
    aligned = compare_batches(cache, rebuilt)
    tab = cross_tab(stats) if len(stats) == len(rebuilt) else {
        "error": "source-window count differs from rebuilt windows",
        "source_windows": len(stats),
        "rebuilt_windows": int(len(rebuilt)),
    }
    return {
        "name": name,
        "alignment": aligned,
        "source": tab,
        "trace_value_flags": int(trace_mismatch),
    }


def main():
    out_dir = Path(os.environ["PF_OUT"])
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = Path(os.environ["PF_RUNS"])
    ext = Path(os.environ["PF_EXT_CACHE"])
    stride = float(default_protocol()["stride_min"])
    t0 = time.time()
    groups = []

    raw_az = {int(s): load_azt1d_subject(int(s)) for s in list_azt1d_subjects()}
    for seed in SEEDS:
        split = azt1d_split(seed)
        with (runs / f"20260830_s{seed}" / "cache" / f"azt1d_s{seed}_str15.pkl").open("rb") as f:
            cache = pickle.load(f)
        for part in ("train", "val", "test"):
            recs = [raw_az[int(s)] for s in split[part]]
            rebuilt = official_batch(recs, stride)
            stats, flags = source_for_records(recs, recs, 1, stride)
            groups.append(summarize_group(f"azt1d_s{seed}_{part}", cache[part], rebuilt, stats, flags))
            print(groups[-1]["name"], "equal" if groups[-1]["alignment"]["fields_equal"] else "DIFF", flush=True)

    raw_ohio = load_raw_ohio()
    ohio_official = ohio_mod.load_ohio_test()
    with (ext / "ohio_str15.pkl").open("rb") as f:
        ohio_cache = pickle.load(f)["test"]
    groups.append(summarize_group(
        "ohiot1dm_test",
        ohio_cache,
        official_batch(ohio_official, stride),
        *source_for_records(raw_ohio, ohio_official, 2, stride),
    ))
    print("ohiot1dm", "equal" if groups[-1]["alignment"]["fields_equal"] else "DIFF", flush=True)

    raw_dt = [load_raw_diatrend(s) for s in dt_mod.BASAL_SUBJECTS]
    dt_official = [dt_mod.load_diatrend_subject(int(s)) for s in dt_mod.BASAL_SUBJECTS]
    with (ext / "diatrend17_str15.pkl").open("rb") as f:
        dt_cache = pickle.load(f)["test"]
    groups.append(summarize_group(
        "diatrend_test",
        dt_cache,
        official_batch(dt_official, stride),
        *source_for_records(raw_dt, dt_official, 2, stride),
    ))
    print("diatrend", "equal" if groups[-1]["alignment"]["fields_equal"] else "DIFF", flush=True)

    # resample_5min is referenced so a missing import fails before the long run.
    _ = resample_5min
    manifest = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "stride_min": stride,
        "elapsed_s": round(time.time() - t0, 1),
        "rice_src_note": "released package via PF_RICE_SRC",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    (out_dir / "alignment.json").write_text(json.dumps(groups, indent=1))
    print(out_dir / "alignment.json", flush=True)


if __name__ == "__main__":
    main()
