"""Origin-causal windows for fixed-model evaluation.

Candidate origins are the 15 min nominal times of a single 5 min grid built on
the raw observations. The prediction time is the latest observation at or
before that nominal time, within 15 min. History uses only observations at or
before that time. Targets use only observations strictly after it. One resample
pass for every cohort. A repeated (subject, t0) keeps the earlier nominal origin.

  PF_RICE_SRC   directory that contains the ceqt package
  RICE_DATA_DIR dataset root used by the released paths
  PF_RUNS       released training-run directory
  PF_EXT_CACHE  symmetric-resampling external-cache directory
  PF_OUT        C3 run directory (created)
"""

from __future__ import annotations

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
from ceqt import DT_MIN, ENC_LOOKBACK_MIN, MARK_LOOKBACK_MIN, PH_MIN  # noqa: E402
from ceqt.data.azt1d import list_azt1d_subjects, load_azt1d_subject  # noqa: E402
from ceqt.data.splits import azt1d_split  # noqa: E402
from ceqt.data.windows import WindowBatch, concat_batches  # noqa: E402
from ceqt.protocol import default_protocol  # noqa: E402

EPS = 1e-6
MAX_GAP = 15.0
N_ENC = int(round(ENC_LOOKBACK_MIN / DT_MIN))
N_PH = int(round(PH_MIN / DT_MIN))


def _raw(rec):
    t = np.asarray(rec.t_min, float)
    y = np.asarray(rec.cgm, float)
    ok = np.isfinite(t) & np.isfinite(y)
    row = np.flatnonzero(ok).astype(np.int64)
    t, y = t[ok], y[ok]
    order = np.lexsort((row, t))
    return t[order], y[order], row[order]


def _nearest(t, y, row, query, t0, after, eps=EPS):
    """Nearest observation to each query time. after=False keeps times <= t0."""
    n = len(t)
    left = np.searchsorted(t, query, side="right") - 1
    right = left + 1
    tl = t[np.clip(left, 0, n - 1)]
    tr = t[np.clip(right, 0, n - 1)]
    if after:
        left_ok = (left >= 0) & (tl > t0 + eps)
        right_ok = (right < n) & (tr > t0 + eps)
    else:
        left_ok = (left >= 0) & (tl <= t0 + eps)
        right_ok = (right < n) & (tr <= t0 + eps)
    dl = np.abs(tl - query)
    dr = np.abs(tr - query)
    left_ok = left_ok & (dl <= MAX_GAP + eps)
    right_ok = right_ok & (dr <= MAX_GAP + eps)
    choose_right = right_ok & (~left_ok | (dr < dl - eps) | (np.abs(dr - dl) <= eps))
    choose_left = left_ok & ~choose_right
    idx = np.where(choose_right, right, left)
    ok = choose_left | choose_right
    idx = np.clip(idx, 0, n - 1)
    return y[idx], t[idx], row[idx], ok


def causal_subject(rec, stride):
    t, y, row = _raw(rec)
    flow = {k: 0 for k in ("candidates", "no_past_observation", "history_missing", "target_missing", "duplicate_t0", "kept", "shared_target_source")}
    empty = dict(batch=None, nominal=np.zeros(0), flow=flow, leak_history=0)
    if len(t) < N_ENC + N_PH:
        return empty
    grid = np.arange(t[0], t[-1] + EPS, DT_MIN)
    n_stride = max(int(round(stride / DT_MIN)), 1)
    origins = np.arange(N_ENC - 1, len(grid) - N_PH, n_stride)
    if len(origins) == 0:
        return empty
    nominal = grid[origins]
    j = np.searchsorted(t, nominal + EPS, side="right") - 1
    age = np.where(j >= 0, nominal - t[np.clip(j, 0, len(t) - 1)], np.inf)
    has_past = (j >= 0) & (age <= MAX_GAP + EPS)
    flow["candidates"] = int(len(origins))
    flow["no_past_observation"] = int((~has_past).sum())
    if not np.any(has_past):
        return empty

    t0 = t[np.clip(j, 0, len(t) - 1)]
    hist_k = np.arange(N_ENC - 1, -1, -1)
    hist_time = t0[:, None] - DT_MIN * hist_k[None, :]
    fut_time = t0[:, None] + DT_MIN * np.arange(1, N_PH + 1)[None, :]
    t0b = t0[:, None]
    yh, _th, _rh, h_ok = _nearest(t, y, row, hist_time, t0b, after=False)
    yf, tf, rf, f_ok = _nearest(t, y, row, fut_time, t0b, after=True)
    hist_bad = ~h_ok.all(axis=1)
    fut_bad = ~f_ok.all(axis=1)
    # A window that already failed the anchor test is not counted again.
    hist_bad = hist_bad & has_past
    fut_bad = fut_bad & has_past & ~hist_bad
    flow["history_missing"] = int(hist_bad.sum())
    flow["target_missing"] = int(fut_bad.sum())
    keep = has_past & h_ok.all(axis=1) & f_ok.all(axis=1)
    # History must not read an observation after t0. This is a check, not a filter.
    leak = int(np.sum(keep & np.any(_th > t0b + EPS, axis=1)))

    seen = {}
    idx = []
    for i in np.flatnonzero(keep):
        key = round(float(t0[i]), 6)
        if key in seen:
            flow["duplicate_t0"] += 1
            continue
        seen[key] = i
        idx.append(i)
    idx = np.asarray(idx, int)
    flow["kept"] = int(len(idx))
    if len(idx) == 0:
        empty["leak_history"] = leak
        empty["flow"] = flow
        return empty
    shared = 0
    for i in idx:
        if len(np.unique(rf[i])) < N_PH:
            shared += 1
    flow["shared_target_source"] = int(shared)

    marks = sorted(rec.marks, key=lambda m: (m.t_min, m.e, m.a))
    mt_all = np.array([m.t_min for m in marks], float) if marks else np.zeros(0)
    ma_all = np.array([m.a for m in marks], float) if marks else np.zeros(0)
    me_all = [m.e for m in marks]
    mt, ma, me = [], [], []
    for i in idx:
        t0_i = float(t0[i])
        lo = np.searchsorted(mt_all, t0_i - MARK_LOOKBACK_MIN, side="left")
        hi = np.searchsorted(mt_all, t0_i + EPS, side="right")
        mt.append(mt_all[lo:hi].copy())
        ma.append(ma_all[lo:hi].copy())
        me.append(np.array(me_all[lo:hi], dtype=object))
    horizons = np.arange(DT_MIN, PH_MIN + EPS, DT_MIN)
    batch = WindowBatch(
        subject_id=np.full(len(idx), rec.subject_id),
        t0=t0[idx].copy(),
        y_hist=yh[idx].copy(),
        t_hist=hist_time[idx].copy(),
        y_future=yf[idx].copy(),
        horizons=horizons,
        mark_t=mt,
        mark_a=ma,
        mark_e=me,
        day_index=(t0[idx] - t[0]) / (24.0 * 60.0),
        n_marks=np.array([len(x) for x in mt], dtype=int),
    )
    return {"batch": batch, "nominal": nominal[idx].copy(), "flow": flow, "leak_history": leak, "target_time_max_before_t0": float(np.max(t0[idx, None] + EPS - tf[idx])) if False else 0.0}


def _add_flow(total, part):
    for k, v in part.items():
        total[k] = total.get(k, 0) + int(v)


def build_group(records, stride):
    batches, nominals = [], []
    flow = {}
    leaks = 0
    subjects = []
    lost = []
    for rec in records:
        got = causal_subject(rec, stride)
        _add_flow(flow, got["flow"])
        leaks += int(got["leak_history"])
        subjects.append(int(rec.subject_id))
        if got["batch"] is None:
            lost.append(int(rec.subject_id))
            continue
        batches.append(got["batch"])
        nominals.append(got["nominal"])
    if not batches:
        return None, np.zeros(0), flow, leaks, subjects, lost
    return concat_batches(batches), np.concatenate(nominals), flow, leaks, subjects, lost


def overlap(cache_batch, nominal, subjects):
    def keyset(sid, tt):
        return set(zip(np.asarray(sid, int).tolist(), np.round(np.asarray(tt, float), 6).tolist()))
    a = keyset(cache_batch.subject_id, cache_batch.t0)
    b = keyset(subjects, nominal)
    return {"both": len(a & b), "cache_only": len(a - b), "causal_only": len(b - a), "cache_windows": len(a), "causal_windows": len(b)}


def load_raw(loader):
    return loader()


def main():
    out_dir = Path(os.environ["PF_OUT"])
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = Path(os.environ["PF_RUNS"])
    ext = Path(os.environ["PF_EXT_CACHE"])
    stride = float(default_protocol()["stride_min"])
    t_start = time.time()
    # Disable the resample inside the loaders so the grid is built on raw samples.
    real_ohio, real_dt = ohio_mod.resample_5min, dt_mod.resample_5min
    ohio_mod.resample_5min = lambda rec: rec
    dt_mod.resample_5min = lambda rec: rec
    try:
        az = {int(s): load_azt1d_subject(int(s)) for s in list_azt1d_subjects()}
        ohio = ohio_mod.load_ohio_test()
        diatrend = [dt_mod.load_diatrend_subject(int(s)) for s in dt_mod.BASAL_SUBJECTS]
    finally:
        ohio_mod.resample_5min = real_ohio
        dt_mod.resample_5min = real_dt

    import pickle as _pkl
    report = {"stride_min": stride, "sets": {}}

    def finish(name, built, cache_batch):
        batch, nominal, flow, leaks, subjects, lost = built
        if leaks:
            raise RuntimeError(f"{name}: {leaks} windows have a history source after t0")
        entry = {
            "flow": flow,
            "subjects": len(subjects),
            "subjects_with_windows": len(subjects) - len(lost),
            "subjects_without_windows": lost,
            "leak_history_windows": leaks,
        }
        if batch is not None and cache_batch is not None:
            entry["overlap_nominal_t0"] = overlap(cache_batch, nominal, batch.subject_id)
        report["sets"][name] = entry
        print(name, flow.get("kept"), "lost", lost, flush=True)
        return batch, nominal

    az_out = {}
    for seed in (20260828, 20260829, 20260830):
        split = azt1d_split(seed)
        with (runs / f"20260830_s{seed}" / "cache" / f"azt1d_s{seed}_str15.pkl").open("rb") as f:
            cache = _pkl.load(f)
        parts = {}
        nominal = {}
        for part in ("train", "val", "test"):
            built = build_group([az[int(s)] for s in split[part]], stride)
            batch, nom = finish(f"azt1d_s{seed}_{part}", built, cache[part])
            parts[part] = batch
            nominal[part] = nom
        with (out_dir / f"azt1d_s{seed}.pkl").open("wb") as f:
            pickle.dump({"batch": parts, "nominal_t0": nominal}, f)
        az_out[seed] = True

    with (ext / "ohio_str15.pkl").open("rb") as f:
        ohio_cache = _pkl.load(f)["test"]
    built = build_group(ohio, stride)
    batch, nom = finish("ohiot1dm_test", built, ohio_cache)
    with (out_dir / "ohio.pkl").open("wb") as f:
        pickle.dump({"batch": {"test": batch}, "nominal_t0": {"test": nom}}, f)

    with (ext / "diatrend17_str15.pkl").open("rb") as f:
        dt_cache = _pkl.load(f)["test"]
    built = build_group(diatrend, stride)
    batch, nom = finish("diatrend_test", built, dt_cache)
    with (out_dir / "diatrend.pkl").open("wb") as f:
        pickle.dump({"batch": {"test": batch}, "nominal_t0": {"test": nom}}, f)

    report["elapsed_s"] = round(time.time() - t_start, 1)
    report["python"] = platform.python_version()
    report["numpy"] = np.__version__
    (out_dir / "window_flow.json").write_text(json.dumps(report, indent=1))
    (out_dir / "manifest.json").write_text(json.dumps({
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "elapsed_s": report["elapsed_s"],
    }, indent=1))
    print(out_dir / "window_flow.json", flush=True)


if __name__ == "__main__":
    main()
