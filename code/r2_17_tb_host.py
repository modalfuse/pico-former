#!/usr/bin/env python3
"""R2-17 T-B (host part): fault injection on clean windows; counts status codes, residual gate g, and output changes.

Windows: 512 stratified-sample windows of the AZT1D held-out set (same as board sample512), seed-20260828 checkpoint.
Two injection levels:
  Preprocessing level (R2-R4): on the 72 raw CGM samples of a window, drop, delay, duplicate, reorder samples or
    set NaN timestamps, then align to the original grid by the release-package rule (nearest sample within 15 min,
    otherwise gap); a gap in the grid gives GAP, and a last valid sample more than 15 min before t0 gives STALE;
  Frame level (R1, R5-R8): on the packed window frame, inject CGM limit readings and out-of-range values, missing marks,
    abnormal doses, abnormal type codes, cleared or excessive marks, and non-finite values,
    then pass through pf2.guard (same rules as firmware validate()).
Host reference is the float64 forward pass (pf2.host_forward); event bits and curves are compared with the clean window.
For missing marks, changes in MAE and 4 h hypo70 FNR/FPR are additionally computed on the full held-out set.
Outputs: results/r2_17/tb_host.csv, tb_host.json; window frames of each injection are stored in
results/r2_17/tb_frames/<name>.npz for board T-B.
"""

from __future__ import annotations

import json
import pickle
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "host2"))
import pf2  # noqa: E402
from ceqt.data.windows import WindowBatch  # noqa: E402

OUT = HERE / "results" / "r2_17"
FR_DIR = OUT / "tb_frames"
SEED = 20261009
CACHE = pf2.BASE / "runs/20260830_s20260828/cache/azt1d_s20260828_str15.pkl"
CKPT = pf2.BASE / "runs/20260830_s20260828/ckpt/ceqt.pt"
SAMPLE512 = pf2.BASE / "validation/sample512.npz"
MAX_GAP = 15.0
STATUS_NAME = {pf2.OK: "OK", pf2.NO_PRED_GAP: "GAP", pf2.NO_PRED_STALE: "STALE", pf2.NO_PRED_INVALID: "INVALID"}


def sub_batch(b: WindowBatch, idx) -> WindowBatch:
    idx = np.asarray(idx)
    return replace(b, subject_id=b.subject_id[idx], t0=b.t0[idx], y_hist=b.y_hist[idx], t_hist=b.t_hist[idx],
                   y_future=b.y_future[idx], mark_t=[b.mark_t[i] for i in idx], mark_a=[b.mark_a[i] for i in idx],
                   mark_e=[b.mark_e[i] for i in idx], day_index=b.day_index[idx], n_marks=b.n_marks[idx])


def assemble(grid, t, y):
    """Align raw samples to the window grid (nearest sample within 15 min); duplicates keep the last arrival."""
    ok = np.isfinite(t) & np.isfinite(y)
    t, y = t[ok], y[ok]
    last = {}
    for k in range(len(t)):
        last[float(t[k])] = float(y[k])
    t = np.array(sorted(last))
    y = np.array([last[v] for v in t])
    out = np.full(len(grid), np.nan, np.float32)
    for g_i, g in enumerate(grid):
        d = np.abs(t - g)
        j = int(np.argmin(d))
        if d[j] <= MAX_GAP:
            out[g_i] = y[j]
    return out, (float(t[-1]) if len(t) else -np.inf)


def preproc_status(y, t_last, t0):
    if t0 - t_last > MAX_GAP:
        return pf2.NO_PRED_STALE
    if not np.isfinite(y).all():
        return pf2.NO_PRED_GAP
    return pf2.OK


class Runner:
    def __init__(self, batch, net, y_mean, y_std, thr):
        self.b, self.net, self.y_mean, self.y_std, self.thr = batch, net, y_mean, y_std, thr
        self.fr = pf2.frames(batch)
        self.clean = self.forward(self.fr, np.zeros(len(batch), np.int32))

    def forward(self, fr, pre_status):
        n = len(fr["y"])
        st = pre_status.copy()
        y = fr["y"].copy()
        feat, mask, q, flags = fr["feat"].copy(), fr["mask"].copy(), fr["q"].copy(), fr["flags"].copy()
        for i in range(n):
            if st[i] != pf2.OK:
                continue
            s, yc, f, m, qq, fl, _ = pf2.guard(fr["y"][i], fr["feat"][i], fr["mask"][i], fr["q"][i], fr["flags"][i],
                                               fr["t0_s"][i], fr["t_last_s"][i])
            st[i], y[i], feat[i], mask[i], q[i], flags[i] = s, yc, f, m, qq, fl
        ok = st == pf2.OK
        logits = np.full((n, 16), np.nan)
        yhat = np.full((n, 48), np.nan)
        g = np.full(n, np.nan)
        if ok.any():
            iner = np.where(((flags & pf2.FLAG_CLAMPED) > 0)[:, None], y[:, -1:].astype(np.float64), fr["iner"])
            l, yh, gg = pf2.host_forward(self.net, self.y_mean, self.y_std, y[ok], feat[ok], mask[ok],
                                         fr["n_marks"][ok], q[ok], iner[ok])
            logits[ok], yhat[ok], g[ok] = l, yh, gg
        return {"status": st, "flags": flags, "logits": logits, "yhat": yhat, "g": g,
                "evt": (logits >= self.thr[None, :]).astype(np.int8)}

    def report(self, name, layer, res, fr=None):
        st = res["status"]
        ok = st == pf2.OK
        c = self.clean
        both = ok & (c["status"] == pf2.OK)
        row = {"injection": name, "layer": layer, "n": int(len(st)),
               **{f"status_{v}": int((st == k).sum()) for k, v in STATUS_NAME.items()},
               "flag_clamped": int(((res["flags"] & pf2.FLAG_CLAMPED) > 0).sum()),
               "flag_dropped": int(((res["flags"] & pf2.FLAG_DROPPED) > 0).sum()),
               "flag_truncated": int(((res["flags"] & pf2.FLAG_TRUNC) > 0).sum()),
               "g_1": int((res["g"][ok] == 1).sum()), "g_045": int((res["g"][ok] == 0.45).sum()),
               "g_0": int((res["g"][ok] == 0).sum()),
               "evt_windows_changed": int((res["evt"][both] != c["evt"][both]).any(1).sum()),
               "evt_cells_changed": int((res["evt"][both] != c["evt"][both]).sum()),
               "logit_maxabs": float(np.nanmax(np.abs(res["logits"][both] - c["logits"][both]))) if both.any() else None,
               "yhat_maxabs": float(np.nanmax(np.abs(res["yhat"][both] - c["yhat"][both]))) if both.any() else None,
               "evt_pos_on_ok": int(res["evt"][ok].sum())}
        if fr is not None:
            FR_DIR.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(FR_DIR / f"{name}.npz", **{k: fr[k] for k in
                                ("y", "feat", "mask", "n_marks", "q", "flags", "t0_s", "t_last_s", "iner")},
                                exp_status=st, exp_flags=res["flags"], exp_logits=res["logits"], exp_yhat=res["yhat"])
        return row


def preproc_injections(b, rng):
    """name -> function(t, y, t0) returning modified raw (t, y)."""
    def drop_mid(k):
        def f(t, y, t0):
            s = rng.randint(1, len(t) - k - 1)
            keep = np.r_[0:s, s + k:len(t)]
            return t[keep], y[keep]
        return f

    def drop_last(k):
        return lambda t, y, t0: (t[:-k], y[:-k])

    def dup(k):
        def f(t, y, t0):
            j = rng.choice(len(t), k, replace=False)
            return np.r_[t, t[j]], np.r_[y, y[j]]
        return f

    def shuffle5(t, y, t0):
        j = rng.choice(len(t), 5, replace=False)
        p = j[rng.permutation(5)]
        t2, y2 = t.copy(), y.copy()
        t2[j], y2[j] = t[p], y[p]
        order = rng.permutation(len(t))
        return t2[order], y2[order]

    def nan_ts(t, y, t0):
        t2 = t.copy()
        t2[rng.randint(1, len(t) - 1)] = np.nan
        return t2, y

    inj = {f"P_missing_mid{k}": drop_mid(k) for k in (1, 3, 6, 7)}
    inj.update({f"P_missing_last{k}": drop_last(k) for k in (1, 2, 3, 4)})
    inj.update({f"P_duplicate{k}": dup(k) for k in (1, 3, 5)})
    inj["P_shuffle5"] = shuffle5
    inj["P_nan_timestamp"] = nan_ts
    return inj


def run_preproc(runner, name, fn):
    b, fr0 = runner.b, runner.fr
    fr = {k: v.copy() for k, v in fr0.items()}
    pre = np.zeros(len(b), np.int32)
    for i in range(len(b)):
        grid = np.asarray(b.t_hist[i], float)
        t, y = fn(grid.copy(), np.asarray(b.y_hist[i], float).copy(), float(b.t0[i]))
        yy, t_last = assemble(grid, t, y)
        pre[i] = preproc_status(yy, t_last, float(b.t0[i]))
        if pre[i] == pf2.OK:
            fr["y"][i] = yy
            fr["iner"][i] = yy[-1]
            fr["t_last_s"][i] = int(round(t_last * 60))
    return runner.forward(fr, pre), fr


def frame_injections(rng):
    """name -> function(fr, i) modifying the frame of window i in place."""
    def cgm(vals):
        def f(fr, i):
            j = rng.choice(72, len(vals), replace=False)
            fr["y"][i, j] = vals
        return f

    def nan_cgm(fr, i):
        fr["y"][i, rng.randint(72)] = np.nan

    def first_active(fr, i):
        a = np.flatnonzero(fr["mask"][i] == 1)
        return int(a[0]) if len(a) else None

    def set_mark(col, val):
        def f(fr, i):
            j = first_active(fr, i)
            if j is not None:
                fr["feat"][i, j, col] = val
        return f

    def carb500(fr, i):
        j = first_active(fr, i)
        if j is not None:
            fr["feat"][i, j, 0] = 0.0
            fr["feat"][i, j, 1] = np.float32(np.log1p(500.0))

    inj = {"F_cgm_limit_38.5": cgm([38.5]), "F_cgm_limit_39x3": cgm([39.0, 39.0, 39.0]),
           "F_cgm_limit_401": cgm([401.0]), "F_cgm_limit_401.5x3": cgm([401.5, 401.5, 401.5]),
           "F_cgm_out_0": cgm([0.0]), "F_cgm_out_20": cgm([20.0]), "F_cgm_out_600": cgm([600.0]),
           "F_cgm_out_1000": cgm([1000.0]), "F_cgm_nan": nan_cgm,
           "F_dose_negative": set_mark(1, np.float32(np.log1p(0.0))),  # pack_marks maps a<0 to log1p(0)=0
           "F_dose_nan": set_mark(1, np.nan), "F_type_7": set_mark(0, 7.0),
           "F_carb_500g": carb500,
           "F_elapsed_negative": set_mark(2, -0.1)}
    return inj


def run_frame(runner, fn):
    fr = {k: v.copy() for k, v in runner.fr.items()}
    for i in range(len(fr["y"])):
        fn(fr, i)
    return runner.forward(fr, np.zeros(len(fr["y"]), np.int32)), fr


def mark_batch(b, rng, kind, frac=None, n_copy=None):
    mt, ma, me, nm = [], [], [], []
    for i in range(len(b)):
        t, a, e = np.asarray(b.mark_t[i]), np.asarray(b.mark_a[i]), np.asarray(b.mark_e[i], dtype=object)
        if kind == "remove":
            sel = np.flatnonzero(e == frac[0])
            k = int(round(len(sel) * frac[1]))
            drop = rng.choice(sel, k, replace=False) if k else np.array([], int)
            keep = np.setdiff1d(np.arange(len(t)), drop)
            t, a, e = t[keep], a[keep], e[keep]
        elif kind == "empty":
            t, a, e = t[:0], a[:0], e[:0]
        elif kind == "excess" and len(t):
            r = rng.choice(len(t), n_copy, replace=True)
            t0 = float(b.t0[i])
            t = np.r_[t, t0 - rng.uniform(0, 240, n_copy)]
            a, e = np.r_[a, a[r]], np.r_[e, e[r]]
        mt.append(t), ma.append(a), me.append(e), nm.append(len(t))
    return replace(b, mark_t=mt, mark_a=ma, mark_e=me, n_marks=np.asarray(nm))


def run_marks(runner, b2):
    fr = pf2.frames(b2)
    return runner.forward(fr, np.zeros(len(b2), np.int32)), fr


def full_holdout_mark_removal(full, net, y_mean, y_std, thr, rng):
    lab = pf2.labels(full)
    yf = np.asarray(full.y_future, float)
    hz = np.asarray(full.horizons, float)
    rows = []

    def metrics(b2, name):
        fr = pf2.frames(b2)
        r = Runner.__new__(Runner)
        r.b, r.net, r.y_mean, r.y_std, r.thr = b2, net, y_mean, y_std, thr
        res = Runner.forward(r, fr, np.zeros(len(b2), np.int32))
        ok = res["status"] == pf2.OK
        err = np.abs(res["yhat"] - yf)
        p, y = res["evt"][ok, 3], lab[ok, 3]
        return {"variant": name, "n_ok": int(ok.sum()), "mae_1h": float(np.nanmean(err[ok][:, hz <= 60])),
                "mae_4h": float(np.nanmean(err[ok])),
                "hypo70_4h_fnr": float(((p == 0) & (y == 1)).sum() / max((y == 1).sum(), 1)),
                "hypo70_4h_fpr": float(((p == 1) & (y == 0)).sum() / max((y == 0).sum(), 1)),
                "g_mean": float(np.nanmean(res["g"][ok])), "evt": res["evt"]}

    base = metrics(full, "clean")
    rows.append(base)
    for typ in ("meal", "bolus"):
        for frac in (0.25, 0.5, 1.0):
            m = metrics(mark_batch(full, rng, "remove", (typ, frac)), f"remove_{typ}_{int(frac * 100)}")
            rows.append(m)
    m = metrics(mark_batch(full, rng, "empty"), "remove_all")
    rows.append(m)
    for r in rows:
        r["evt_cells_changed_vs_clean"] = int((r.pop("evt") != base["evt"]).sum()) if r is not base else 0
    base.pop("evt", None)
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(SEED)
    full = pf2.load_cache(CACHE)["test"]
    idx = np.asarray(np.load(SAMPLE512)["idx"], int)
    b = sub_batch(full, idx)
    net, y_mean, y_std, thr = pf2.load_model(CKPT)
    runner = Runner(b, net, y_mean, y_std, thr)
    rows = [runner.report("clean", "—", runner.clean, runner.fr)]
    for name, fn in preproc_injections(b, rng).items():
        res, fr = run_preproc(runner, name, fn)
        rows.append(runner.report(name, "preprocessing", res, fr))
        print(rows[-1], flush=True)
    for name, fn in frame_injections(rng).items():
        res, fr = run_frame(runner, fn)
        rows.append(runner.report(name, "frame", res, fr))
        print(rows[-1], flush=True)
    for name, b2 in (("M_remove_meal_50", mark_batch(b, rng, "remove", ("meal", 0.5))),
                     ("M_remove_bolus_50", mark_batch(b, rng, "remove", ("bolus", 0.5))),
                     ("M_empty", mark_batch(b, rng, "empty")),
                     ("M_excess_60", mark_batch(b, rng, "excess", n_copy=60))):
        res, fr = run_marks(runner, b2)
        rows.append(runner.report(name, "marks", res, fr))
        print(rows[-1], flush=True)
    full_rows = full_holdout_mark_removal(full, net, y_mean, y_std, thr, rng)
    for r in full_rows:
        print(r, flush=True)
    import pandas as pd
    pd.DataFrame(rows).to_csv(OUT / "tb_host.csv", index=False)
    pd.DataFrame(full_rows).to_csv(OUT / "tb_host_mark_removal_full.csv", index=False)
    (OUT / "tb_host.json").write_text(json.dumps({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "seed": SEED,
                                                  "windows": "sample512", "rows": rows, "full_holdout": full_rows},
                                                 indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
