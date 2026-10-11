#!/usr/bin/env python3
"""R2-17 T-A: distribution of input states at every 5 min tick in real recordings (under the R1-R7 input checks).

Data: all 25 AZT1D participants, 12 OhioT1DM test participants, 17 DiaTrend participants (release-package loaders).
Each participant's CGM series is resampled to a 5 min grid by the release-package rule (nearest observation
within 15 min, otherwise gap), and each tick from 6 h after the recording start is classified, in priority order:
  STALE   no CGM within 15 min of the tick (R4);
  GAP     a gap in the 6 h history (72 points) (R3);
  RANGE   a history value outside 40-400 mg/dL by 2 mg/dL or more (R1; inference rejected; same status code as GAP,
          counted separately here);
  OK      inference allowed; ticks whose history contains 38-40 or 400-402 mg/dL limit readings are clamped to
          40/400 and counted as CLAMPED; among OK ticks, also counted are mark count > 24 in the 4 h look-back
          (R7 clamp) and doses over limit or non-positive (R5; mark removed, g = 0.45).
In deployment the node runs inference at every tick and needs no future values, so there are more inferable ticks
than training/evaluation windows (which also require a complete 4 h future and a 15 min stride).
Outputs: results/r2_17/ta_input_states.csv (per participant), ta_summary.json
"""

from __future__ import annotations

import os
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(os.environ.get("PF_WORKSPACE", "."))
sys.path.insert(0, str(BASE / "rice-former/src"))
from ceqt.data.azt1d import list_azt1d_subjects, load_azt1d_subject  # noqa: E402
from ceqt.data.diatrend import load_diatrend_basal17  # noqa: E402
from ceqt.data.ohio import load_ohio_test  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "r2_17"
DT, N_HIST, MAX_GAP, LOOKBACK, N_MARK = 5.0, 72, 15.0, 240.0, 24
CGM_MIN, CGM_MAX, CGM_TOL = 40.0, 400.0, 2.0
LIMIT = {"meal": 250.0, "bolus": 30.0, "correction": 30.0, "basal": 35.0}


def grid_series(t, y):
    ok = np.isfinite(t) & np.isfinite(y)
    t, y = t[ok], y[ok]
    order = np.argsort(t, kind="stable")
    t, y = t[order], y[order]
    n_dup = int((np.diff(t) == 0).sum())
    grid = np.arange(t[0], t[-1] + 1e-6, DT)
    j = np.clip(np.searchsorted(t, grid), 1, len(t) - 1)
    left, right = t[j - 1], t[j]
    pick = np.where(np.abs(grid - left) <= np.abs(right - grid), j - 1, j)
    out = np.where(np.abs(t[pick] - grid) <= MAX_GAP, y[pick], np.nan)
    return grid, out, n_dup


def subject_states(cohort, rec):
    t, y, n_dup = grid_series(np.asarray(rec.t_min, float), np.asarray(rec.cgm, float))
    n = len(t)
    idx = np.arange(N_HIST - 1, n)
    fin = np.isfinite(y)
    rng = fin & (y >= CGM_MIN - CGM_TOL) & (y <= CGM_MAX + CGM_TOL)
    inr = fin & (y >= CGM_MIN) & (y <= CGM_MAX)
    cs_cl = np.concatenate([[0], np.cumsum(rng & ~inr)])
    cs_nf = np.concatenate([[0], np.cumsum(~fin)])
    cs_nr = np.concatenate([[0], np.cumsum(fin & ~rng)])
    nf = cs_nf[idx + 1] - cs_nf[idx + 1 - N_HIST]
    nr = cs_nr[idx + 1] - cs_nr[idx + 1 - N_HIST]
    stale = ~fin[idx]
    gap = ~stale & (nf > 0)
    rge = ~stale & ~gap & (nr > 0)
    ok = ~stale & ~gap & ~rge
    clamped = ok & ((cs_cl[idx + 1] - cs_cl[idx + 1 - N_HIST]) > 0)

    mt = np.array([m.t_min for m in rec.marks], float)
    order = np.argsort(mt)
    mt = mt[order]
    bad = np.array([not (np.isfinite(m.a) and 0 < m.a <= LIMIT.get(m.e, -1)) for m in rec.marks], bool)[order]
    t0 = t[idx]
    lo, hi = np.searchsorted(mt, t0 - LOOKBACK, "right"), np.searchsorted(mt, t0, "right")
    n_marks = hi - lo
    cs_bad = np.concatenate([[0], np.cumsum(bad)])
    n_bad = cs_bad[hi] - cs_bad[lo]
    return {
        "cohort": cohort, "subject": int(rec.subject_id), "ticks": int(len(idx)),
        "ok": int(ok.sum()), "stale": int(stale.sum()), "gap": int(gap.sum()), "range": int(rge.sum()),
        "ok_clamped": int(clamped.sum()),
        "ok_marks_gt24": int((ok & (n_marks > N_MARK)).sum()),
        "ok_marks_zero": int((ok & (n_marks == 0)).sum()),
        "ok_dose_dropped": int((ok & (n_bad > 0)).sum()),
        "cgm_limit_points": int((rng & ~inr).sum()),
        "cgm_out_of_range_points": int((fin & ~rng).sum()), "cgm_points": int(fin.sum()),
        "dup_timestamps": n_dup, "marks": int(len(mt)), "marks_over_limit": int(bad.sum()),
    }


def records():
    for s in list_azt1d_subjects():
        yield "AZT1D", load_azt1d_subject(s)
    for r in load_ohio_test():
        yield "OhioT1DM", r
    for r in load_diatrend_basal17():
        yield "DiaTrend", r


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for cohort, rec in records():
        r = subject_states(cohort, rec)
        rows.append(r)
        print(f"{cohort:9s} {r['subject']:4d} ticks={r['ticks']:6d} ok={r['ok'] / r['ticks']:.3f} "
              f"stale={r['stale'] / r['ticks']:.3f} gap={r['gap'] / r['ticks']:.3f} range={r['range']}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "ta_input_states.csv", index=False)
    summ = {"time": time.strftime("%Y-%m-%d %H:%M:%S")}
    for cohort, g in df.groupby("cohort", sort=False):
        s = g.sum(numeric_only=True)
        frac = g["ok"] / g["ticks"]
        summ[cohort] = {
            "subjects": int(len(g)), "ticks": int(s.ticks),
            **{f"{k}_pct": float(100 * s[k] / s.ticks) for k in ("ok", "stale", "gap", "range")},
            **{f"{k}_pct_of_ok": float(100 * s[k] / s.ok) for k in ("ok_clamped", "ok_marks_gt24", "ok_marks_zero", "ok_dose_dropped")},
            "ok_pct_subject_min": float(100 * frac.min()), "ok_pct_subject_max": float(100 * frac.max()),
            "cgm_limit_points": int(s.cgm_limit_points),
            "cgm_out_of_range_points": int(s.cgm_out_of_range_points), "cgm_points": int(s.cgm_points),
            "dup_timestamps": int(s.dup_timestamps), "marks": int(s.marks), "marks_over_limit": int(s.marks_over_limit),
        }
    (OUT / "ta_summary.json").write_text(json.dumps(summ, indent=1, ensure_ascii=False))
    print(json.dumps(summ, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
