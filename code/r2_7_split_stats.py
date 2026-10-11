#!/usr/bin/env python3
"""R2-7 (also R1-3, R2-17): participants, window counts, prevalence of the 16 event outputs, and mark-count
distribution for each split of each seed.

Data: release caches of the three AZT1D seeds (train/val/test), and the OhioT1DM and DiaTrend external test caches.
Mark count is WindowBatch.n_marks (total marks in the 4 h look-back before clamping); the encoding keeps the
most recent 24.
Outputs: results/r2_7/split_stats.csv, split_stats.json
"""

from __future__ import annotations

import os
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(os.environ.get("PF_WORKSPACE", "."))
SRC = BASE / "rice-former/src"
RUNS = BASE / "runs"
HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "r2_7"
EXT = HERE / "results" / "external_cache"
MAX_MARKS = 24

sys.path.insert(0, str(SRC))
from ceqt import EVENT_SPECS, PH_WINDOWS  # noqa: E402
from ceqt.metrics import event_labels  # noqa: E402

NAMES = [f"{e}_{ph // 60}h" for e, _, _ in EVENT_SPECS for ph in PH_WINDOWS]


def sources():
    for seed in (20260828, 20260829, 20260830):
        yield f"azt1d_s{seed}", RUNS / f"20260830_s{seed}/cache/azt1d_s{seed}_str15.pkl"
    yield "ohio", EXT / "ohio_str15.pkl"
    yield "diatrend17", EXT / "diatrend17_str15.pkl"


def stats(cohort, split, b):
    y = event_labels(b.y_future, b.horizons).astype(int)
    subj = np.asarray(b.subject_id)
    nm = np.asarray(b.n_marks) if len(b.n_marks) == len(b) else np.array([len(m) for m in b.mark_t])
    row = {"cohort": cohort, "split": split, "n_subjects": int(len(np.unique(subj))),
           "subjects": " ".join(str(s) for s in sorted(np.unique(subj), key=lambda s: (len(str(s)), str(s)))),
           "n_windows": int(len(b)),
           "marks_gt24_pct": float(100 * (nm > MAX_MARKS).mean()),
           "marks_zero_pct": float(100 * (nm == 0).mean()),
           "marks_median": float(np.median(nm))}
    for j, n in enumerate(NAMES):
        row[f"prev_{n}"] = float(100 * y[:, j].mean())
        row[f"pos_{n}"] = int(y[:, j].sum())
    return row


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for cohort, path in sources():
        d = pickle.loads(path.read_bytes())
        for split in ("train", "val", "test"):
            if split in d:
                r = stats(cohort, split, d[split])
                rows.append(r)
                print(f"{cohort:16s} {split:5s} subj={r['n_subjects']:3d} win={r['n_windows']:7d} "
                      f"hypo70_4h={r['prev_hypo70_4h']:.1f}% hyper180_4h={r['prev_hyper180_4h']:.1f}% "
                      f">24={r['marks_gt24_pct']:.1f}% zero={r['marks_zero_pct']:.1f}%", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "split_stats.csv", index=False)
    (OUT / "split_stats.json").write_text(json.dumps(
        {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "rows": rows}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
