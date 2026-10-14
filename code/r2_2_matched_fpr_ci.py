#!/usr/bin/env python3
"""R2-2: participant-clustered 95% CI of the miss-rate difference at matched held-out FPR, 16 outputs.

Reads results/r2_2/scores.npz written by r2_2_matched_fpr.py (held-out event-head logits, trajectory scores,
labels, participant IDs, and frozen tau of the deployed checkpoint).
Difference = trajectory FNR read from its held-out ROC at the event head's held-out FPR, minus the event-head
FNR at frozen tau (the "Difference (curve - head)" column of Table S4a).
Each bootstrap resample draws the held-out participants with replacement and recomputes the event-head FPR,
both FNRs, and the difference on that resample. Resamples, seed, and count follow r2_2_matched_fpr.py.
Output: results/r2_2/matched_fpr_fnr_ci.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_curve

OUT = Path(__file__).resolve().parent / "results" / "r2_2"
N_BOOT = 2000
BOOT_SEED = 20261009
EVENTS = ["hypo70", "hypo54", "hyper180", "hyper250"]
HORIZONS_H = [1, 2, 3, 4]


def fnr_difference(y, logit, traj, tau):
    pred = logit >= tau
    fnr_head = ((~pred) & (y == 1)).sum() / max((y == 1).sum(), 1)
    fpr_head = (pred & (y == 0)).sum() / max((y == 0).sum(), 1)
    fpr, tpr, _ = roc_curve(y, traj)
    return float(1.0 - np.interp(fpr_head, fpr, tpr) - fnr_head)


def main():
    d = np.load(OUT / "scores.npz")
    lt, tt, yt, subj, tau = d["logits_test"], d["traj_test"], d["y_test"], d["subject_test"], d["tau"]
    subjects = np.unique(subj)
    rng = np.random.default_rng(BOOT_SEED)
    boots = [np.concatenate([np.where(subj == s)[0] for s in rng.choice(subjects, len(subjects))])
             for _ in range(N_BOOT)]
    rows = []
    for k in range(16):
        y = yt[:, k]
        diffs = []
        for idx in boots:
            yy = y[idx]
            if yy.sum() == 0 or yy.sum() == len(yy):
                continue
            diffs.append(fnr_difference(yy, lt[idx, k], tt[idx, k], tau[k]))
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        rows.append({"event": EVENTS[k // 4], "horizon_h": HORIZONS_H[k % 4], "n_pos_test": int(y.sum()),
                     "fnr_curve_minus_head": fnr_difference(y, lt[:, k], tt[:, k], tau[k]),
                     "ci95": [float(lo), float(hi)], "n_boot_valid": len(diffs),
                     "n_participants": int(len(subjects))})
        r = rows[-1]
        print(f"{r['event']} {r['horizon_h']} h: {r['fnr_curve_minus_head']:+.3f} "
              f"[{lo:+.3f}, {hi:+.3f}] (n={len(diffs)})")
    (OUT / "matched_fpr_fnr_ci.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
