#!/usr/bin/env python3
"""R2-2 figure: ROC and PR curves of the four event types within the 4 h window (event-head logit and trajectory
readout), marking the checkpoint thresholds and the trajectory cut-offs matched to validation FPR.

Inputs: results/r2_2/scores.npz, r2_2_matched_fpr.json
Output: r2-2-roc-pr.svg in code/results/figures/ (see OUT)
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import precision_recall_curve, roc_curve  # noqa: E402

HERE = Path(__file__).resolve().parent
RES = HERE / "results" / "r2_2"
OUT = HERE / "results" / "figures" / "r2-2-roc-pr.svg"
EVENTS = [("hypo70", "Glucose < 70 mg/dL"), ("hypo54", "Glucose < 54 mg/dL"),
          ("hyper180", "Glucose > 180 mg/dL"), ("hyper250", "Glucose > 250 mg/dL")]
PH = [60, 120, 180, 240]

plt.rcParams.update({"svg.fonttype": "path", "font.family": "DejaVu Sans", "font.size": 8})


def main():
    z = np.load(RES / "scores.npz")
    rows = {(r["event"], r["ph_min"]): r for r in json.loads((RES / "r2_2_matched_fpr.json").read_text())}
    lt, tt, yt, tau = z["logits_test"], z["traj_test"], z["y_test"], z["tau"]
    fig, axes = plt.subplots(2, 4, figsize=(7.16, 3.9), constrained_layout=True)
    for j, (ev, title) in enumerate(EVENTS):
        k = j * len(PH) + PH.index(240)
        y, sh, st = yt[:, k], lt[:, k], tt[:, k]
        r = rows[(ev, 240)]
        cut = r["OP1"]["traj_cut_mgdl"]
        cut_s = -cut if ev.startswith("hypo") else cut
        ax = axes[0, j]
        for s, lab, c in ((sh, "Event head", "#1f4e79"), (st, "Trajectory", "#c55a11")):
            f, t, _ = roc_curve(y, s)
            ax.plot(f, t, color=c, lw=1.1, label=lab)
        ax.plot([0, 1], [0, 1], color="0.75", lw=0.6, ls="--")
        oh, ot = r["OP0"]["head"], r["OP1"]["traj"]
        ax.plot(oh["fpr"], 1 - oh["fnr"], "o", ms=4, color="#1f4e79")
        ax.plot(ot["fpr"], 1 - ot["fnr"], "s", ms=4, color="#c55a11")
        ax.set_title(f"{title}, 4 h\nAUROC {r['auroc']['head']:.3f} / {r['auroc']['traj']:.3f}", fontsize=7.5)
        ax.set_xlabel("False positive rate")
        if j == 0:
            ax.set_ylabel("True positive rate")
            ax.legend(loc="lower right", fontsize=6.5, frameon=False)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax = axes[1, j]
        for s, c, thr in ((sh, "#1f4e79", tau[k]), (st, "#c55a11", cut_s)):
            p, rc, _ = precision_recall_curve(y, s)
            ax.plot(rc, p, color=c, lw=1.1)
            pred = s >= thr
            tp = (pred & (y == 1)).sum()
            ax.plot(tp / max(y.sum(), 1), tp / max(pred.sum(), 1), "o" if c == "#1f4e79" else "s", ms=4, color=c)
        ax.axhline(y.mean(), color="0.75", lw=0.6, ls="--")
        ax.set_title(f"AUPRC {r['auprc']['head']:.3f} / {r['auprc']['traj']:.3f}", fontsize=7.5)
        ax.set_xlabel("Recall")
        if j == 0:
            ax.set_ylabel("Precision")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT)
    print(OUT)


if __name__ == "__main__":
    main()
