#!/usr/bin/env python3
"""R2-6 Fig. 2 (revised): FPR-FNR operating points of host and board event heads at frozen thresholds,
16 outputs x 5 full sets.

Host is the float64 reference (same input checks as the node; ref_logits and thr in audit.npz); board is the
decision bits returned by the node (evt); labels come from results/r2_4_r2_5/<cohort>_scores.npz.
The figure shows host-board agreement only and excludes the trajectory-threshold readout.
Outputs: fig2-host-board-event-heads.svg and .pdf in code/results/figures/
(see IMG); values are written to results/r2_6/fig2_points.json
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

CODE = Path(__file__).resolve().parents[1]
RES = CODE / "results"
IMG = CODE / "results" / "figures"
COHORTS = [("AZT1D, model 1 (seed 20260828)", "azt1d_s20260828", "azt1d_full", "WiFiduino"),
           ("AZT1D, model 2 (seed 20260829)", "azt1d_s20260829", "s20260829_full", "WiFiduino"),
           ("AZT1D, model 3 (seed 20260830)", "azt1d_s20260830", "s20260830_full", "WiFiduino"),
           ("OhioT1DM, model 1", "ohio", "ohio_full", "WiFiduino"),
           ("DiaTrend, model 1", "diatrend17", "devkitc_diatrend17_full", "Lvshen")]
EVTS = ["hypo70", "hypo54", "hyper180", "hyper250"]
COLOR = {"hypo70": "#4C72B0", "hypo54": "#55A868", "hyper180": "#DD8452", "hyper250": "#C44E52"}
SIZE = {1: 16, 2: 30, 3: 46, 4: 64}

plt.rcParams["svg.fonttype"] = "path"
plt.rcParams["font.size"] = 8


def rates(pred, y):
    fn = (~pred & y).sum(0)
    fp = (pred & ~y).sum(0)
    return fn / np.maximum(y.sum(0), 1), fp / np.maximum((~y).sum(0), 1)


def main():
    fig, grid = plt.subplots(2, 3, figsize=(7.16, 4.6), sharex=True, sharey=True, constrained_layout=True)
    axes = grid.ravel()
    pts = {}
    for ax, (title, cohort, run, board_name) in zip(axes, COHORTS):
        s = np.load(RES / "r2_4_r2_5" / f"{cohort}_scores.npz")
        a = np.load(RES / "audit" / run / "audit.npz")
        y = s["y"].astype(bool)
        host = a["ref_logits"] >= a["thr"][None, :]
        board = a["evt"].astype(bool)
        fnr_h, fpr_h = rates(host, y)
        fnr_b, fpr_b = rates(board, y)
        pts[cohort] = dict(board=board_name, windows=int(len(y)), cells_differ=int((host != board).sum()),
                           fnr_host=fnr_h.tolist(), fpr_host=fpr_h.tolist(),
                           fnr_board=fnr_b.tolist(), fpr_board=fpr_b.tolist())
        for k in range(16):
            e, h = EVTS[k // 4], k % 4 + 1
            ax.scatter(fpr_h[k], fnr_h[k], s=SIZE[h], color=COLOR[e], lw=0, zorder=2)
            ax.scatter(fpr_b[k], fnr_b[k], s=SIZE[h] + 22, facecolors="none", edgecolors="k", lw=0.7, marker="D",
                       zorder=3)
        ax.axvline(0.06, color="0.5", ls="--", lw=0.7)
        ax.set_title(f"{title}\n{board_name} board\n{len(y):,} windows, {int((host != board).sum())} cells differ",
                     fontsize=7)
        ax.set_xlim(0, 0.085)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.25, lw=0.5)
    for ax in (grid[1, 0], grid[1, 1], grid[0, 2]):
        ax.set_xlabel("False-positive rate")
        ax.xaxis.set_tick_params(labelbottom=True)
    for ax in grid[:, 0]:
        ax.set_ylabel("False-negative rate")
    handles = [plt.Line2D([], [], color=COLOR[e], marker="o", ls="", ms=5, label=e) for e in EVTS]
    handles += [plt.Line2D([], [], color="0.4", marker="o", ls="", ms=ms, label=f"{h} h")
                for h, ms in ((1, 3), (2, 4.2), (3, 5.2), (4, 6.2))]
    handles += [plt.Line2D([], [], color="k", marker="D", mfc="none", ls="", ms=6, label="board"),
                plt.Line2D([], [], color="0.4", marker="o", ls="", ms=5, label="host"),
                plt.Line2D([], [], color="0.5", ls="--", lw=0.7, label="0.06 validation FPR budget")]
    leg_ax = axes[-1]
    leg_ax.axis("off")
    leg_ax.legend(handles=handles, loc="center", ncol=2, fontsize=7, frameon=False)
    IMG.mkdir(parents=True, exist_ok=True)
    fig.savefig(IMG / "fig2-host-board-event-heads.svg")
    fig.savefig(IMG / "fig2-host-board-event-heads.pdf")
    (RES / "r2_6").mkdir(exist_ok=True)
    (RES / "r2_6" / "fig2_points.json").write_text(json.dumps(pts, indent=1))
    print({c: v["cells_differ"] for c, v in pts.items()})


if __name__ == "__main__":
    main()
