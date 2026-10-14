#!/usr/bin/env python3
"""Fig. 3: golden window (subject 14) with both readouts.

A: CGM history, held-out future, and the predicted path yhat. B: the sixteen event outputs, on when logit >= tau.
Data: gold/window0.json, written by host/export_golden.py from the deployed checkpoint and the AZT1D cache (holds
participant data and is not distributed; set PICO_GOLD_DIR to read it from another directory).
Output: figures/fig3-golden-window-dual-readout.{svg,png}.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
GOLD = Path(os.environ.get("PICO_GOLD_DIR", ROOT / "gold")).expanduser() / "window0.json"
OUT = ROOT / "figures" / "fig3-golden-window-dual-readout"
EVT = ["hypo70", "hypo54", "hyper180", "hyper250"]
PH = ["1h", "2h", "3h", "4h"]

plt.rcParams.update({
    "svg.fonttype": "path",
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 11,
    "figure.facecolor": "white",
})


def main() -> None:
    gold = json.loads(GOLD.read_text())
    y_hist = np.asarray(gold["y_hist"], dtype=float)
    yhat = np.asarray(gold["yhat"], dtype=float)
    yfut = np.asarray(gold["y_future"], dtype=float)
    on = np.asarray(gold["logits"], dtype=float) >= np.asarray(gold["thresholds"], dtype=float)
    t_hist = np.arange(-71, 1) * 5
    t_fut = np.arange(1, 49) * 5

    fig = plt.figure(figsize=(10.6, 5.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.55, 1.0], wspace=0.28)
    ax = fig.add_subplot(gs[0])
    ax.plot(t_hist, y_hist, color="#555555", lw=1.6, label="Observed CGM")
    ax.plot(t_fut, yfut, color="#555555", lw=1.2, ls=":", label="Held-out future")
    ax.plot(t_fut, yhat, color="#0072B2", lw=2.0, label=r"Pico-Former $\hat y$")
    ax.axhline(70, color="#C0392B", ls="--", lw=0.9)
    ax.axhline(180, color="#B9770E", ls="--", lw=0.9)
    ax.axvline(0, color="#bbbbbb", lw=0.8)
    ax.set_xlabel("Minutes from prediction origin")
    ax.set_ylabel("Glucose (mg/dL)")
    ax.set_title(f"A  Golden window (subject {gold['subject_id']})")
    ax.set_ylim(40, 220)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.grid(alpha=0.25)
    ax.text(5, 74, "70", color="#C0392B", fontsize=7)
    ax.text(5, 184, "180", color="#B9770E", fontsize=7)

    ax = fig.add_subplot(gs[1])
    ax.set_xlim(-0.5, 4.5)
    ax.set_ylim(-0.5, 4.5)
    ax.set_xticks(range(4), PH)
    ax.set_yticks(range(4), EVT[::-1])
    ax.set_title(r"B  Frozen $\ell\geq\tau$ chips")
    ax.tick_params(length=0)
    for i in range(len(EVT)):
        for j in range(4):
            hit = bool(on[i * 4 + j])
            y = 3 - i
            fc = "#F5B7B1" if hit else "#D5F5E3"
            ec = "#922B21" if hit else "#196F3D"
            ax.add_patch(mpatches.FancyBboxPatch((j - 0.42, y - 0.42), 0.84, 0.84,
                                                 boxstyle="round,pad=0.02,rounding_size=0.08",
                                                 facecolor=fc, edgecolor=ec, linewidth=1.1))
            ax.text(j, y, "on" if hit else "off", ha="center", va="center", fontsize=8, color=ec)
    ax.set_aspect("equal")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", OUT.with_suffix(".svg"))
    print("on:", [f"{EVT[k // 4]} {k % 4 + 1} h" for k in range(16) if on[k]],
          "min yhat", round(float(yhat.min()), 1))


if __name__ == "__main__":
    main()
