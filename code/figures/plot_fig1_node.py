#!/usr/bin/env python3
"""Revised Fig. 1: four-layer structure of the Pico-Former node. Run in the streamlit3.13 environment.

Layout follows fig1() in scripts/make_figures.py of the first manuscript draft; the curve-head formula matches
Sec. 3.2 and is written as y_iner + g·δ.
Output: ../../figures/fig1-device-inference-contract.{svg,png}
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams["svg.fonttype"] = "path"

OUT = Path(__file__).resolve().parents[2] / "figures"
STEM = "fig1-device-inference-contract"


def box(ax, xy, w, h, text, fc="#E8F1F8", ec="#1F4E79", fontsize=8.0):
    p = FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.02,rounding_size=0.08", linewidth=1.2, facecolor=fc, edgecolor=ec)
    ax.add_patch(p)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=fontsize, color="#1a1a1a", linespacing=1.15)


def main() -> None:
    fig, ax = plt.subplots(figsize=(11.2, 5.05))
    ax.set_xlim(0, 11.2)
    ax.set_ylim(0, 5.05)
    ax.axis("off")
    for x, lab in ((1.18, "Input"), (3.72, "Model"), (6.85, "Decision"), (10.15, "Connection")):
        ax.text(x, 4.78, lab, ha="center", va="center", fontsize=9.5, color="#1a1a1a", fontweight="bold")
    box(ax, (0.15, 1.575), 2.10, 1.75, "Window at $t_0$\n72-pt CGM (6 h)\nup to 24 marks\n(meal / insulin)", fc="#F4F4F4", ec="#555555")
    box(ax, (2.55, 0.90), 2.35, 3.10, "96-token encoder\n$d{=}48$, 2 layers\nself- + cross-attn\nCore 1, FP32 SRAM", fc="#D6EAF8", ec="#1F4E79")
    box(ax, (5.25, 2.75), 2.15, 1.25, "Curve head\n$\\hat y = y_{\\mathrm{iner}}+g\\,\\delta$\n48 points", fc="#D5F5E3", ec="#1B7A4A")
    box(ax, (5.25, 0.90), 2.15, 1.25, "Event head\n16 logits\n$\\ell \\geq \\tau$ (Flash)", fc="#FDEBD0", ec="#B9770E")
    box(ax, (7.75, 2.75), 2.15, 1.25, "Trajectory\n$\\hat y \\in \\mathbb{R}^{48}$", fc="#E8F8F0", ec="#1B7A4A")
    box(ax, (7.75, 0.90), 2.15, 1.25, "Event chips\n$\\ell\\geq\\tau$, $4\\times4$", fc="#FEF5E7", ec="#B9770E")
    box(ax, (10.05, 2.50), 1.00, 1.25, "Core 0\nWi-Fi\nHTTP(S)", fc="#F5EEF8", ec="#6C3483")
    box(ax, (10.05, 0.90), 1.00, 1.25, "Serial\naudit\nframes", fc="#F5EEF8", ec="#6C3483")
    for x0, x1, y in ((2.25, 2.55, 2.45), (4.90, 5.25, 3.375), (4.90, 5.25, 1.525), (7.40, 7.75, 3.375), (7.40, 7.75, 1.525)):
        ax.annotate("", xy=(x1, y), xytext=(x0, y), arrowprops=dict(arrowstyle="-|>", color="#333333", lw=1.3))
    ax.annotate("", xy=(10.05, 3.125), xytext=(9.90, 3.125), arrowprops=dict(arrowstyle="-", color="#888888", lw=1.0, ls=":"))
    ax.annotate("", xy=(10.05, 1.525), xytext=(9.90, 1.525), arrowprops=dict(arrowstyle="-", color="#888888", lw=1.0, ls=":"))
    ax.text(5.60, 0.32, r"Generation is direct multi-step. Thresholds $\tau$ live in Flash.", fontsize=8, color="#444444")
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{STEM}.svg", bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / f"{STEM}.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", OUT / f"{STEM}.svg")


if __name__ == "__main__":
    main()
